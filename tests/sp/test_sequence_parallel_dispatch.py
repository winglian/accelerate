# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-module SP dispatch: a hybrid model routes each mixer to the strategy for its mechanism —
attention -> Ulysses (registry-key fast path), Mamba/SSM -> state passing (per-module wrap) — and the
state-passing scan matches a global scan under a 2-rank gloo group.
"""

import os
import queue as queue_mod
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from accelerate.utils import sequence_parallel as sp


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- stub hybrid model
class _Config:
    def __init__(self, impl, n_kv):
        self._attn_implementation = impl
        self.num_key_value_heads = n_kv

    def get_text_config(self):
        return self


class _StubAttention(nn.Module):  # name matched by is_attention_module
    def __init__(self, config):
        super().__init__()
        self.config = config


class Mamba2Mixer(nn.Module):  # name matched by is_recurrent_mixer
    """Toy recurrent mixer: forward runs a linear scan via the hookable ``sp_scan`` (default: local)."""

    def __init__(self):
        super().__init__()
        self.sp_scan = sp.local_linear_scan

    def forward(self, a, b):
        return self.sp_scan(a, b)


class _HybridModel(nn.Module):
    def __init__(self, impl="_sp_test_base", n_kv=4):
        super().__init__()
        self.config = _Config(impl, n_kv)
        self.attn = _StubAttention(self.config)
        self.mixer = Mamba2Mixer()


@pytest.fixture()
def sp_world1():
    started = not dist.is_initialized()
    if started:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield dist.group.WORLD
    if started:
        dist.destroy_process_group()


@pytest.fixture()
def clean_registries():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    base_attn, base_hooks = dict(ALL_ATTENTION_FUNCTIONS), list(sp._SP_MODEL_HOOKS)
    base_bases = dict(sp._SP_ATTN_BASES)
    ALL_ATTENTION_FUNCTIONS["_sp_test_base"] = lambda module, q, k, v, mask, **kw: (q, None)
    try:
        yield ALL_ATTENTION_FUNCTIONS
    finally:
        ALL_ATTENTION_FUNCTIONS.clear()
        ALL_ATTENTION_FUNCTIONS.update(base_attn)
        sp._SP_MODEL_HOOKS[:] = base_hooks
        sp._SP_ATTN_BASES.clear()
        sp._SP_ATTN_BASES.update(base_bases)


# --------------------------------------------------------------------------- dispatch routing
def test_default_dispatch_classifies_by_mechanism():
    model = _HybridModel()
    assert isinstance(sp.default_sp_dispatch(model.attn), sp.UlyssesStrategy)
    assert isinstance(sp.default_sp_dispatch(model.mixer), sp.StatePassingStrategy)
    assert sp.default_sp_dispatch(nn.Linear(2, 2)) is None  # neither attention nor recurrent


def test_hybrid_model_routes_attention_and_mamba(sp_world1, clean_registries):
    all_attn = clean_registries
    model = _HybridModel()

    handler = sp.enable_sequence_parallel(model, sp_world1, dispatch=sp.default_sp_dispatch)

    # attention -> Ulysses via a unique registry key wrapping the base
    key = model.attn.config._attn_implementation
    assert key != "_sp_test_base" and all_attn[key] is handler
    assert isinstance(handler, sp.UlyssesAttention)

    # Mamba mixer -> per-module state-passing wrap (its scan hook is rebound onto the sp group)
    assert getattr(model.mixer, "_sp_group", None) is sp_world1
    assert model.mixer.sp_scan is not sp.local_linear_scan
    assert sp.StatePassingStrategy().requires_contiguous_shard is True


# --------------------------------------------------------------------------- state-passing scan
def _scan_worker(rank, world, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(0)  # every rank builds the SAME global a, b
        batch, seq, dim = 1, 8, 3
        a = torch.rand(batch, seq, dim) * 0.9 + 0.05  # decays in (0,1)
        b = torch.randn(batch, seq, dim)
        a_local = a.chunk(world, dim=1)[rank].contiguous()
        b_local = b.chunk(world, dim=1)[rank].contiguous()

        # wrap a toy mixer exactly as the dispatch would, then run it on the local shard
        mixer = Mamba2Mixer()
        sp.StatePassingStrategy().wrap_module(mixer, dist.group.WORLD)
        out_local = mixer(a_local, b_local)  # cp-corrected local outputs

        gathered = [torch.empty_like(out_local) for _ in range(world)]
        dist.all_gather(gathered, out_local.contiguous())
        if rank == 0:
            cp_full = torch.cat(gathered, dim=1)
            ref = sp.local_linear_scan(a, b)  # single-GPU global scan
            q.put(float((cp_full - ref).abs().max()))
    finally:
        dist.destroy_process_group()


def _spawn(target, world=2, get_timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=target, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    msgs = []
    deadline = time.monotonic() + get_timeout
    while time.monotonic() < deadline and len(msgs) < 1:
        try:
            msgs.append(q.get(timeout=0.5))
        except queue_mod.Empty:
            if all(p.exitcode is not None for p in procs):
                break
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    return msgs, [p.exitcode for p in procs]


def test_state_passing_scan_matches_global_scan():
    msgs, exitcodes = _spawn(_scan_worker)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    assert msgs[0] < 1e-5, f"CP scan diverged from global scan by {msgs[0]}"
