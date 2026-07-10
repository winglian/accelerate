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

"""Extensibility of native sequence parallelism:

* per-model attention install (enabling SP on one model never rebinds another's attention),
* the pluggable ``SequenceParallelStrategy`` seam,
* model hooks for non-attention mixers,
* varlen ``cu_seqlens`` self-derived from the sharded ``position_ids`` (no dataloader side channel).

The first three run in-process against a world=1 gloo group with tiny stub models; the last spawns a
2-rank gloo group.
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


# --------------------------------------------------------------------------- stub HF model
class _StubConfig:
    def __init__(self, impl, n_kv):
        self._attn_implementation = impl
        self.num_key_value_heads = n_kv

    def get_text_config(self):
        return self


class _StubModel(nn.Module):
    """Minimal stand-in exposing the surface enable_sequence_parallel touches: a shared config with
    ``_attn_implementation`` / ``num_key_value_heads`` reachable via ``get_text_config`` and modules."""

    def __init__(self, impl, n_kv=4):
        super().__init__()
        self.config = _StubConfig(impl, n_kv)
        self.attn = nn.Linear(2, 2)
        self.attn.config = self.config


@pytest.fixture()
def sp_world1():
    """A world=1 gloo group so ``dist.get_world_size(group)`` works without multi-process launch."""
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
    """Snapshot/restore the process-global attention + strategy + hook registries around a test."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    base_attn = dict(ALL_ATTENTION_FUNCTIONS)
    base_strats = dict(sp._SP_STRATEGIES)
    base_hooks = list(sp._SP_MODEL_HOOKS)
    base_bases = dict(sp._SP_ATTN_BASES)
    ALL_ATTENTION_FUNCTIONS["_sp_test_base"] = lambda module, q, k, v, mask, **kw: (q, None)
    try:
        yield ALL_ATTENTION_FUNCTIONS
    finally:
        ALL_ATTENTION_FUNCTIONS.clear()
        ALL_ATTENTION_FUNCTIONS.update(base_attn)
        sp._SP_STRATEGIES.clear()
        sp._SP_STRATEGIES.update(base_strats)
        sp._SP_MODEL_HOOKS[:] = base_hooks
        sp._SP_ATTN_BASES.clear()
        sp._SP_ATTN_BASES.update(base_bases)


# --------------------------------------------------------------------------- (a) per-model install
def test_enable_is_per_model_and_does_not_rebind_others(sp_world1, clean_registries):
    all_attn = clean_registries
    base_fn = all_attn["_sp_test_base"]
    model_a = _StubModel("_sp_test_base")
    model_b = _StubModel("_sp_test_base")

    handler_a = sp.enable_sequence_parallel(model_a, sp_world1)

    # model A now routes through a UNIQUE key that wraps the true base...
    key_a = model_a.config._attn_implementation
    assert key_a != "_sp_test_base" and all_attn[key_a] is handler_a
    assert handler_a.attn_fn is base_fn
    # ...and the shared base impl is untouched, so model B (not yet enabled) still uses it.
    assert all_attn["_sp_test_base"] is base_fn
    assert model_b.config._attn_implementation == "_sp_test_base"

    handler_b = sp.enable_sequence_parallel(model_b, sp_world1)
    assert model_b.config._attn_implementation != key_a, "each model gets its own attention key"
    assert handler_b is not handler_a


def test_reenable_unwraps_instead_of_double_wrapping(sp_world1, clean_registries):
    all_attn = clean_registries
    base_fn = all_attn["_sp_test_base"]
    model = _StubModel("_sp_test_base")

    sp.enable_sequence_parallel(model, sp_world1)
    handler2 = sp.enable_sequence_parallel(model, sp_world1)  # re-prepare
    # the second handler wraps the ORIGINAL base, not the first handler (no double all-to-all)
    assert handler2.attn_fn is base_fn


# --------------------------------------------------------------------------- (b) strategy seam
def test_strategy_registry(clean_registries):
    assert isinstance(sp.get_sp_strategy("ulysses"), sp.UlyssesStrategy)

    class _Custom(sp.SequenceParallelStrategy):
        name = "custom"

    sp.register_sp_strategy("custom", _Custom)
    assert isinstance(sp.get_sp_strategy("custom"), _Custom)
    with pytest.raises(ValueError):
        sp.get_sp_strategy("nope")


def test_enable_uses_registered_strategy_backend(sp_world1, clean_registries):
    seen = {}

    class _Custom(sp.SequenceParallelStrategy):
        name = "custom"

        def build_attention(self, base_attn_fn, sp_group):
            seen["base"] = base_attn_fn
            return sp.UlyssesAttention(sp_group, base_attn_fn)

    sp.register_sp_strategy("custom", _Custom)
    model = _StubModel("_sp_test_base")
    sp.enable_sequence_parallel(model, sp_world1, backend="custom")
    assert seen["base"] is clean_registries["_sp_test_base"]


# --------------------------------------------------------------------------- (e) mixer hook
def test_model_hook_runs_on_enable(sp_world1, clean_registries):
    calls = []
    sp.register_sp_model_hook(lambda model, group: calls.append((model, group)))
    model = _StubModel("_sp_test_base")
    sp.enable_sequence_parallel(model, sp_world1)
    assert calls == [(model, sp_world1)]


# --------------------------------------------------------------------------- (c) self-derived cu_seqlens
def _cu_seqlens_worker(rank, world, packed, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        # GLOBAL positions: two docs of length 4 (packed) or one contiguous run (unpacked). seq=8, sp=2.
        if packed:
            global_pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        else:
            global_pos = torch.arange(8).unsqueeze(0)
        local = global_pos.chunk(world, dim=1)[rank].contiguous()  # this rank's shard of positions

        handler = sp.UlyssesAttention(dist.group.WORLD, attn_fn=None)
        cu, max_len = handler._global_cu_seqlens(local)
        # second call with the SAME tensor must hit the memo (no re-gather), same result
        cu2, _ = handler._global_cu_seqlens(local)
        if rank == 0:
            q.put(
                (
                    None if cu is None else cu.tolist(),
                    max_len,
                    None if cu2 is None else cu2.tolist(),
                    handler._cache_ptr is not None,
                )
            )
    finally:
        dist.destroy_process_group()


def _spawn(target, packed, world=2, get_timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=target, args=(r, world, packed, port, q)) for r in range(world)]
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


def test_attention_self_derives_global_cu_seqlens_when_packed():
    msgs, exitcodes = _spawn(_cu_seqlens_worker, packed=True)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    cu, max_len, cu_memo, cached = msgs[0]
    # two docs of length 4 over the global sequence of 8 -> boundaries [0, 4, 8]
    assert cu == [0, 4, 8] and max_len == 4
    assert cu_memo == cu and cached, "second call should reuse the memoized gather"


def test_attention_self_derives_none_when_unpacked():
    msgs, exitcodes = _spawn(_cu_seqlens_worker, packed=False)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    cu, max_len, _, _ = msgs[0]
    assert cu is None and max_len is None, "a single contiguous document needs no varlen cu_seqlens"
