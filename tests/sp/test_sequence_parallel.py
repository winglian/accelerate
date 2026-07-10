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

"""CPU/gloo tests for native Ulysses SP: sequence padding, self-derived varlen cu_seqlens, per-model
attention install, the strategy registry, and per-module dispatch (the extension seam)."""

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


# --------------------------------------------------------------------------- stubs
class _Config:
    def __init__(self, impl, n_kv):
        self._attn_implementation = impl
        self.num_key_value_heads = n_kv

    def get_text_config(self):
        return self


class _StubAttention(nn.Module):  # matched by is_attention_module
    def __init__(self, config):
        super().__init__()
        self.config = config


class ToyMixer(nn.Module):  # a non-attention mixer for the dispatch/seam test
    pass


class _Model(nn.Module):
    def __init__(self, impl="_sp_test_base", n_kv=4):
        super().__init__()
        self.config = _Config(impl, n_kv)
        self.attn = _StubAttention(self.config)
        self.mixer = ToyMixer()


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

    base_attn, base_strats = dict(ALL_ATTENTION_FUNCTIONS), dict(sp._SP_STRATEGIES)
    base_hooks = list(sp._SP_MODEL_HOOKS)
    ALL_ATTENTION_FUNCTIONS["_sp_test_base"] = lambda module, q, k, v, mask, **kw: (q, None)
    try:
        yield ALL_ATTENTION_FUNCTIONS
    finally:
        ALL_ATTENTION_FUNCTIONS.clear()
        ALL_ATTENTION_FUNCTIONS.update(base_attn)
        sp._SP_STRATEGIES.clear()
        sp._SP_STRATEGIES.update(base_strats)
        sp._SP_MODEL_HOOKS[:] = base_hooks


# --------------------------------------------------------------------------- gloo helpers
def _spawn(target, arg, world=2, get_timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=target, args=(r, world, arg, port, q)) for r in range(world)]
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


# --------------------------------------------------------------------------- padding
def _shard_worker(rank, world, seq_len, port, q):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        batch = {"input_ids": torch.arange(seq_len).unsqueeze(0)}
        local = sp.shard_sequence_batch(batch, dist.group.WORLD)["input_ids"].shape[1]
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        if rank == 0:
            q.put(gathered)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seq_len", [pytest.param(8, id="divisible"), pytest.param(7, id="indivisible")])
def test_padding_produces_even_shards(seq_len):
    msgs, exitcodes = _spawn(_shard_worker, seq_len)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    assert len(set(msgs[0])) == 1, f"uneven shards for seq_len={seq_len}: {msgs[0]}"


def test_pad_masks_loss_and_continues_positions():
    ignore = -100
    batch = {
        "input_ids": torch.arange(1, 8).unsqueeze(0),
        "labels": torch.arange(1, 8).unsqueeze(0),
        "shift_labels": torch.arange(1, 8).unsqueeze(0),
        "position_ids": torch.arange(7).unsqueeze(0),
    }
    padded = sp._pad_sequence_to_multiple(dict(batch), 4, seq_dim=1, ignore_index=ignore)
    assert padded["input_ids"].shape[1] == 8
    assert (padded["labels"][0, 7:] == ignore).all() and (padded["shift_labels"][0, 7:] == ignore).all()
    assert padded["position_ids"][0, 7].item() == 7  # continues, no 0-reset
    assert torch.equal(padded["input_ids"][0, :7], batch["input_ids"][0])


# --------------------------------------------------------------------------- self-derived varlen
def _cu_worker(rank, world, packed, port, q):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]) if packed else torch.arange(8).unsqueeze(0)
        local = pos.chunk(world, dim=1)[rank].contiguous()
        handler = sp.UlyssesAttention(dist.group.WORLD, attn_fn=None)
        cu, max_len = handler._global_cu_seqlens(local)
        if rank == 0:
            q.put((None if cu is None else cu.tolist(), max_len))
    finally:
        dist.destroy_process_group()


def test_attention_self_derives_cu_seqlens_packed():
    msgs, exitcodes = _spawn(_cu_worker, True)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    assert msgs[0] == ([0, 4, 8], 4)


def test_attention_self_derives_none_unpacked():
    msgs, exitcodes = _spawn(_cu_worker, False)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    assert msgs[0] == (None, None)


# --------------------------------------------------------------------------- per-model install
def test_enable_is_per_model(sp_world1, clean_registries):
    all_attn = clean_registries
    base = all_attn["_sp_test_base"]
    a, b = _Model(), _Model()
    handler = sp.enable_sequence_parallel(a, sp_world1)

    # `_attn_implementation` is UNCHANGED (stays valid for flash-kernel resolution); a dispatcher is
    # installed under it and this model's attention module is tagged with the handler.
    assert a.attn.config._attn_implementation == "_sp_test_base"
    assert a.attn._sp_handler is handler
    assert all_attn["_sp_test_base"]._sp_base is base  # dispatcher wraps the real base
    assert getattr(b.attn, "_sp_handler", None) is None  # other model not tagged -> uses base


def test_enable_preserves_hub_kernel_attn_impl(sp_world1, clean_registries):
    # regression: when the impl is a hub-kernel repo, enabling SP must NOT rewrite `_attn_implementation`
    # (a mangled per-model key would be fetched from the hub -> 401 / flash resolution failure).
    all_attn = clean_registries
    all_attn["kernels-community/flash-attn2"] = lambda module, q, k, v, mask, **kw: (q, None)
    model = _Model(impl="kernels-community/flash-attn2")
    handler = sp.enable_sequence_parallel(model, sp_world1)
    assert model.attn.config._attn_implementation == "kernels-community/flash-attn2"  # unchanged
    assert model.attn._sp_handler is handler
    assert all_attn["kernels-community/flash-attn2"]._sp_base is not None  # dispatcher installed


def test_reenable_unwraps(sp_world1, clean_registries):
    base = clean_registries["_sp_test_base"]
    model = _Model()
    sp.enable_sequence_parallel(model, sp_world1)
    handler2 = sp.enable_sequence_parallel(model, sp_world1)
    assert handler2.attn_fn is base  # wraps the real base, not the dispatcher or the first handler


# --------------------------------------------------------------------------- strategy registry + seam
def test_strategy_registry(clean_registries):
    assert isinstance(sp.get_sp_strategy("ulysses"), sp.UlyssesStrategy)
    with pytest.raises(ValueError):
        sp.get_sp_strategy("nope")


def test_dispatch_routes_by_mechanism(sp_world1, clean_registries):
    all_attn = clean_registries

    class _MarkStrategy(sp.SequenceParallelStrategy):  # a per-module (non-registry) strategy
        name = "mark"

        def wrap_module(self, module, sp_group):
            module._sp_marked = sp_group
            return module

    def dispatch(module):
        if isinstance(module, ToyMixer):
            return _MarkStrategy()
        if sp.is_attention_module(module):
            return sp.UlyssesStrategy()
        return None

    model = _Model()
    handler = sp.enable_sequence_parallel(model, sp_world1, dispatch=dispatch)

    assert model.attn._sp_handler is handler  # attention -> Ulysses (tagged, dispatcher under base key)
    assert all_attn["_sp_test_base"]._sp_base is not None  # dispatcher installed
    assert getattr(model.mixer, "_sp_marked", None) is sp_world1  # ToyMixer -> per-module wrap


def test_model_hook_runs_on_enable(sp_world1, clean_registries):
    calls = []
    sp.register_sp_model_hook(lambda model, group: calls.append((model, group)))
    model = _Model()
    sp.enable_sequence_parallel(model, sp_world1)
    assert calls == [(model, sp_world1)]


# --------------------------------------------------------------------------- USP seam
class _FakeSubMesh:
    def __init__(self, group):
        self._group = group

    def get_group(self):
        return self._group


class _FakeMesh:
    """Minimal stand-in for a device mesh with named sub-dims (a strategy reads sub-groups off it)."""

    def __init__(self, groups):
        self._groups = groups
        self.mesh_dim_names = tuple(groups)

    def __getitem__(self, name):
        return _FakeSubMesh(self._groups[name])


def test_sp_context_wraps_bare_group(sp_world1):
    ctx = sp._to_sp_context(sp_world1)
    assert ctx.group is sp_world1 and ctx.mesh is None


def test_usp_degenerates_to_ulysses_at_ring_size_1(sp_world1):
    # ring dim of size 1 -> UspAttention delegates to the Ulysses leg
    mesh = _FakeMesh({"ulysses": sp_world1, "ring": sp_world1})  # world1: both trivial
    ctx = sp.SPContext(group=sp_world1, mesh=mesh)
    att = sp.UspStrategy().build_attention(lambda *a, **k: (a[1], None), ctx)
    assert att.ring_size == 1 and isinstance(att._ulysses, sp.UlyssesAttention)
    assert sp.get_sp_strategy("usp").name == "usp"


def test_usp_requires_ulysses_ring_mesh_dims(sp_world1):
    # a strategy that decomposes the degree asks the mesh for ITS dims; a bare group can't provide them
    with pytest.raises(ValueError):
        sp.UspStrategy().build_attention(None, sp.SPContext(group=sp_world1, mesh=None))


def _usp_worker(rank, world, _arg, port, q):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=30))
    try:
        # 4 ranks = 2 nodes x 2 gpus: ulysses (intra-node) {0,1},{2,3}; ring (inter-node) {0,2},{1,3}.
        # Every rank builds all sub-groups in the same order (new_group is collective).
        ulysses = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        ring = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        mesh = _FakeMesh({"ulysses": ulysses[rank // 2], "ring": ring[rank % 2]})
        ctx = sp.SPContext(group=dist.group.WORLD, mesh=mesh)  # generic context; strategy reads the mesh
        att = sp.UspStrategy().build_attention(None, ctx)
        raised = False
        try:  # ring leg is stubbed -> a >1 ring degree must raise, proving the leg is reached
            att(None, torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2), None)
        except NotImplementedError:
            raised = True
        if rank == 0:
            q.put((dist.get_world_size(att.ulysses_group), att.ring_size, raised))
    finally:
        dist.destroy_process_group()


def test_usp_strategy_reads_ulysses_and_ring_off_mesh():
    msgs, exitcodes = _spawn(_usp_worker, None, world=4)
    assert all(c == 0 for c in exitcodes) and msgs, f"workers failed: {exitcodes}"
    ulysses_size, ring_size, raised = msgs[0]
    assert (ulysses_size, ring_size) == (2, 2)  # strategy pulled ulysses x ring off the mesh
    assert raised  # the (stubbed) ring leg was reached


# --------------------------------------------------------------------------- ND-composition seams
def test_sp_context_from_mesh_extracts_seq_axis_and_keeps_full_mesh(sp_world1):
    # a full ND mesh (dp/sp/tp): group is the sequence axis, the WHOLE mesh is kept for cross-axis reads
    mesh = _FakeMesh({"dp": None, "sp": sp_world1, "tp": None})
    ctx = sp.sp_context_from_mesh(mesh)
    assert ctx.group is sp_world1 and ctx.mesh is mesh


def test_dataloader_routes_shard_layout_through_strategy(sp_world1):
    class _MarkShard(sp.SequenceParallelStrategy):
        def shard_batch(self, batch, group, seq_dim=1, ignore_index=-100):
            return {"marked": True}

    dl = [{"input_ids": torch.arange(4).unsqueeze(0)}]  # trivial iterable dataloader
    marked = list(sp.SequenceShardingDataLoader(dl, sp_world1, strategy=_MarkShard()))
    assert marked == [{"marked": True}]  # strategy.shard_batch owns the layout
    # default (no strategy) -> contiguous shard_sequence_batch
    assert "position_ids" in list(sp.SequenceShardingDataLoader(dl, sp_world1))[0]
