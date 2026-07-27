# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""
Native (torch-only) Ulysses sequence parallelism; flash kernel or sdpa, no DeepSpeed.

Ulysses: each rank holds a 1/sp contiguous sequence shard (all heads); an all-to-all re-shards q/k/v
to (1/sp heads, full sequence), the model's own attention runs, and the inverse all-to-all restores
the layout. Requires sp_size | num_kv_heads.

`enable_sequence_parallel` installs the attention transform per-module (a dispatcher under the real
`_attn_implementation` key routes tagged modules through the handler), so it never rebinds another
model's attention and keeps the key valid for flash-kernel resolution. It routes each mixer to a
`SequenceParallelStrategy` via an optional `dispatch` — the seam for hybrid models and downstream
backends (Ring/USP, Mamba/SSM state passing). Ulysses is the built-in strategy; USP (Ulysses x Ring)
is scaffolded — a strategy reads whatever sub-dims it needs off the `SPContext` mesh — with its ring
leg stubbed. `register_sp_strategy` / `register_sp_model_hook` are the extension points.

`shard_sequence_batch` builds global position_ids/shift_labels, pads the sequence to a multiple of the
sp size (so shards are even), and shards contiguously. Packed/varlen cu_seqlens are self-derived inside
the attention from the sharded position_ids, so there is no dataloader -> handler side channel: flash
impls get the varlen indices, sdpa/eager get the equivalent block-diagonal mask rebuilt for the
gathered sequence. Padding masks are honored only as right-padding (safe under causal attention);
left/interior padding is rejected at sharding time.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def _cu_seqlens_from_position_ids(position_ids):
    """Global packed cu_seqlens + max doc length (doc boundaries are where the position resets to 0).
    Copied from transformers' `prepare_fa_kwargs_from_position_ids` (modeling_flash_attention_utils)."""
    pos = position_ids.reshape(-1)
    cu = torch.cat([(pos == 0).nonzero().view(-1), torch.tensor([pos.numel()], device=pos.device)]).to(torch.int32)
    return cu, int(cu.diff().max())


def _check_kv_divisible(model, sp_size):
    text_config = model.config.get_text_config()
    n_kv = getattr(text_config, "num_key_value_heads", None) or text_config.num_attention_heads
    if n_kv % sp_size:
        raise ValueError(f"sp_size={sp_size} must divide num_key_value_heads={n_kv} for Ulysses.")


# --------------------------------------------------------------------------------------
# Ulysses all-to-all (heads <-> sequence). The permutes emit the flash (b, s, h, d) layout, which
# the attention path then transposes to HF's (b, h, s, d) for the model's own attention. The group
# + sp size are passed through the autograd Functions (stored on ctx for backward), not via globals.
# --------------------------------------------------------------------------------------
def _all_to_all_single(t, group):
    out = torch.empty_like(t)
    dist.all_to_all_single(out, t, group=group)
    return out


def seq_gather(t, group, sp):
    """[b, H, s, d] -> [b, sp*s, H/sp, d]: scatter head groups, gather the sequence."""
    b, n_heads, s, d = t.shape
    t = t.view(b, sp, n_heads // sp, s, d).permute(1, 0, 2, 3, 4).contiguous()
    out = _all_to_all_single(t, group)
    return out.permute(1, 0, 3, 2, 4).reshape(b, sp * s, n_heads // sp, d)


def head_gather(t, group, sp):
    """[b, sp*s, H/sp, d] -> [b, s, H, d]: scatter sequence chunks, gather head groups."""
    b, full, hp, d = t.shape
    s = full // sp
    t = t.view(b, sp, s, hp, d).permute(1, 0, 2, 3, 4).contiguous()
    out = _all_to_all_single(t, group)
    return out.permute(1, 2, 0, 3, 4).reshape(b, s, sp * hp, d)


class GatherSeqScatterHeads(torch.autograd.Function):
    """Ulysses all-to-all: [b, H, s, d] -> [b, S, H/sp, d] (gather the sequence, scatter the
    heads); backward is the inverse exchange."""

    @staticmethod
    def forward(ctx, t, group, sp):
        ctx.group, ctx.sp = group, sp
        return seq_gather(t, group, sp)

    @staticmethod
    def backward(ctx, grad):
        return head_gather(grad, ctx.group, ctx.sp).transpose(1, 2), None, None


class GatherHeadsScatterSeq(torch.autograd.Function):
    """Inverse Ulysses all-to-all: [b, S, H/sp, d] -> [b, s, H, d] (gather the heads, scatter the
    sequence); backward is the inverse exchange."""

    @staticmethod
    def forward(ctx, t, group, sp):
        ctx.group, ctx.sp = group, sp
        return head_gather(t, group, sp)

    @staticmethod
    def backward(ctx, grad):
        return seq_gather(grad.transpose(1, 2), ctx.group, ctx.sp), None, None


class UlyssesAttention:
    """Pure Ulysses: all-to-all to (1/sp heads, full sequence), then the model's OWN attention on
    the gathered sequence, inverse all-to-all. Reuses whatever impl the model loaded (sdpa / pip or
    hub flash) — causal, sliding-window and packed/varlen all handled by HF — so no custom backend
    is needed. In q [b,Hq,s,d] / k,v [b,Hkv,s,d]; out [b,s,Hq,d]. Requires sp_size | Hkv."""

    def __init__(self, sp_group, attn_fn):
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.attn_fn = attn_fn  # the model's original HF attention, captured before we override it
        self._cache_ref = None  # the exact position_ids tensor the varlen cache was derived from
        self._cu_seqlens = None
        self._max_seqlen = None
        self._global_pos = None  # gathered full-sequence positions, kept to rebuild a packed mask
        self._mask_cache = None

    @property
    def cu_seqlens(self):
        return self._cu_seqlens

    @property
    def max_seqlen(self):
        return self._max_seqlen

    def set_varlen(self, cu_seqlens, max_seqlen):  # noqa: D401 - back-compat shim
        """Deprecated no-op: varlen ``cu_seqlens`` are now self-derived from ``position_ids``."""

    def _global_cu_seqlens(self, local_position_ids, device=None):
        """GLOBAL packed cu_seqlens (or ``(None, None)`` when unpacked) from the rank-local
        ``position_ids`` the model threads in: all-gather across the sp group to rebuild the
        full-sequence positions, then read the doc boundaries. Memoized on tensor IDENTITY: the model
        threads ONE position_ids tensor through every layer, so N layers share one gather per forward,
        and a new batch is a new tensor, so every rank re-gathers together. Identity, not `data_ptr`
        — the held reference keeps the cached tensor alive, so the allocator can never hand a new
        batch the same pointer with different contents (which would silently reuse stale boundaries
        and desync the collective across ranks)."""
        if local_position_ids is None or self.sp_size == 1:
            return None, None
        if local_position_ids is not self._cache_ref:
            local = local_position_ids.contiguous()
            if device is not None and local.device != device:
                local = local.to(device)
            gathered = [torch.empty_like(local) for _ in range(self.sp_size)]
            dist.all_gather(gathered, local, group=self.sp_group)
            global_pos = torch.cat(gathered, dim=-1)
            packed = int((global_pos[0] == 0).sum()) > 1
            self._cu_seqlens, self._max_seqlen = _cu_seqlens_from_position_ids(global_pos) if packed else (None, None)
            self._global_pos = global_pos if packed else None
            self._mask_cache = None
            self._cache_ref = local_position_ids
        return self._cu_seqlens, self._max_seqlen

    def _packed_attention_mask(self, dtype, device):
        """Additive [b, 1, S, S] block-diagonal causal mask over the gathered full sequence, rebuilt
        from the global position_ids — the same mask transformers' masking_utils derives from packed
        position_ids at sp=1 (the model built its mask for the LOCAL shard, which we drop). Costs
        O(S^2) like any non-varlen packed mask; flash varlen avoids it via cu_seqlens."""
        if self._mask_cache is None or self._mask_cache.device != device or self._mask_cache.dtype != dtype:
            pos = self._global_pos
            seg = (torch.diff(pos, prepend=pos[:, :1] - 1, dim=-1) != 1).cumsum(-1)  # doc id per token
            keep = seg[:, None, :, None] == seg[:, None, None, :]  # same-doc [b, 1, S, S]
            length = pos.shape[-1]
            keep = keep & torch.tril(torch.ones(length, length, dtype=torch.bool, device=pos.device))
            mask = torch.zeros(keep.shape, dtype=dtype, device=device)
            self._mask_cache = mask.masked_fill_(~keep.to(device), torch.finfo(dtype).min)
        return self._mask_cache

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        group, sp = self.sp_group, self.sp_size
        cu_seqlens, max_seqlen = self._global_cu_seqlens(kwargs.get("position_ids"), device=query.device)
        # heads -> full sequence, then to HF's [b, h, S, d] layout
        q = GatherSeqScatterHeads.apply(query, group, sp).transpose(1, 2)
        k = GatherSeqScatterHeads.apply(key, group, sp).transpose(1, 2)
        v = GatherSeqScatterHeads.apply(value, group, sp).transpose(1, 2)
        # Drop the LOCAL-shard kwargs (position_ids + flash varlen indices keyed to the pre-gather
        # length, plus the local-length mask); packed boundaries are re-established for the full
        # sequence below, and an all-ones/right-padded mask is safely dropped (causal attention only
        # lets pad queries — whose labels are masked — see the trailing pad keys).
        for k_ in ("position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(k_, None)
        inner_mask = None
        if cu_seqlens is not None:
            impl = getattr(getattr(module, "config", None), "_attn_implementation", "")
            if "flash" in impl:
                # flash consumes the varlen indices directly
                kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens.to(q.device)
                kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen
            elif impl in ("sdpa", "eager"):
                # sdpa/eager ignore cu_seq_lens kwargs: hand them the equivalent dense mask instead
                inner_mask = self._packed_attention_mask(q.dtype, q.device)
            else:
                raise NotImplementedError(
                    f"Packed/varlen sequences under native SP require a flash attention implementation "
                    f"(or sdpa/eager, for which the packed mask is rebuilt), got "
                    f"attn_implementation={impl!r}."
                )
        out, _ = self.attn_fn(module, q, k, v, inner_mask, **kwargs)
        return GatherHeadsScatterSeq.apply(out, group, sp), None


# Extensibility seam: each mixer gets a `SequenceParallelStrategy` for its mechanism, installed either
# by rebinding the attention registry key (dense attention) or by a per-module wrap (recurrent/sparse).
# Ulysses is the only built-in; register_sp_strategy / register_sp_model_hook are the extension points.
@dataclass
class SPContext:
    """What a strategy is handed. `group` is the full CP degree (sharding, hooks, and any strategy
    that uses the whole degree — Ulysses, Ring, state passing, DSA). `mesh` is the sp (sub-)mesh when
    one was built, so a strategy that decomposes the degree reads whatever named sub-dims IT needs
    (e.g. USP reads `ulysses`/`ring`); it is `None` for a bare process group."""

    group: "dist.ProcessGroup"
    mesh: Optional[object] = None


# The sequence-parallel axes of a device mesh; the rest (dp/tp/ep) compose orthogonally.
_SP_MESH_DIMS = ("ulysses", "ring", "sp", "cp")


def sp_context_from_mesh(mesh, seq_dims=None) -> SPContext:
    """Wrap a device mesh in an `SPContext`. `group` is the sequence axis (the `seq_dims`, defaulting
    to the recognized sequence-parallel dims); the WHOLE mesh is kept on `ctx.mesh` so a strategy can
    read its own sub-dims (USP's `ulysses`/`ring`) or a composing axis (`tp`/`ep`) — pass the full ND
    mesh here, not a pre-sliced group."""
    dims = mesh.mesh_dim_names or ()
    seq = list(seq_dims) if seq_dims else [d for d in dims if d in _SP_MESH_DIMS]
    if len(seq) == 1:
        group = mesh[seq[0]].get_group()
    elif len(seq) > 1:
        group = mesh[tuple(seq)]._flatten().get_group()
    else:
        group = mesh[dims[0]].get_group() if dims else mesh.get_group()
    return SPContext(group, mesh)


def _to_sp_context(sp) -> SPContext:
    if isinstance(sp, SPContext):
        return sp
    if hasattr(sp, "mesh_dim_names"):  # a torch DeviceMesh
        return sp_context_from_mesh(sp)
    return SPContext(sp)  # a bare ProcessGroup


class SequenceParallelStrategy:
    """Base class pairing a sequence sharder with a per-mixer transform. Strategies implement
    `build_attention(base_attn_fn, ctx)` (registry-key install) or `wrap_module` (per-module install),
    and may override `validate` to check model/topology requirements."""

    name = "sequence_parallel"
    requires_contiguous_shard = False
    installs_via_registry = False

    def validate(self, model, ctx: SPContext):
        """Raise if the model/topology can't support this strategy (e.g. head divisibility)."""

    def shard_batch(self, batch, shard_group, seq_dim=1, ignore_index=-100):
        return shard_sequence_batch(batch, shard_group, seq_dim=seq_dim, ignore_index=ignore_index)

    def build_attention(self, base_attn_fn, ctx: SPContext):
        """Registry-key install: return the attention callable wrapping `base_attn_fn`."""
        raise NotImplementedError

    def wrap_module(self, module, sp_group):
        """Per-module install: wrap `module` in place over `sp_group`; return it."""
        raise NotImplementedError


class UlyssesStrategy(SequenceParallelStrategy):
    """Ulysses attention over the whole CP group. Requires `sp_size | num_kv_heads`."""

    name = "ulysses"
    installs_via_registry = True

    def validate(self, model, ctx: SPContext):
        _check_kv_divisible(model, dist.get_world_size(ctx.group))

    def build_attention(self, base_attn_fn, ctx: SPContext):
        return UlyssesAttention(ctx.group, base_attn_fn)


class UspAttention:
    """USP (Unified SP): Ulysses all-to-all over the intra-node `ulysses_group`, then ring attention
    over the inter-node `ring_group`, then the inverse. With `ring_size == 1` it degenerates to pure
    Ulysses. The ring leg is a stub here — a downstream backend (e.g. ringmaster) provides it; this
    establishes that the seam plumbs both groups to the transform."""

    def __init__(self, ulysses_group, ring_group, attn_fn):
        self.attn_fn = attn_fn
        self.ulysses_group = ulysses_group
        self.ring_group = ring_group
        self.ring_size = dist.get_world_size(ring_group) if ring_group is not None else 1
        self._ulysses = UlyssesAttention(ulysses_group, attn_fn)

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if self.ring_size == 1:  # no inter-node ring dim -> plain Ulysses over the ulysses group
            return self._ulysses(module, query, key, value, attention_mask, **kwargs)
        raise NotImplementedError(
            "USP inter-node ring leg is not implemented in accelerate; register a downstream strategy "
            "(e.g. ringmaster) that provides ring attention over `ring_group`."
        )


class UspStrategy(SequenceParallelStrategy):
    """Ulysses (intra-node) x Ring (inter-node), read off the `ulysses`/`ring` mesh dims. The Ulysses
    degree (not the full degree) must divide the KV heads; the ring leg is stubbed (provided
    downstream)."""

    name = "usp"
    installs_via_registry = True

    def _groups(self, ctx: SPContext):
        dims = getattr(ctx.mesh, "mesh_dim_names", None) or ()
        if "ulysses" not in dims or "ring" not in dims:
            raise ValueError(f"USP needs an sp mesh with `ulysses` and `ring` dims (got mesh dims {dims!r})")
        return ctx.mesh["ulysses"].get_group(), ctx.mesh["ring"].get_group()

    def validate(self, model, ctx: SPContext):
        ulysses_group, _ = self._groups(ctx)
        _check_kv_divisible(model, dist.get_world_size(ulysses_group))

    def build_attention(self, base_attn_fn, ctx: SPContext):
        ulysses_group, ring_group = self._groups(ctx)
        return UspAttention(ulysses_group, ring_group, base_attn_fn)


_SP_STRATEGIES = {"ulysses": UlyssesStrategy, "usp": UspStrategy}


def register_sp_strategy(name, strategy_cls):
    """Register a sequence-parallel strategy under `name` (e.g. a downstream Ring/USP/state-passing backend)."""
    _SP_STRATEGIES[name] = strategy_cls


def get_sp_strategy(name):
    if name not in _SP_STRATEGIES:
        raise ValueError(f"unknown sequence-parallel strategy {name!r}; registered: {sorted(_SP_STRATEGIES)}")
    return _SP_STRATEGIES[name]()


_SP_MODEL_HOOKS = []


def register_sp_model_hook(fn):
    """Register `fn(model, sp_group)` to run when SP is enabled — the seam for non-attention mixers
    (e.g. Mamba/SSM state passing) to wire onto the `sp` group. Returns `fn`."""
    _SP_MODEL_HOOKS.append(fn)
    return fn


def is_attention_module(module) -> bool:
    """True for a transformers attention submodule (routes through `config._attn_implementation`)."""
    cfg = getattr(module, "config", None)
    return "attention" in type(module).__name__.lower() and cfg is not None and hasattr(cfg, "_attn_implementation")


def default_sp_dispatch(module):
    """Dense attention -> Ulysses; everything else -> `None` (a downstream dispatch covers other mixers)."""
    if is_attention_module(module):
        return UlyssesStrategy()
    return None


def _make_sp_dispatcher(base):
    """Wrap the base attention so it routes per-module: an SP-enabled module (tagged with
    ``_sp_handler``) goes through its handler, any other module falls back to the base. Installed once
    under the real ``_attn_implementation`` key, so it never rebinds another model's attention and
    keeps the key valid for transformers' flash-kernel / mask resolution (which reads it verbatim)."""

    def dispatcher(module, query, key, value, attention_mask, **kwargs):
        handler = getattr(module, "_sp_handler", None)
        if handler is not None:
            return handler(module, query, key, value, attention_mask, **kwargs)
        return base(module, query, key, value, attention_mask, **kwargs)

    dispatcher._sp_base = base
    return dispatcher


def _ensure_sp_dispatcher(model):
    """Install (once) the per-module dispatcher under the model's ``_attn_implementation`` key and
    return the unwrapped base attention. ``_attn_implementation`` itself is left unchanged."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config.get_text_config()._attn_implementation
    current = ALL_ATTENTION_FUNCTIONS[impl]
    base = getattr(current, "_sp_base", current)  # unwrap if the dispatcher is already installed
    if getattr(current, "_sp_base", None) is None:
        ALL_ATTENTION_FUNCTIONS[impl] = _make_sp_dispatcher(base)
    return base


def _install_registry_attention(model, attn_modules, strategy, ctx):
    """Install the per-module dispatcher under the model's ``_attn_implementation``, build the
    strategy's handler over the real base attention, and tag the given attention modules with it.
    Returns the handler."""
    base = _ensure_sp_dispatcher(model)
    handler = strategy.build_attention(base, ctx)
    for module in attn_modules:
        module._sp_handler = handler
    return handler


def enable_sequence_parallel(model, sp_group, strategy=None, backend="ulysses", dispatch=None):
    """Enable sequence parallelism on `model` over `sp_group`.

    `sp_group` may be a ProcessGroup (pure Ulysses over the whole group), a device mesh (factored into
    ulysses/ring for USP by `sp_context_from_mesh`), or an `SPContext`. `dispatch(module) -> strategy |
    None` routes each mixer to its strategy (the seam for hybrid models); without it the whole model
    uses one `strategy` / `backend`. The attention transform is installed per-module (via a dispatcher
    under the real impl key), so it never rebinds another model's attention. Returns the handler."""
    if not hasattr(model, "config"):
        raise ValueError(
            f"Sequence parallelism expects a HF Transformers model with a `config` attribute, got "
            f"{type(model).__name__}."
        )
    ctx = _to_sp_context(sp_group)
    if dispatch is not None:
        assigned = [(m, dispatch(m)) for m in model.modules()]
        assigned = [(m, s) for m, s in assigned if s is not None]
        for _, strat in assigned:
            strat.validate(model, ctx)
        # One handler per registry-strategy TYPE (dispatch builds a fresh instance per module, so the
        # type is the identity): modules routed to the same strategy share one handler — and one
        # per-step varlen gather. A registry strategy needing per-module configuration should install
        # via `wrap_module` instead.
        handler = None
        handlers_by_type = {}
        for module, strat in assigned:
            if strat.installs_via_registry:
                shared = handlers_by_type.get(type(strat))
                if shared is None:
                    shared = handlers_by_type[type(strat)] = _install_registry_attention(model, [], strat, ctx)
                module._sp_handler = shared
                handler = handler or shared
            else:
                strat.wrap_module(module, ctx.group)
        for hook in _SP_MODEL_HOOKS:
            hook(model, ctx.group)
        return handler

    strategy = strategy or get_sp_strategy(backend)
    strategy.validate(model, ctx)
    # Model-wide: tag every module (the dispatcher only fires for the ones that call the attention
    # interface) so this model's attention routes through the handler regardless of layer naming.
    handler = _install_registry_attention(model, list(model.modules()), strategy, ctx)
    for hook in _SP_MODEL_HOOKS:
        hook(model, ctx.group)
    return handler


def enable_ulysses_sp(model, sp_group):
    """Back-compat wrapper: enable native Ulysses SP on ``model`` over ``sp_group``. Prefer
    :func:`enable_sequence_parallel`."""
    return enable_sequence_parallel(model, sp_group, backend="ulysses")


def _pad_sequence_to_multiple(batch, multiple, seq_dim, ignore_index):
    """Right-pad the sequence-dim tensors to a multiple of ``multiple``. Pads are label-masked and
    ``position_ids`` continue from the last position, so they add no loss and form no varlen boundary."""
    ref = next(
        (batch[k] for k in ("input_ids", "position_ids", "labels", "shift_labels") if batch.get(k) is not None), None
    )
    if ref is None:
        return batch
    pad_len = (-ref.shape[seq_dim]) % multiple
    if pad_len == 0:
        return batch

    def _pad(t, value):
        shape = list(t.shape)
        shape[seq_dim] = pad_len
        return torch.cat([t, torch.full(shape, value, dtype=t.dtype, device=t.device)], dim=seq_dim)

    for key, value in (("input_ids", 0), ("labels", ignore_index), ("shift_labels", ignore_index)):
        if batch.get(key) is not None:
            batch[key] = _pad(batch[key], value)
    if batch.get("attention_mask") is not None:
        batch["attention_mask"] = _pad(batch["attention_mask"], 0)
    if batch.get("position_ids") is not None:
        pos = batch["position_ids"]
        cont = pos[..., -1:] + torch.arange(1, pad_len + 1, device=pos.device).expand(pos.shape[0], pad_len)
        batch["position_ids"] = torch.cat([pos, cont], dim=seq_dim)
    return batch


# Shared by the dataloader adapter (SFT) and callers that shard an already-materialized batch at the
# forward (e.g. generated GRPO rollouts).
def shard_sequence_batch(batch, shard_group, attention=None, seq_dim=1, ignore_index=-100):
    """Contiguously shard a batch's sequence over ``shard_group`` for native Ulysses SP.

    Builds the GLOBAL ``position_ids``/``shift_labels`` (shifted on the full sequence so shard
    boundaries keep next-token alignment), pads the sequence to a multiple of the world size (so the
    shard is even), then shards along ``seq_dim``. Packed/varlen ``cu_seqlens`` are self-derived in the
    attention from the sharded ``position_ids`` — no side channel. Returns a new dict (input unchanged).

    Args:
        batch: mapping with at least ``input_ids`` (and optionally ``labels``/``position_ids``).
        shard_group: the sequence-shard (``sp``) process group.
        attention: deprecated / unused (kept for back-compat; varlen is self-derived).
        seq_dim: the sequence dimension. ignore_index: final-token / pad label id.
    """
    rank, world = dist.get_rank(shard_group), dist.get_world_size(shard_group)
    batch = dict(batch)
    # 1. global position_ids
    if "position_ids" not in batch and "input_ids" in batch:
        ids = batch["input_ids"]
        pos = torch.arange(ids.shape[seq_dim], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        batch["position_ids"] = pos.contiguous()
    # 2. global shift_labels (shift before sharding)
    if "labels" in batch and "shift_labels" not in batch:
        labels = batch["labels"]
        shift = torch.full_like(labels, ignore_index)
        shift[..., :-1] = labels[..., 1:]
        batch["shift_labels"] = shift
    # 3. reject padding layouts the attention cannot honor: the Ulysses attention runs on the
    # gathered full sequence WITHOUT the padding mask, which is only safe for right-padding (under
    # causal attention pad keys are visible solely to pad queries, whose labels are masked).
    # Left/interior padding would silently attend wrong, so fail loudly instead.
    mask = batch.get("attention_mask")
    if world > 1 and mask is not None and seq_dim == 1 and mask.dim() == 2:
        if (mask[..., :-1] < mask[..., 1:]).any():
            raise ValueError(
                "Native SP received an attention_mask with left/interior padding; only right-padding "
                "is supported, because the Ulysses attention drops the padding mask when it runs on "
                "the gathered full sequence. Pack sequences or right-pad instead."
            )
    # 4. pad to a multiple of the world size so the shard is EVEN (uneven shards are a hard Ulysses
    # all-to-all size mismatch); pads are label-masked and continue position_ids (no varlen boundary).
    if world > 1:
        batch = _pad_sequence_to_multiple(batch, world, seq_dim, ignore_index)
    # 5. contiguous (even) sequence shard. varlen cu_seqlens are self-derived in the attention from the
    # sharded position_ids, so there is no side channel here (`attention` is unused).
    if world > 1:
        for key in ("input_ids", "labels", "position_ids", "shift_labels", "attention_mask"):
            t = batch.get(key)
            if t is not None:
                batch[key] = t.chunk(world, dim=seq_dim)[rank].contiguous()
    return batch


# --------------------------------------------------------------------------------------
# Dataloader adapter: replicate-then-shard the sequence across the shard group.
# --------------------------------------------------------------------------------------
class SequenceShardingDataLoader:
    """Sequence-sharding DataLoader wrapper for native Ulysses SP (torch analogue of DeepSpeed's
    ``UlyssesSPDataLoaderAdapter``). accelerate's ``prepare_data_loader`` already hands every sp
    rank the SAME sample (its data-parallel sharding divides out tp*cp*sp), so per batch this just
    builds global ``position_ids``/``shift_labels`` (shifted on the full sequence so shard
    boundaries keep next-token alignment) and shards the sequence contiguously over the sp group.
    Loss normalization is left to the trainer/accelerate.

    Args:
        dataloader: the accelerate-prepared DataLoader to wrap (must hand sp ranks the same sample).
        shard_group: the sequence-shard (``sp``) group.
        strategy: the sequence-parallel strategy whose ``shard_batch`` decides the layout (e.g. a Ring
            strategy that wants zigzag). ``None`` uses the contiguous default.
        attention: deprecated / unused (kept for back-compat; varlen is self-derived).
        seq_dim / ignore_index: sequence dim / final-token label id.
    """

    def __init__(self, dataloader, shard_group, strategy=None, attention=None, seq_dim=1, ignore_index=-100):
        self.dataloader = dataloader
        self.strategy = strategy
        self.attention = attention
        self.seq_dim = seq_dim
        self.ignore_index = ignore_index
        self.shard_group = shard_group
        self.rank = dist.get_rank(shard_group)
        self.world = dist.get_world_size(shard_group)

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        # Delegate everything else (set_epoch, total_batch_size, batch_sampler, dataset, ...) to the
        # wrapped loader so trainer code sees the usual accelerate DataLoader surface.
        try:
            dataloader = self.__dict__["dataloader"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(dataloader, name)

    def __iter__(self):
        for batch in self.dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"SequenceShardingDataLoader needs dict-style batches with 'input_ids' to shard "
                    f"the sequence, got {type(batch).__name__}; return a mapping from your collate_fn."
                )
            yield self._process(dict(batch))

    def _process(self, batch):
        # The strategy owns the shard layout (contiguous for Ulysses, zigzag for a causal ring, ...);
        # `None` falls back to the contiguous default.
        if self.strategy is not None:
            return self.strategy.shard_batch(
                batch, self.shard_group, seq_dim=self.seq_dim, ignore_index=self.ignore_index
            )
        return shard_sequence_batch(batch, self.shard_group, seq_dim=self.seq_dim, ignore_index=self.ignore_index)
