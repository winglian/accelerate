# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

Ulysses: each rank holds a 1/sp sequence shard (all heads); an all-to-all re-shards q/k/v to
(1/sp heads, full sequence), one local attention runs, the inverse all-to-all restores the layout.
Requires sp_size | num_kv_heads.

Design notes:
  * ``enable_sequence_parallel`` installs the attention transform under a UNIQUE per-model impl key,
    so enabling SP on one model never rebinds another model's attention (ref_model, custom kernels).
  * varlen ``cu_seqlens`` are self-derived inside the attention from the sharded ``position_ids`` (an
    all-gather over the sp group), so there is no dataloader -> handler side channel.
  * a :class:`SequenceParallelStrategy` pairs the sequence sharder with the attention transform, and
    :func:`register_sp_model_hook` lets non-attention mixers (Mamba/SSM) wire onto the ``sp`` group —
    so Ring/USP/hybrid backends compose without touching the enable path.
"""

import itertools

import torch
import torch.distributed as dist


def _cu_seqlens_from_position_ids(position_ids):
    """Global packed cu_seqlens + max doc length (doc boundaries are where the position resets to 0).
    Copied from transformers' `prepare_fa_kwargs_from_position_ids` (modeling_flash_attention_utils)."""
    pos = position_ids.reshape(-1)
    cu = torch.cat([(pos == 0).nonzero().view(-1), torch.tensor([pos.numel()], device=pos.device)]).to(torch.int32)
    return cu, int(cu.diff().max())


def _check_kv_divisible(model, sp_size):
    n_kv = model.config.get_text_config().num_key_value_heads
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
    is needed. In q [b,Hq,s,d] / k,v [b,Hkv,s,d]; out [b,s,Hq,d]. Requires sp_size | Hkv.

    The GLOBAL packed ``cu_seqlens`` are derived here (``_global_cu_seqlens``) by all-gathering the
    rank-local ``position_ids`` the model already threads into the attention call — no dataloader ->
    handler side channel. The gather is memoized per step (keyed on the ``position_ids`` storage), so
    it runs once and every layer reuses it."""

    def __init__(self, sp_group, attn_fn):
        self.sp_group = sp_group
        self.sp_size = dist.get_world_size(sp_group)
        self.attn_fn = attn_fn  # the model's original HF attention, captured before we override it
        self._cache_ptr = None
        self._cu_seqlens = None
        self._max_seqlen = None

    def set_varlen(self, cu_seqlens, max_seqlen):  # noqa: D401 - back-compat shim
        """Deprecated no-op: varlen ``cu_seqlens`` are now self-derived from ``position_ids``."""

    def _global_cu_seqlens(self, local_position_ids):
        """GLOBAL packed cu_seqlens (or ``(None, None)`` when unpacked) from the rank-local
        ``position_ids``: all-gather across the sp group to rebuild the full-sequence positions, then
        read the doc boundaries. Memoized on the ``position_ids`` storage so N attention layers share
        one gather per step."""
        if local_position_ids is None or self.sp_size == 1:
            return None, None
        key = local_position_ids.data_ptr()
        if key != self._cache_ptr:
            gathered = [torch.empty_like(local_position_ids) for _ in range(self.sp_size)]
            dist.all_gather(gathered, local_position_ids.contiguous(), group=self.sp_group)
            global_pos = torch.cat(gathered, dim=-1)
            packed = int((global_pos[0] == 0).sum()) > 1
            self._cu_seqlens, self._max_seqlen = _cu_seqlens_from_position_ids(global_pos) if packed else (None, None)
            self._cache_ptr = key
        return self._cu_seqlens, self._max_seqlen

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        group, sp = self.sp_group, self.sp_size
        cu_seqlens, max_seqlen = self._global_cu_seqlens(kwargs.get("position_ids"))
        # heads -> full sequence, then to HF's [b, h, S, d] layout
        q = GatherSeqScatterHeads.apply(query, group, sp).transpose(1, 2)
        k = GatherSeqScatterHeads.apply(key, group, sp).transpose(1, 2)
        v = GatherSeqScatterHeads.apply(value, group, sp).transpose(1, 2)
        # Drop the LOCAL-shard kwargs (position_ids + flash varlen indices keyed to the pre-gather
        # length); re-inject the GLOBAL packed cu_seqlens for the full sequence the all-to-all rebuilt.
        for k_ in ("position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            kwargs.pop(k_, None)
        if cu_seqlens is not None:
            kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seqlens.to(q.device)
            kwargs["max_length_q"] = kwargs["max_length_k"] = max_seqlen
        out, _ = self.attn_fn(module, q, k, v, None, **kwargs)
        return GatherHeadsScatterSeq.apply(out, group, sp), None


# --------------------------------------------------------------------------------------
# State-passing scan for recurrent mixers (Mamba/SSM, gated linear attention). A recurrent mixer
# CANNOT gather like attention — the scan h_t = a_t*h_{t-1} + b_t has a left-to-right dependency. But
# it is LINEAR in its initial state, so each rank scans its CONTIGUOUS chunk (h0=0), the per-chunk
# (decay, carry) are combined across ranks to get the state entering each chunk, and an exact additive
# correction is applied. Comm is O(P), sequence-length independent. Real Mamba2/GDN kernels realize
# this via their initial_states/return_final_states args (see ringmaster.strategies.mamba); the
# primitive below is the framework-neutral core.
# --------------------------------------------------------------------------------------
def local_linear_scan(a, b):
    """Sequential reference scan ``h_t = a_t*h_{t-1} + b_t`` with zero init. ``a``, ``b``:
    ``[batch, L, d]``. Returns ``[batch, L, d]``."""
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def cp_linear_scan(a, b, group):
    """Exact context-parallel linear scan; equals a global scan of the full sequence. ``a``, ``b`` are
    THIS rank's contiguous chunk ``[batch, L_local, d]``. One all-gather of two ``[batch, d]`` vectors
    (O(P), sequence-length independent) carries the boundary state across ranks."""
    local_h = local_linear_scan(a, b)
    world = dist.get_world_size(group) if group is not None else 1
    if world == 1:
        return local_h
    cum_a = torch.cumprod(a, dim=1)  # [batch, L, d]
    rank = dist.get_rank(group)
    stacked = torch.stack([cum_a[:, -1], local_h[:, -1]], dim=0).contiguous()  # [2, batch, d]
    gathered = [torch.empty_like(stacked) for _ in range(world)]
    dist.all_gather(gathered, stacked, group=group)
    h_in = torch.zeros_like(local_h[:, -1])  # state entering this chunk (prefix over predecessors)
    for j in range(rank):
        h_in = gathered[j][0] * h_in + gathered[j][1]
    return local_h + cum_a * h_in.unsqueeze(1)


# --------------------------------------------------------------------------------------
# Extensibility seam: a sequence-parallel *strategy* pairs a sequence sharder with a per-mixer
# transform. The unit of SP is the MIXER, not the model: ``enable_sequence_parallel(dispatch=...)``
# routes each module to the strategy for its MECHANISM — dense-gather (Ulysses) for attention, state
# passing for Mamba/SSM/GDN, or a model-specific transform (registered by a downstream lib). Uniform
# dense attention collapses onto one shared ALL_ATTENTION_FUNCTIONS key (fast path); everything else is
# wrapped per module.
# --------------------------------------------------------------------------------------
class SequenceParallelStrategy:
    """Base class for a sequence-parallel backend for one mixer *mechanism*.

    A strategy installs one of two ways:
      * ``installs_via_registry = True`` implements ``build_attention`` — installed by rebinding the
        model's ``ALL_ATTENTION_FUNCTIONS`` key (dense attention, where all layers share the key and
        the reused kernel handles causal/sliding/varlen).
      * ``installs_via_registry = False`` implements ``wrap_module`` — a direct per-module forward wrap
        (recurrent/sparse/custom mixers that don't route through the attention registry).
    """

    name = "sequence_parallel"
    requires_kv_divisible = False
    requires_contiguous_shard = False
    installs_via_registry = False

    def shard_batch(self, batch, shard_group, seq_dim=1, ignore_index=-100):
        return shard_sequence_batch(batch, shard_group, seq_dim=seq_dim, ignore_index=ignore_index)

    def build_attention(self, base_attn_fn, sp_group):
        """Registry-key install: return the callable registered as attention (wrapping ``base_attn_fn``)."""
        raise NotImplementedError

    def wrap_module(self, module, sp_group):
        """Per-module install: wrap ``module`` in place over ``sp_group``; return the module."""
        raise NotImplementedError


class UlyssesStrategy(SequenceParallelStrategy):
    name = "ulysses"
    requires_kv_divisible = True  # sp_size | num_kv_heads
    installs_via_registry = True

    def build_attention(self, base_attn_fn, sp_group):
        return UlyssesAttention(sp_group, base_attn_fn)


class StatePassingStrategy(SequenceParallelStrategy):
    """CP for recurrent mixers (Mamba/SSM, GDN/linear-attention): scan the contiguous shard and pass
    the boundary state across the ``sp`` group (``cp_linear_scan``). Installed per-module (recurrent
    layers are not attention). Requires contiguous shards — the recurrence is ordered. A mixer opts in
    by exposing its scan as ``module.sp_scan(a, b)``; real Mamba2/GDN kernels are wrapped analogously
    (their ``initial_states``/``return_final_states`` are the state injection/extraction points)."""

    name = "state_passing"
    requires_contiguous_shard = True
    installs_via_registry = False

    def wrap_module(self, module, sp_group):
        module._sp_group = sp_group
        if hasattr(module, "sp_scan"):
            module.sp_scan = lambda a, b, g=sp_group: cp_linear_scan(a, b, g)
        return module


# Name-based mixer classification (case-insensitive) so dispatch needs no per-architecture imports.
_RECURRENT_HINTS = ("mamba", "gateddelta", "deltanet", "linearattention", "lineattention", "gateddeltanet")


def is_recurrent_mixer(module) -> bool:
    """True for Mamba/SSM or gated-linear-attention mixers (Nemotron-H, Falcon-H1, Granite-MoE-Hybrid,
    Qwen3-Next/GDN, ...)."""
    return any(h in type(module).__name__.lower() for h in _RECURRENT_HINTS)


def is_attention_module(module) -> bool:
    """True for a transformers attention submodule (routes through ``config._attn_implementation``)."""
    cfg = getattr(module, "config", None)
    return "attention" in type(module).__name__.lower() and cfg is not None and hasattr(cfg, "_attn_implementation")


def default_sp_dispatch(module):
    """Map a mixer module to the strategy for its MECHANISM (``None`` = the model owns SP for it, e.g.
    a sparse-attention kernel doing its own compressed-KV gather — only sequence sharding applies).
    Dense attention (full / sliding-window / sinks) is one mechanism → one Ulysses transform."""
    if is_recurrent_mixer(module):
        return StatePassingStrategy()
    if is_attention_module(module):
        return UlyssesStrategy()
    return None


_SP_STRATEGIES = {"ulysses": UlyssesStrategy, "state_passing": StatePassingStrategy}


def register_sp_strategy(name, strategy_cls):
    """Register a sequence-parallel strategy under ``name`` (e.g. a downstream Ring/USP backend)."""
    _SP_STRATEGIES[name] = strategy_cls


def get_sp_strategy(name):
    if name not in _SP_STRATEGIES:
        raise ValueError(f"unknown sequence-parallel strategy {name!r}; registered: {sorted(_SP_STRATEGIES)}")
    return _SP_STRATEGIES[name]()


_SP_MODEL_HOOKS = []


def register_sp_model_hook(fn):
    """Register ``fn(model, sp_group)`` to run when SP is enabled on a model — the extension point for
    non-attention mixers (e.g. Mamba/GDN CP state passing) to wire themselves onto the ``sp`` group.
    Returns ``fn`` so it can be used as a decorator."""
    _SP_MODEL_HOOKS.append(fn)
    return fn


# Unique per-model attention-impl keys so enabling SP on one model never rebinds another model's
# attention (e.g. a DPO/GRPO ref_model, or a model that owns a custom kernel) through the shared
# ``ALL_ATTENTION_FUNCTIONS`` registry.
_SP_ATTN_COUNTER = itertools.count()
_SP_ATTN_BASES = {}  # our_key -> the original (unwrapped) attention fn


def _set_model_attn_impl(model, key):
    """Point the model (and every sub-config that carries one) at ``key`` so only THIS model routes
    through the SP handler."""
    text_config = model.config.get_text_config()
    text_config._attn_implementation = key
    model.config._attn_implementation = key
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = key


def _install_registry_attention(model, attn_modules, strategy, sp_group):
    """Collapse uniform dense attention onto ONE unique per-model impl key wrapping the base attention,
    and point the given attention modules (and the text config) at it. Returns the shared handler."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    text_config = model.config.get_text_config()
    impl = text_config._attn_implementation
    # Unwrap on re-enable so we never wrap a wrapper (double all-to-all).
    base = _SP_ATTN_BASES.get(impl, ALL_ATTENTION_FUNCTIONS[impl])
    handler = strategy.build_attention(base, sp_group)
    key = f"{impl}__{strategy.name}_sp_{next(_SP_ATTN_COUNTER)}"
    ALL_ATTENTION_FUNCTIONS[key] = handler
    _SP_ATTN_BASES[key] = base
    text_config._attn_implementation = key
    for module in attn_modules:
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = key
    return handler


def _enable_per_module(model, sp_group, dispatch):
    """Walk the model and install each mixer's strategy: dense attention collapses onto one registry
    key (fast path), other mixers are wrapped in place. Returns the attention handler (or None)."""
    assigned = [(m, dispatch(m)) for m in model.modules()]
    assigned = [(m, s) for m, s in assigned if s is not None]
    if any(s.requires_kv_divisible for _, s in assigned):
        _check_kv_divisible(model, dist.get_world_size(sp_group))

    attn_modules = [m for m, s in assigned if s.installs_via_registry]
    attention_handler = None
    if attn_modules:
        strategy = next(s for _, s in assigned if s.installs_via_registry)
        attention_handler = _install_registry_attention(model, attn_modules, strategy, sp_group)
    for module, strategy in assigned:
        if not strategy.installs_via_registry:
            strategy.wrap_module(module, sp_group)

    for hook in _SP_MODEL_HOOKS:
        hook(model, sp_group)
    return attention_handler


def enable_sequence_parallel(model, sp_group, strategy=None, backend="ulysses", dispatch=None):
    """Enable sequence parallelism on ``model`` over ``sp_group``.

    ``dispatch(module) -> strategy | None`` routes each mixer to the strategy for its mechanism (use
    :func:`default_sp_dispatch` for hybrid models: attention -> Ulysses, Mamba/GDN -> state passing).
    Without ``dispatch`` the whole model uses a single ``strategy`` / ``backend`` (uniform dense
    attention). Either way the attention transform is installed under a UNIQUE per-model impl key (no
    global rebinding of other models' attention). Returns the attention handler."""
    if not hasattr(model, "config"):
        raise ValueError(
            f"Sequence parallelism expects a HF Transformers model with a `config` attribute, got "
            f"{type(model).__name__}."
        )
    if dispatch is not None:
        return _enable_per_module(model, sp_group, dispatch)

    strategy = strategy or get_sp_strategy(backend)
    if strategy.requires_kv_divisible:
        _check_kv_divisible(model, dist.get_world_size(sp_group))

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config.get_text_config()._attn_implementation
    # If SP was already enabled on this model (re-prepare), unwrap to the true base so we never wrap a
    # wrapper (double all-to-all).
    base = _SP_ATTN_BASES.get(impl, ALL_ATTENTION_FUNCTIONS[impl])
    handler = strategy.build_attention(base, sp_group)

    key = f"{impl}__{strategy.name}_sp_{next(_SP_ATTN_COUNTER)}"
    ALL_ATTENTION_FUNCTIONS[key] = handler
    _SP_ATTN_BASES[key] = base
    _set_model_attn_impl(model, key)

    for hook in _SP_MODEL_HOOKS:
        hook(model, sp_group)
    return handler


def enable_ulysses_sp(model, sp_group):
    """Back-compat wrapper: enable native Ulysses SP on ``model`` over ``sp_group``. Prefer
    :func:`enable_sequence_parallel`."""
    return enable_sequence_parallel(model, sp_group, backend="ulysses")


# --------------------------------------------------------------------------------------
# Sequence-sharding primitive: shared by the dataloader adapter (SFT) and by callers that need to
# shard an already-materialized batch at the forward (e.g. generated GRPO rollouts).
# --------------------------------------------------------------------------------------
def _pad_sequence_to_multiple(batch, multiple, seq_dim, ignore_index):
    """Right-pad the sequence-dim tensors so the length is a multiple of ``multiple`` (needed so the
    contiguous shard below is EVEN — the Ulysses all-to-all requires the same shard length on every
    rank). Pads are label-masked (``ignore_index``) so they cost no loss and are excluded from
    ``num_items_in_batch``; ``position_ids`` continue from the last position (no ``0`` reset) so the
    pad joins the trailing document and never becomes a spurious varlen boundary, and causal
    attention keeps real tokens from ever attending to it."""
    ids = batch["input_ids"]
    seq_len = ids.shape[seq_dim]
    pad_len = (-seq_len) % multiple
    if pad_len == 0:
        return batch, seq_len
    bsz = ids.shape[0]
    device = ids.device

    def _pad(t, value):
        shape = list(t.shape)
        shape[seq_dim] = pad_len
        return torch.cat([t, torch.full(shape, value, dtype=t.dtype, device=t.device)], dim=seq_dim)

    for key, value in (("input_ids", 0), ("labels", ignore_index), ("shift_labels", ignore_index)):
        if batch.get(key) is not None:
            batch[key] = _pad(batch[key], value)
    if batch.get("position_ids") is not None:
        pos = batch["position_ids"]
        cont = pos[..., -1:] + torch.arange(1, pad_len + 1, device=device).expand(bsz, pad_len)
        batch["position_ids"] = torch.cat([pos, cont], dim=seq_dim)
    if batch.get("attention_mask") is not None:
        batch["attention_mask"] = _pad(batch["attention_mask"], 0)
    return batch, seq_len


def shard_sequence_batch(batch, shard_group, attention=None, seq_dim=1, ignore_index=-100):
    """Contiguously shard a batch's sequence over ``shard_group`` for native Ulysses SP.

    ``prepare_data_loader`` already hands every shard rank the SAME sample (its data-parallel
    sharding divides out tp*cp*sp), so this only does the sequence split: it builds the GLOBAL
    ``position_ids``/``shift_labels`` (computed on the full sequence so shard boundaries keep
    next-token alignment), **right-pads the sequence to a multiple of the shard world size** so the
    shard is even, then shards ``input_ids``/``labels``/``position_ids``/``shift_labels`` along
    ``seq_dim``. Packed/varlen ``cu_seqlens`` are self-derived inside the attention from these sharded
    ``position_ids`` (see :class:`UlyssesAttention`), so there is no side channel here.

    Returns the local shard as a new dict (the input ``batch`` is not mutated).

    Args:
        batch: mapping with at least ``input_ids`` (and optionally ``labels``/``position_ids``).
        shard_group: the sequence-shard (``sp``) process group.
        attention: deprecated / unused; kept for back-compat (varlen is self-derived).
        seq_dim: the sequence dimension. ignore_index: final-token / pad label id.
    """
    rank, world = dist.get_rank(shard_group), dist.get_world_size(shard_group)
    batch = dict(batch)
    if "input_ids" not in batch:
        return batch
    # 1. global position_ids
    if "position_ids" not in batch:
        ids = batch["input_ids"]
        pos = torch.arange(ids.shape[seq_dim], device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        batch["position_ids"] = pos.contiguous()
    # 2. global shift_labels (shift before sharding)
    if "labels" in batch and "shift_labels" not in batch:
        labels = batch["labels"]
        shift = torch.full_like(labels, ignore_index)
        shift[..., :-1] = labels[..., 1:]
        batch["shift_labels"] = shift
    # 3. pad the GLOBAL sequence up to a multiple of the shard world size so `chunk` is even
    # (uneven shards are a hard Ulysses all-to-all size mismatch).
    if world > 1:
        batch, _ = _pad_sequence_to_multiple(batch, world, seq_dim, ignore_index)
    # 4. varlen cu_seqlens are self-derived inside the attention (from the sharded position_ids that
    # travel with the batch), so there is no dataloader -> handler side channel here. ``attention`` is
    # accepted only for back-compat and is unused.
    # 5. contiguous (now EVEN) sequence shard across the group
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
        attention: the attention handler; packed batches push the GLOBAL cu_seqlens onto it
            (``set_varlen``). None to disable varlen.
        seq_dim / ignore_index: sequence dim / final-token label id.
    """

    def __init__(self, dataloader, shard_group, attention=None, seq_dim=1, ignore_index=-100):
        self.dataloader = dataloader
        self.attention = attention
        self.seq_dim = seq_dim
        self.ignore_index = ignore_index
        self.shard_group = shard_group
        self.rank = dist.get_rank(shard_group)
        self.world = dist.get_world_size(shard_group)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield self._process(dict(batch))

    def _process(self, batch):
        # accelerate's prepare_data_loader already hands every sp rank the SAME sample, so this just
        # builds the global position_ids / shift_labels and shards the sequence over the sp group.
        return shard_sequence_batch(
            batch, self.shard_group, attention=self.attention, seq_dim=self.seq_dim, ignore_index=self.ignore_index
        )
