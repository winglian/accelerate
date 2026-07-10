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

"""Native Ulysses SP over sequences whose length is NOT divisible by ``sp_size``.

The Ulysses ``all_to_all_single`` (``GatherSeqScatterHeads``) requires an IDENTICAL
``[b, sp, H/sp, s, d]`` tensor on every rank, so the per-rank sequence shard must be EVEN. A raw
``tensor.chunk(sp_size, dim=seq_dim)`` is uneven when ``seq_len % sp_size != 0`` (the last shard is
shorter), which was a hard distributed-collective size mismatch (gloo ``EnforceNotMet``; NCCL
hangs/corrupts).

``shard_sequence_batch`` now right-pads the global sequence to a multiple of ``sp_size`` (pads
label-masked so they cost no loss and continue ``position_ids`` so they never form a spurious varlen
boundary). These CPU/gloo tests prove the shard is even and the all-to-all no longer breaks for an
indivisible length, plus a fast unit test that the pad is loss-masked.
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


SP_SIZE = 2
NHEADS = 4  # divisible by SP_SIZE (kv-head constraint is orthogonal to the seq-length one tested here)
HEAD_DIM = 8


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank, world, port, timeout_s=30):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=timeout_s))


def _shard_len_worker(rank, world, seq_len, port, q):
    """Report every rank's local shard length after ``shard_sequence_batch``."""
    _init(rank, world, port)
    try:
        from accelerate.utils.sequence_parallel import shard_sequence_batch

        batch = {"input_ids": torch.arange(seq_len).unsqueeze(0)}  # [1, seq_len]
        local = shard_sequence_batch(batch, dist.group.WORLD)["input_ids"].shape[1]
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        if rank == 0:
            q.put(gathered)
    finally:
        dist.destroy_process_group()


def _all_to_all_worker(rank, world, seq_len, port, q):
    """Run the real Ulysses all-to-all on this rank's shard; report OK or the exception."""
    _init(rank, world, port, timeout_s=12)  # short so a stranded peer fails fast instead of hanging
    try:
        from accelerate.utils.sequence_parallel import GatherSeqScatterHeads, shard_sequence_batch

        s = shard_sequence_batch({"input_ids": torch.arange(seq_len).unsqueeze(0)}, dist.group.WORLD)
        s_local = s["input_ids"].shape[1]
        q_bhsd = torch.randn(1, NHEADS, s_local, HEAD_DIM)  # [b, H, s_local, d]
        out = GatherSeqScatterHeads.apply(q_bhsd, dist.group.WORLD, world)
        q.put((rank, "OK", tuple(out.shape)))
    except BaseException as exc:  # noqa: BLE001 - a collective mismatch may surface oddly
        q.put((rank, "RAISED", f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"))
    finally:
        try:
            dist.destroy_process_group()
        except BaseException:  # noqa: BLE001
            pass


def _spawn(target, seq_len, world=SP_SIZE, get_timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=target, args=(r, world, seq_len, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    # Collect reports, but stop as soon as every worker has exited — a collective mismatch aborts a
    # worker (SIGABRT) before it can enqueue, so we must not block for a message that never comes.
    msgs = []
    deadline = time.monotonic() + get_timeout
    while time.monotonic() < deadline and len(msgs) < world:
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


@pytest.mark.parametrize("seq_len", [pytest.param(8, id="divisible"), pytest.param(7, id="indivisible")])
def test_shard_sequence_batch_produces_even_shards(seq_len):
    """Ulysses requires every rank to hold the same shard length, so ``shard_sequence_batch`` pads
    to a multiple of ``sp_size`` before splitting — the shard is even for both a divisible length
    and an indivisible one (padded up)."""
    msgs, _ = _spawn(_shard_len_worker, seq_len)
    assert msgs, "rank 0 did not report shard lengths"
    shard_lens = msgs[0]
    assert len(set(shard_lens)) == 1, f"uneven shards for seq_len={seq_len}: {shard_lens}"


@pytest.mark.parametrize("seq_len", [pytest.param(8, id="divisible"), pytest.param(7, id="indivisible")])
def test_ulysses_all_to_all_handles_indivisible_seq_len(seq_len):
    """The end-to-end consequence of padding: the Ulysses ``all_to_all_single`` now sees a matching
    per-rank tensor size for both a divisible and an indivisible sequence, so every rank succeeds
    (before padding, the indivisible case aborted with a collective size mismatch)."""
    _, exitcodes = _spawn(_all_to_all_worker, seq_len)
    assert all(c == 0 for c in exitcodes), f"seq_len={seq_len} should succeed, got exitcodes {exitcodes}"


def test_pad_masks_loss_and_continues_positions():
    """Fast (no distributed) check of the pad itself: pad tokens are label-masked (``ignore_index``
    in labels/shift_labels so they add no loss and drop out of ``num_items_in_batch``), and
    ``position_ids`` continue past the last real position (no ``0`` reset that would fake a varlen
    doc boundary)."""
    from accelerate.utils.sequence_parallel import _pad_sequence_to_multiple

    seq_len, ignore = 7, -100
    batch = {
        "input_ids": torch.arange(1, seq_len + 1).unsqueeze(0),  # [1,7], nonzero so pad (0) is distinct
        "labels": torch.arange(1, seq_len + 1).unsqueeze(0),
        "shift_labels": torch.arange(1, seq_len + 1).unsqueeze(0),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
    }
    padded, orig = _pad_sequence_to_multiple(dict(batch), multiple=4, seq_dim=1, ignore_index=ignore)
    assert orig == seq_len
    new_len = padded["input_ids"].shape[1]
    assert new_len == 8 and new_len % 4 == 0  # 7 -> 8
    pad = slice(seq_len, new_len)
    assert torch.equal(padded["input_ids"][0, pad], torch.zeros(1, dtype=batch["input_ids"].dtype))
    assert (padded["labels"][0, pad] == ignore).all(), "pad must not contribute to the loss"
    assert (padded["shift_labels"][0, pad] == ignore).all()
    # positions keep increasing (…6 -> 7); a reset to 0 would fabricate a varlen boundary
    assert padded["position_ids"][0, seq_len].item() == seq_len
    assert torch.equal(padded["input_ids"][0, :seq_len], batch["input_ids"][0]), "real tokens untouched"
