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

"""Native Ulysses SP requires the global sequence length to be divisible by ``sp_size``.

``shard_sequence_batch`` splits the sequence with ``tensor.chunk(sp_size, dim=seq_dim)``, which is
UNEVEN when ``seq_len % sp_size != 0`` (torch.chunk makes the last shard shorter). Every rank then
feeds its local shard into the Ulysses ``all_to_all_single`` (``GatherSeqScatterHeads``), which
requires an IDENTICAL ``[b, sp, H/sp, s, d]`` tensor on every rank — so uneven ``s`` is a hard
distributed-collective size mismatch (gloo raises ``EnforceNotMet``; NCCL hangs/corrupts).

The library currently pushes this invariant onto the caller (see ``SEQLEN = 64  # must be divisible
by sp_size`` in ``test_utils/scripts/external_deps/test_accelerate_ulysses_sp.py``) instead of
padding. These CPU/gloo tests prove the failure mode; the first is ``xfail(strict=True)`` so it
flips to a real failure — prompting removal of the marker — once ``shard_sequence_batch`` pads.
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


@pytest.mark.parametrize(
    "seq_len",
    [
        pytest.param(8, id="divisible"),
        pytest.param(
            7,
            id="indivisible",
            marks=pytest.mark.xfail(
                strict=True,
                reason="shard_sequence_batch does not pad; seq_len % sp_size != 0 yields uneven "
                "shards (remove this marker once padding lands)",
            ),
        ),
    ],
)
def test_shard_sequence_batch_produces_even_shards(seq_len):
    """Ulysses requires every rank to hold the same shard length. ``shard_sequence_batch`` must
    therefore split the sequence evenly across ``sp_size`` — it does for a divisible length and
    (currently) does NOT for an indivisible one (``xfail``)."""
    msgs, _ = _spawn(_shard_len_worker, seq_len)
    assert msgs, "rank 0 did not report shard lengths"
    shard_lens = msgs[0]
    assert len(set(shard_lens)) == 1, f"uneven shards for seq_len={seq_len}: {shard_lens}"


def test_ulysses_all_to_all_crashes_on_indivisible_seq_len():
    """The concrete consequence: with an indivisible sequence the Ulysses ``all_to_all_single`` sees
    mismatched per-rank tensor sizes — a hard distributed-collective failure (gloo aborts; NCCL would
    hang/corrupt). The divisible case is the control and succeeds on every rank."""
    _, exitcodes_ok = _spawn(_all_to_all_worker, 8)
    assert all(c == 0 for c in exitcodes_ok), f"divisible seq should succeed, got exitcodes {exitcodes_ok}"

    _, exitcodes_bad = _spawn(_all_to_all_worker, 7)
    assert not all(c == 0 for c in exitcodes_bad), (
        "indivisible seq should break the Ulysses all-to-all, but every rank exited cleanly "
        f"(exitcodes {exitcodes_bad}) — did padding get added?"
    )
