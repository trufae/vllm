# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable Humming route alignment without native CUDA extensions or CPU reads."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _count_routes(
    Ids,
    Ranks,
    Counts,
    Rows,
    Experts,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    CHUNK: tl.constexpr,
    INIT: tl.constexpr,
):
    expert = tl.program_id(0)
    row = tl.arange(0, CHUNK)
    ids = tl.load(Ids + row, row < N, other=-1)
    matches = (row < N) & (ids == expert)
    ranks = tl.cumsum(matches.to(tl.int32)) - 1
    tl.store(Ranks + row, ranks, matches)
    tl.store(Counts + expert, tl.sum(matches.to(tl.int32)))
    # Disjoint initialization; the next launch scatters real routes over it.
    init = expert * INIT + tl.arange(0, INIT)
    tl.store(Rows + init, N, init < CAPACITY)
    tl.store(Experts + init, -1, init < tl.cdiv(CAPACITY, BLOCK))


@triton.jit
def _finish_decode(
    Ids,
    Ranks,
    Counts,
    Rows,
    Experts,
    Total,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    EP: tl.constexpr,
    RP: tl.constexpr,
):
    expert = tl.arange(0, EP)
    counts = tl.load(Counts + expert, expert < E, other=0)
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    starts = tl.cumsum(padded) - padded
    tl.store(Total, tl.sum(padded))
    row = tl.arange(0, RP)
    ids = tl.load(Ids + row, row < N, other=-1)
    valid = (row < N) & (ids >= 0) & (ids < E)
    rank = tl.load(Ranks + row, valid, other=0)
    start = tl.gather(
        starts, tl.minimum(tl.maximum(ids, 0), E - 1).to(tl.int32), axis=0
    )
    dest = start + rank
    tl.store(Rows + dest, row, valid)
    tl.store(Experts + dest // BLOCK, ids, valid & (rank % BLOCK == 0))


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group routed rows stably by expert, padding each group to `block_size`.

    Invalid expert IDs are excluded. Padding rows contain `topk_ids.numel()`;
    unused expert blocks contain -1. Output shapes depend only on metadata,
    and the padded row count stays on the GPU for CUDA graph replay.
    """
    if not ignore_invalid_experts or block_size <= 0 or num_experts <= 0:
        raise ValueError(
            "Thor alignment requires positive sizes and ignore_invalid_experts=True"
        )
    if not topk_ids.is_cuda:
        raise ValueError("Triton alignment requires CUDA route indices")
    flat = topk_ids.reshape(-1)
    count = flat.numel()
    capacity = count + num_experts * (block_size - 1)
    if count < num_experts:
        capacity = min(count * block_size, capacity)
    rows = torch.empty(capacity, dtype=torch.int32, device=flat.device)
    experts = torch.empty(
        triton.cdiv(capacity, block_size), dtype=torch.int32, device=flat.device
    )
    if not count:
        return rows, experts, torch.zeros(1, dtype=torch.int32, device=flat.device)
    total = torch.empty(1, dtype=torch.int32, device=flat.device)
    if count > 256:
        raise ValueError("The low-M alignment kernel supports at most 256 routed rows")
    ranks = torch.empty(count, dtype=torch.int32, device=flat.device)
    counts = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    with torch.accelerator.device_index(flat.device.index):
        _count_routes[(num_experts,)](
            flat,
            ranks,
            counts,
            rows,
            experts,
            count,
            num_experts,
            block_size,
            capacity,
            triton.next_power_of_2(count),
            triton.next_power_of_2(triton.cdiv(capacity, num_experts)),
        )
        _finish_decode[(1,)](
            flat,
            ranks,
            counts,
            rows,
            experts,
            total,
            count,
            num_experts,
            block_size,
            triton.next_power_of_2(num_experts),
            triton.next_power_of_2(count),
        )
    return rows, experts, total
