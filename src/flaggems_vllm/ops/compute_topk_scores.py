# Copyright 2026 FlagOS Contributors
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

"""Triton implementation of vLLM's ``compute_topk_scores``.

Operator boundary
-----------------
Inputs (identical to ``vllm.v1.worker.gpu.sample.logprob.compute_topk_scores``
in vLLM 0.29.0):

* ``logits``: 2D ``[batch, vocab]`` float16 / bfloat16 / float32 tensor with a
  contiguous innermost dimension.
* ``num_logprobs``: ``0 <= k <= 256``.
* ``sampled_token_ids``: 1D int32/int64 tensor, one id per row.
* optional ``cu_num_logits`` (list or tensor), ``logprob_token_ids_state`` +
  ``expanded_idx_mapping`` for per-request custom token ids, and
  ``logits_mode`` for raw-logit scores.

Output: ``LogprobsTensors`` whose column 0 is the sampled token, columns
``1..k`` are the top-k ids in descending order, ``logprobs`` are log-softmax
values (raw logits when ``logits_mode``), and ``selected_token_ranks`` counts
the logits ``>=`` the sampled logit.

Forward inference only.  ``k > 256``, non-float logits, a non-contiguous
innermost stride, an empty vocabulary or a non-int32/int64 sampled-id dtype are
rejected with ``NotImplementedError``/``ValueError``.

Design
------
No torch compute op is used on the production path.

* One streaming kernel per (row, vocab split) computes the online row max,
  the log-sum-exp and the sampled-token rank, together with a 256-bin
  histogram of the top 8 bits of an order-preserving fp32 key.
* A per-row kernel combines the splits, resolves the threshold bin and emits
  the log-softmax scores.  When a split is small enough its whole threshold
  bin is gathered as candidates; otherwise one refinement level (the next
  8 key bits) narrows the group first.
* A final per-row kernel selects the top-k candidates by value and writes the
  sorted ids and scores.  fp16/bf16 keys have 16 significant bits, so the
  two histogram levels make the candidate group exactly value-tied; fp32
  candidates are compared by value in the final selection.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_BINS = 256
_MAX_TOPK = 256
_MAX_SPLITS = 32
_TARGET_CTAS = 1024
_GATHER_BLOCK = 4096


@triton.jit
def _sortable_key(x):
    """Map floats to monotonically ordered uint32 keys."""
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    is_neg = (bits & sign_mask) != 0
    return tl.where(is_neg, ~bits, bits | sign_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_scores_scan"),
    key=["vocab_size"],
    strategy=["log"],
)
@triton.jit
def _scan_level1_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    hist_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_rank_ptr,
    vocab_size,
    chunk,
    COMPUTE_LSE: tl.constexpr,
    COMPUTE_HIST: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    out_idx = row * tl.num_programs(1) + split

    token = tl.load(sampled_ptr + row).to(tl.int64)
    token_ok = (token >= 0) & (token < vocab_size)
    sampled_val = tl.load(row_ptr + token, mask=token_ok, other=float("inf")).to(
        tl.float32
    )

    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    hist = tl.zeros([BINS], dtype=tl.int32)
    padding = 0
    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if COMPUTE_LSE:
            tile_max = tl.max(x)
            new_max = tl.maximum(running_max, tile_max)
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            contrib = tl.where(new_max == -float("inf"), 0.0, tl.exp(x - new_max))
            contrib = tl.where(mask, contrib, 0.0)
            running_sum = running_sum * scale + tl.sum(contrib)
            running_max = new_max
        rank += tl.sum(((x >= sampled_val) & mask).to(tl.int32))
        if COMPUTE_HIST:
            hist += tl.histogram((_sortable_key(x) >> 24).to(tl.int32), BINS)
            padding += tl.sum((~mask).to(tl.int32))

    # Masked lanes carry -inf, whose key lands in bin 0; remove them again.
    hist -= tl.where(tl.arange(0, BINS) == 0, padding, 0)
    tl.store(hist_ptr + out_idx * BINS + tl.arange(0, BINS), hist)
    tl.store(partial_max_ptr + out_idx, running_max)
    tl.store(partial_sum_ptr + out_idx, running_sum)
    tl.store(partial_rank_ptr + out_idx, rank)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_scores_scan"),
    key=["vocab_size"],
    strategy=["log"],
)
@triton.jit
def _scan_level2_kernel(
    logits_ptr,
    logits_stride,
    hist_ptr,
    threshold1_ptr,
    refine_ptr,
    vocab_size,
    chunk,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(refine_ptr + row) == 0:
        return
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    out_idx = row * tl.num_programs(1) + split
    parent = tl.load(threshold1_ptr + row)

    hist = tl.zeros([BINS], dtype=tl.int32)
    padding = 0
    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf"))
        key = _sortable_key(x)
        in_group = mask & ((key >> 24).to(tl.int32) == parent)
        hist += tl.histogram(
            tl.where(in_group, ((key >> 16) & 0xFF).to(tl.int32), 0), BINS
        )
        padding += tl.sum((~in_group).to(tl.int32))

    # Non-group lanes were folded into bin 0; remove them again.
    hist -= tl.where(tl.arange(0, BINS) == 0, padding, 0)
    tl.store(hist_ptr + out_idx * BINS + tl.arange(0, BINS), hist)


@triton.jit
def _combine_level1_kernel(
    hist_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_rank_ptr,
    lse_ptr,
    rank_ptr,
    threshold_ptr,
    need_ptr,
    group_ptr,
    above_ptr,
    refine_ptr,
    selected_count_ptr,
    group_count_ptr,
    num_splits,
    topk,
    capacity,
    COMPUTE_LSE: tl.constexpr,
    BINS: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, BINS)

    total = tl.zeros([BINS], dtype=tl.int32)
    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    for split in tl.range(0, num_splits):
        out_idx = row * num_splits + split
        total += tl.load(hist_ptr + out_idx * BINS + bins)
        rank += tl.load(partial_rank_ptr + out_idx)
        if COMPUTE_LSE:
            part_max = tl.load(partial_max_ptr + out_idx)
            part_sum = tl.load(partial_sum_ptr + out_idx)
            new_max = tl.maximum(running_max, part_max)
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            grow = tl.where(new_max == -float("inf"), 0.0, tl.exp(part_max - new_max))
            running_sum = running_sum * scale + part_sum * grow
            running_max = new_max

    if COMPUTE_LSE:
        lse = running_max + tl.log(running_sum)
    else:
        lse = 0.0
    tl.store(lse_ptr + row, lse)
    tl.store(rank_ptr + row, rank.to(tl.int64))

    if topk > 0:
        # cum[i] counts keys in bins >= i.
        cum = tl.cumsum(total, axis=0, reverse=True)
        is_threshold = (cum >= topk) & ((cum - total) < topk)
        threshold = tl.argmax(is_threshold.to(tl.int32), axis=0)
        above = tl.sum(tl.where(bins > threshold, total, 0))
        in_bin = tl.sum(tl.where(bins == threshold, total, 0))
        need = topk - above
        tl.store(threshold_ptr + row, threshold)
        tl.store(need_ptr + row, need)
        tl.store(group_ptr + row, in_bin)
        tl.store(above_ptr + row, above)
        tl.store(
            refine_ptr + row,
            (need < in_bin).to(tl.int32) & (in_bin > capacity).to(tl.int32),
        )
    else:
        tl.store(threshold_ptr + row, 0)
        tl.store(need_ptr + row, 0)
        tl.store(group_ptr + row, 0)
        tl.store(above_ptr + row, 0)
        tl.store(refine_ptr + row, 0)

    tl.store(selected_count_ptr + row, 0)
    tl.store(group_count_ptr + row, 0)


@triton.jit
def _combine_level2_kernel(
    hist_ptr,
    threshold1_ptr,
    need1_ptr,
    group1_ptr,
    above1_ptr,
    refine1_ptr,
    threshold2_ptr,
    need2_ptr,
    group2_ptr,
    above2_ptr,
    num_splits,
    BINS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(refine1_ptr + row) == 0:
        tl.store(threshold2_ptr + row, 0)
        tl.store(need2_ptr + row, tl.load(need1_ptr + row))
        tl.store(group2_ptr + row, tl.load(group1_ptr + row))
        tl.store(above2_ptr + row, 0)
        return

    bins = tl.arange(0, BINS)
    total = tl.zeros([BINS], dtype=tl.int32)
    for split in tl.range(0, num_splits):
        total += tl.load(hist_ptr + (row * num_splits + split) * BINS + bins)

    need = tl.load(need1_ptr + row)
    cum = tl.cumsum(total, axis=0, reverse=True)
    is_threshold = (cum >= need) & ((cum - total) < need)
    threshold = tl.argmax(is_threshold.to(tl.int32), axis=0)
    above = tl.sum(tl.where(bins > threshold, total, 0))
    in_bin = tl.sum(tl.where(bins == threshold, total, 0))
    tl.store(threshold2_ptr + row, threshold)
    tl.store(need2_ptr + row, need - above)
    tl.store(group2_ptr + row, in_bin)
    tl.store(above2_ptr + row, above)


@triton.jit
def _gather_kernel(
    logits_ptr,
    logits_stride,
    threshold1_ptr,
    threshold2_ptr,
    refine1_ptr,
    selected_count_ptr,
    group_count_ptr,
    candidate_ptr,
    vocab_size,
    chunk,
    topk,
    capacity,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    out_row = candidate_ptr + row * (topk + capacity)

    thr1 = tl.load(threshold1_ptr + row)
    thr2 = tl.load(threshold2_ptr + row)
    refine1 = tl.load(refine1_ptr + row)

    for off in tl.range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf"))
        key = _sortable_key(x)
        level1 = (key >> 24).to(tl.int32)
        selected = tl.zeros([BLOCK], dtype=tl.int1)
        group = tl.zeros([BLOCK], dtype=tl.int1)
        if refine1 == 0:
            selected = mask & (level1 > thr1)
            group = mask & (level1 == thr1)
        else:
            in_bin1 = mask & (level1 == thr1)
            level2 = ((key >> 16) & 0xFF).to(tl.int32)
            selected = mask & ((level1 > thr1) | (in_bin1 & (level2 > thr2)))
            group = in_bin1 & (level2 == thr2)

        selected_i = selected.to(tl.int32)
        base = tl.atomic_add(selected_count_ptr + row, tl.sum(selected_i))
        positions = base + tl.cumsum(selected_i, axis=0) - 1
        tl.store(out_row + positions, idx, mask=selected & (positions < topk))

        group_i = group.to(tl.int32)
        group_total = tl.sum(group_i)
        group_base = tl.atomic_add(group_count_ptr + row, group_total)
        group_pos = group_base + tl.cumsum(group_i, axis=0) - 1
        tl.store(
            out_row + topk + group_pos,
            idx,
            mask=group & (group_pos < capacity),
        )


@triton.jit
def _select_partial_kernel(
    candidate_ptr,
    logits_ptr,
    logits_stride,
    above1_ptr,
    above2_ptr,
    group1_ptr,
    group2_ptr,
    refine1_ptr,
    partial_ptr,
    topk,
    capacity,
    TOPK: tl.constexpr,
    NUM_CAND: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    above = tl.load(above1_ptr + row)
    if tl.load(refine1_ptr + row) != 0:
        above += tl.load(above2_ptr + row)
        group = tl.minimum(tl.load(group2_ptr + row), capacity)
    else:
        group = tl.minimum(tl.load(group1_ptr + row), capacity)

    lane = part * CHUNK + tl.arange(0, CHUNK)
    valid = (lane < above) | ((lane >= TOPK) & (lane < TOPK + group))
    ids = tl.load(
        candidate_ptr + row * (TOPK + capacity) + lane, mask=valid, other=0
    ).to(tl.int64)
    vals = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + ids,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)

    out_row = partial_ptr + (row * tl.num_programs(1) + part) * TOPK
    idx = tl.arange(0, CHUNK)
    for j in tl.static_range(TOPK):
        best = tl.argmax(vals, axis=0)
        best_val = tl.max(vals, axis=0)
        best_id = tl.sum(tl.where(idx == best, ids, 0))
        best_id = tl.where(best_val == float("-inf"), -1, best_id)
        tl.store(out_row + j, best_id)
        vals = tl.where(idx == best, float("-inf"), vals)


@triton.jit
def _merge_kernel(
    partial_ptr,
    logits_ptr,
    logits_stride,
    sampled_ptr,
    lse_ptr,
    topk_ids_ptr,
    out_ids_ptr,
    out_scores_ptr,
    topk,
    LOGITS_MODE: tl.constexpr,
    OUT_I64: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_PART: tl.constexpr,
    NUM_CAND: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, NUM_CAND)
    valid = idx < NUM_PART * TOPK
    ids = tl.load(partial_ptr + row * (NUM_PART * TOPK) + idx, mask=valid, other=-1).to(
        tl.int64
    )
    valid = valid & (ids >= 0)
    vals = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + ids,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)

    out_row = topk_ids_ptr + row * TOPK
    for j in tl.static_range(TOPK):
        best = tl.argmax(vals, axis=0)
        best_id = tl.sum(tl.where(idx == best, ids, 0))
        best_val = tl.max(vals, axis=0)
        tl.store(out_row + j, best_id)
        if WRITE_SCORES:
            if LOGITS_MODE:
                score = best_val
            else:
                score = best_val - tl.load(lse_ptr + row)
            out_idx = row * (TOPK + 1) + j + 1
            if OUT_I64:
                tl.store(out_ids_ptr + out_idx, best_id)
            else:
                tl.store(out_ids_ptr + out_idx, best_id.to(tl.int32))
            tl.store(out_scores_ptr + out_idx, score)
        vals = tl.where(idx == best, float("-inf"), vals)

    if WRITE_SCORES:
        sampled = tl.load(sampled_ptr + row).to(tl.int64)
        x = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + sampled).to(
            tl.float32
        )
        if LOGITS_MODE:
            score = x
        else:
            score = x - tl.load(lse_ptr + row)
        out_idx = row * (TOPK + 1)
        if OUT_I64:
            tl.store(out_ids_ptr + out_idx, sampled)
        else:
            tl.store(out_ids_ptr + out_idx, sampled.to(tl.int32))
        tl.store(out_scores_ptr + out_idx, score)


@triton.jit
def _sample_row_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    rank_ptr,
    out_ids_ptr,
    out_scores_ptr,
    vocab_size,
    LOGITS_MODE: tl.constexpr,
    OUT_I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One-program-per-row scan used when k == 0 (no selection needed)."""
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * logits_stride
    token = tl.load(sampled_ptr + row).to(tl.int64)
    token_ok = (token >= 0) & (token < vocab_size)
    sampled_val = tl.load(row_ptr + token, mask=token_ok, other=float("inf")).to(
        tl.float32
    )

    running_max = -float("inf")
    running_sum = 0.0
    rank = 0
    for off in tl.range(0, vocab_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < vocab_size
        x = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
        if not LOGITS_MODE:
            new_max = tl.maximum(running_max, tl.max(x))
            scale = tl.where(
                running_max == -float("inf"), 0.0, tl.exp(running_max - new_max)
            )
            contrib = tl.where(new_max == -float("inf"), 0.0, tl.exp(x - new_max))
            running_sum = running_sum * scale + tl.sum(tl.where(mask, contrib, 0.0))
            running_max = new_max
        rank += tl.sum(((x >= sampled_val) & mask).to(tl.int32))

    if LOGITS_MODE:
        score = sampled_val
    else:
        score = sampled_val - (running_max + tl.log(running_sum))
    tl.store(rank_ptr + row, rank.to(tl.int64))
    if OUT_I64:
        tl.store(out_ids_ptr + row, token)
    else:
        tl.store(out_ids_ptr + row, token.to(tl.int32))
    tl.store(out_scores_ptr + row, score)


@triton.jit
def _sample_only_kernel(
    logits_ptr,
    logits_stride,
    sampled_ptr,
    lse_ptr,
    out_ids_ptr,
    out_scores_ptr,
    LOGITS_MODE: tl.constexpr,
    OUT_I64: tl.constexpr,
):
    row = tl.program_id(0)
    sampled = tl.load(sampled_ptr + row).to(tl.int64)
    x = tl.load(logits_ptr + row.to(tl.int64) * logits_stride + sampled).to(tl.float32)
    if LOGITS_MODE:
        score = x
    else:
        score = x - tl.load(lse_ptr + row)
    if OUT_I64:
        tl.store(out_ids_ptr + row, sampled)
    else:
        tl.store(out_ids_ptr + row, sampled.to(tl.int32))
    tl.store(out_scores_ptr + row, score)


@triton.jit
def _fill_custom_ids_kernel(
    out_token_ids_ptr,
    out_token_ids_stride,
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_ptr,
    topk_ids_ptr,
    topk_ids_stride,
    expanded_idx_mapping_ptr,
    num_custom_ptr,
    custom_ids_ptr,
    custom_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_I64: tl.constexpr,
):
    """Adapted from vLLM 0.29.0 ``_fill_logprob_token_ids_kernel``."""
    row = tl.program_id(0)

    sampled = tl.load(sampled_ptr + row)
    if OUT_I64:
        tl.store(out_token_ids_ptr + row * out_token_ids_stride, sampled.to(tl.int64))
    else:
        tl.store(out_token_ids_ptr + row * out_token_ids_stride, sampled.to(tl.int32))
    tl.store(out_valid_mask_ptr + row * out_valid_mask_stride, 1)

    req_idx = tl.load(expanded_idx_mapping_ptr + row)
    num_custom = tl.load(num_custom_ptr + req_idx)

    col = tl.arange(0, PADDED_COLS)
    ids_base = out_token_ids_ptr + row * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + row * out_valid_mask_stride + 1

    # Default every padded column to (id 0, invalid) so the score gather never
    # reads an uninitialized id.  ids_base already points at column 1.
    in_row = col < OUT_WIDTH - 1
    if OUT_I64:
        tl.store(ids_base + col, tl.zeros([PADDED_COLS], dtype=tl.int64), mask=in_row)
    else:
        tl.store(ids_base + col, tl.zeros([PADDED_COLS], dtype=tl.int32), mask=in_row)
    tl.store(mask_base + col, tl.zeros([PADDED_COLS], dtype=tl.int1), mask=in_row)

    if num_custom > 0:
        src = custom_ids_ptr + req_idx * custom_ids_stride
        valid = col < num_custom
    else:
        src = topk_ids_ptr + row * topk_ids_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    if OUT_I64:
        tl.store(ids_base + col, tokens, mask=valid)
    else:
        tl.store(ids_base + col, tokens.to(tl.int32), mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


@triton.jit
def _custom_scores_kernel(
    logits_ptr,
    logits_stride,
    ids_ptr,
    valid_mask_ptr,
    lse_ptr,
    scores_ptr,
    num_cols,
    LOGITS_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < num_cols
    ids = tl.load(ids_ptr + row * num_cols + cols, mask=mask, other=0).to(tl.int64)
    valid = tl.load(valid_mask_ptr + row * num_cols + cols, mask=mask, other=0)
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_stride + ids, mask=mask, other=0.0
    ).to(tl.float32)
    if LOGITS_MODE:
        score = x
    else:
        score = x - tl.load(lse_ptr + row)
    score = tl.where(valid, score, float("-inf"))
    tl.store(scores_ptr + row * num_cols + cols, score, mask=mask)


def _choose_splits(batch_size, vocab_size, topk):
    """Programs per row, calibrated on H20.

    Small vocabularies are cheap enough that one program per row is best; wider
    rows need the vocab split to fill the GPU.  Without top-k there is much less
    work, so the fused path stays preferable up to a larger batch.
    """
    if batch_size <= 0:
        return 1
    if topk == 0 and vocab_size <= 65536:
        # No selection: one fused program per row is cheapest.
        return 1
    target = min(_MAX_SPLITS, max(2, _TARGET_CTAS // batch_size))
    return max(1, min(target, vocab_size // 8192))


def _candidate_capacity(vocab_size):
    # Roughly one octave of logits holds ~vocab/42 entries near the k-th value;
    # leave a wide margin so the common path only needs the first level.
    return min(8192, max(1024, vocab_size // 16))


def _select_topk(
    logits, sampled_token_ids, topk, out_ids, out_scores, write_scores, logits_mode
):
    batch_size, vocab_size = logits.shape
    device = logits.device
    splits = _choose_splits(batch_size, vocab_size, topk)
    chunk = triton.cdiv(vocab_size, splits)
    capacity = min(_candidate_capacity(vocab_size), vocab_size)

    hist1 = torch.empty((batch_size, splits, _BINS), dtype=torch.int32, device=device)
    partial_max = torch.empty((batch_size, splits), dtype=torch.float32, device=device)
    partial_sum = torch.empty((batch_size, splits), dtype=torch.float32, device=device)
    partial_rank = torch.empty((batch_size, splits), dtype=torch.int32, device=device)

    lse = torch.empty((batch_size,), dtype=torch.float32, device=device)
    ranks = torch.empty((batch_size,), dtype=torch.int64, device=device)
    threshold1 = torch.empty((batch_size,), dtype=torch.int32, device=device)
    need1 = torch.empty((batch_size,), dtype=torch.int32, device=device)
    group1 = torch.empty((batch_size,), dtype=torch.int32, device=device)
    above1 = torch.empty((batch_size,), dtype=torch.int32, device=device)
    refine1 = torch.empty((batch_size,), dtype=torch.int32, device=device)
    selected_count = torch.empty((batch_size,), dtype=torch.int32, device=device)
    group_count = torch.empty((batch_size,), dtype=torch.int32, device=device)

    _scan_level1_kernel[(batch_size, splits)](
        logits,
        logits.stride(0),
        sampled_token_ids,
        hist1,
        partial_max,
        partial_sum,
        partial_rank,
        vocab_size,
        chunk,
        COMPUTE_LSE=not logits_mode,
        COMPUTE_HIST=topk > 0,
        BINS=_BINS,
    )
    _combine_level1_kernel[(batch_size,)](
        hist1,
        partial_max,
        partial_sum,
        partial_rank,
        lse,
        ranks,
        threshold1,
        need1,
        group1,
        above1,
        refine1,
        selected_count,
        group_count,
        splits,
        topk,
        capacity,
        COMPUTE_LSE=not logits_mode,
        BINS=_BINS,
    )

    topk_ids = None
    if topk > 0:
        hist2 = torch.empty(
            (batch_size, splits, _BINS), dtype=torch.int32, device=device
        )
        threshold2 = torch.empty((batch_size,), dtype=torch.int32, device=device)
        need2 = torch.empty((batch_size,), dtype=torch.int32, device=device)
        group2 = torch.empty((batch_size,), dtype=torch.int32, device=device)
        above2 = torch.empty((batch_size,), dtype=torch.int32, device=device)

        _scan_level2_kernel[(batch_size, splits)](
            logits,
            logits.stride(0),
            hist2,
            threshold1,
            refine1,
            vocab_size,
            chunk,
            BINS=_BINS,
        )
        _combine_level2_kernel[(batch_size,)](
            hist2,
            threshold1,
            need1,
            group1,
            above1,
            refine1,
            threshold2,
            need2,
            group2,
            above2,
            splits,
            BINS=_BINS,
        )

        candidate = torch.empty(
            (batch_size, topk + capacity), dtype=torch.int32, device=device
        )
        _gather_kernel[(batch_size, splits)](
            logits,
            logits.stride(0),
            threshold1,
            threshold2,
            refine1,
            selected_count,
            group_count,
            candidate,
            vocab_size,
            chunk,
            topk,
            capacity,
            BLOCK=_GATHER_BLOCK,
            num_warps=8,
        )

        topk_ids = torch.empty((batch_size, topk), dtype=torch.int32, device=device)
        num_cand = triton.next_power_of_2(topk + capacity)
        num_part = min(16, max(1, triton.next_power_of_2(max(1, capacity // 512))))
        chunk = num_cand // num_part
        partial = torch.empty(
            (batch_size, num_part, topk), dtype=torch.int32, device=device
        )
        _select_partial_kernel[(batch_size, num_part)](
            candidate,
            logits,
            logits.stride(0),
            above1,
            above2,
            group1,
            group2,
            refine1,
            partial,
            topk,
            capacity,
            TOPK=topk,
            NUM_CAND=num_cand,
            CHUNK=chunk,
        )
        _merge_kernel[(batch_size,)](
            partial,
            logits,
            logits.stride(0),
            sampled_token_ids,
            lse,
            topk_ids,
            out_ids,
            out_scores,
            topk,
            LOGITS_MODE=logits_mode,
            OUT_I64=sampled_token_ids.dtype == torch.int64,
            WRITE_SCORES=write_scores,
            TOPK=topk,
            NUM_PART=num_part,
            NUM_CAND=triton.next_power_of_2(num_part * topk),
        )

    return lse, ranks, topk_ids


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
):
    """Return the top-k logprobs of ``logits`` for the sampled tokens."""
    from vllm.v1.outputs import LogprobsTensors

    assert num_logprobs >= 0
    if logits.dim() != 2:
        raise NotImplementedError("compute_topk_scores expects 2D logits")
    if logits.stride(-1) != 1:
        raise NotImplementedError("only contiguous rows are supported")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"unsupported logits dtype: {logits.dtype}")
    if sampled_token_ids.numel() != logits.shape[0]:
        raise ValueError("sampled_token_ids must have one entry per row")
    if sampled_token_ids.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError(
            f"unsupported sampled_token_ids dtype: {sampled_token_ids.dtype}"
        )
    if sampled_token_ids.device != logits.device:
        raise ValueError("logits and sampled_token_ids must be on one device")
    if num_logprobs > _MAX_TOPK:
        raise NotImplementedError(
            f"num_logprobs={num_logprobs} exceeds the supported maximum "
            f"({_MAX_TOPK}); this operator is forward inference only"
        )

    batch_size, vocab_size = logits.shape
    device = logits.device
    out_i64 = sampled_token_ids.dtype == torch.int64

    if batch_size == 0:
        width = 1 + max(num_logprobs, max_per_req_token_ids)
        return LogprobsTensors(
            logprob_token_ids=torch.empty(
                (0, width), dtype=sampled_token_ids.dtype, device=device
            ),
            logprobs=torch.empty((0, width), dtype=torch.float32, device=device),
            selected_token_ranks=torch.empty((0,), dtype=torch.int64, device=device),
            cu_num_generated_tokens=(
                None if isinstance(cu_num_logits, torch.Tensor) else cu_num_logits
            ),
            cu_num_generated_tokens_tensor=(
                cu_num_logits if isinstance(cu_num_logits, torch.Tensor) else None
            ),
        )
    if vocab_size == 0:
        raise NotImplementedError("compute_topk_scores requires vocab_size > 0")
    if num_logprobs > vocab_size:
        raise ValueError("num_logprobs cannot exceed vocab_size")

    if max_per_req_token_ids == 0:
        width = num_logprobs + 1
        out_ids = torch.empty(
            (batch_size, width), dtype=sampled_token_ids.dtype, device=device
        )
        scores = torch.empty((batch_size, width), dtype=torch.float32, device=device)
        if num_logprobs == 0 and _choose_splits(batch_size, vocab_size, 0) == 1:
            ranks = torch.empty((batch_size,), dtype=torch.int64, device=device)
            _sample_row_kernel[(batch_size,)](
                logits,
                logits.stride(0),
                sampled_token_ids,
                ranks,
                out_ids,
                scores,
                vocab_size,
                LOGITS_MODE=logits_mode,
                OUT_I64=out_i64,
                BLOCK=8192,
                num_warps=8,
            )
        else:
            lse, ranks, _ = _select_topk(
                logits,
                sampled_token_ids,
                num_logprobs,
                out_ids,
                scores,
                True,
                logits_mode,
            )
            if num_logprobs == 0:
                _sample_only_kernel[(batch_size,)](
                    logits,
                    logits.stride(0),
                    sampled_token_ids,
                    lse,
                    out_ids,
                    scores,
                    LOGITS_MODE=logits_mode,
                    OUT_I64=out_i64,
                )
    else:
        if logprob_token_ids_state is None or expanded_idx_mapping is None:
            raise ValueError(
                "logprob_token_ids_state and expanded_idx_mapping are required "
                "when max_per_req_token_ids > 0"
            )
        num_cols = max(num_logprobs, max_per_req_token_ids)
        width = num_cols + 1
        state_ids = logprob_token_ids_state.token_ids.gpu
        out_ids = torch.empty(
            (batch_size, width), dtype=sampled_token_ids.dtype, device=device
        )
        valid_mask = torch.empty((batch_size, width), dtype=torch.bool, device=device)
        scores = torch.empty((batch_size, width), dtype=torch.float32, device=device)
        lse, ranks, topk_ids = _select_topk(
            logits, sampled_token_ids, num_logprobs, out_ids, scores, False, logits_mode
        )
        _fill_custom_ids_kernel[(batch_size,)](
            out_ids,
            out_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_ids if topk_ids is not None else state_ids,
            topk_ids.stride(0) if topk_ids is not None else state_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            state_ids,
            state_ids.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(width),
            OUT_WIDTH=width,
            OUT_I64=out_i64,
        )
        _custom_scores_kernel[(batch_size, triton.cdiv(width, 128))](
            logits,
            logits.stride(0),
            out_ids,
            valid_mask,
            lse,
            scores,
            width,
            LOGITS_MODE=logits_mode,
            BLOCK=128,
        )

    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=out_ids,
        logprobs=scores,
        selected_token_ranks=ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )
