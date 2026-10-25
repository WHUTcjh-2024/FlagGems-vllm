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

"""Triton implementation of vLLM 0.20.2 ``compute_topk_logprobs``.

This is the logprob path used by the V1 sampler and the prompt-logprob worker.
vLLM builds ``[sampled_token_id] + topk(logits)`` with ``torch.topk`` and
``torch.cat`` before running its two Triton kernels. The production path here
must not use torch compute/copy ops, so the top-k selection, the token-id
packing and the rank count are all implemented as kernels:

    _block_topk_kernel        top-k candidates per (row, vocab block)
    _merge_topk_kernel        per-row candidate merge + token-id packing
    _topk_log_softmax_kernel  online logsumexp + gather at the packed ids
    _ranks_kernel             rank of the sampled token

The last two keep the kernel bodies of vLLM 0.29.0 (fused online logsumexp,
full-block/tail split, pipelined sweep) with 0.20.2's two-kernel shape.

Semantics match vLLM 0.20.2 for fp16/bf16/fp32 row-major logits; ``+inf``
logits select normally, NaN logits are unsupported (``tl.max`` has no NaN
ordering, same as the reference kernels). Ties are broken by the lowest vocab
index, which is a valid ``torch.topk`` ordering.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# Vocab tile of the first top-k stage. The candidate workspace layout is
# derived from this value, so it is fixed on the host instead of being a
# tunable knob; only num_warps/num_stages come from tune_configs.yaml.
_BLOCK_TOPK_TILE = 2048
# Candidate tiles are rounded up to a power of two for the merge kernel.
_MIN_MERGE_BLOCK = 128


class LogprobsTensors(NamedTuple):
    """vLLM-compatible fallback when the real vllm package is absent."""

    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor
    cu_num_generated_tokens: list[int] | None = None

    def tolists(self, cu_num_generated_tokens=None):
        from vllm.v1.outputs import LogprobsLists

        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            (
                cu_num_generated_tokens
                if cu_num_generated_tokens is not None
                else self.cu_num_generated_tokens
            ),
        )


def _make_logprobs_tensors(logprob_token_ids, logprobs, token_ranks, cu_num_logits):
    try:
        from vllm.v1.outputs import LogprobsTensors as VllmLogprobsTensors
    except ImportError:
        VllmLogprobsTensors = LogprobsTensors
    return VllmLogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_logprobs_block_topk"),
    key=["vocab_size", "topk"],
    strategy=["log", "log"],
)
@triton.jit
def _block_topk_kernel(
    cand_val_ptr,
    cand_idx_ptr,
    logits_ptr,
    logits_stride0,
    logits_stride1,
    vocab_size,
    topk,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    alive = offs < vocab_size
    vals = tl.load(
        logits_ptr + row * logits_stride0 + offs * logits_stride1,
        mask=alive,
        other=float("-inf"),
    ).to(tl.float32)

    num_blocks = tl.cdiv(vocab_size, BLOCK)
    out = (row * num_blocks + blk) * topk

    for j in tl.range(0, topk):
        best = tl.max(vals)
        # Lowest index among the lanes holding `best`; dead lanes carry the
        # sentinel index so an exhausted block emits nothing selectable.
        cand = tl.where(alive, offs, vocab_size)
        idx = tl.min(tl.where(vals == best, cand, vocab_size))
        tl.store(cand_val_ptr + out + j, best)
        tl.store(cand_idx_ptr + out + j, idx)
        alive = alive & (offs != idx)
        vals = tl.where(alive, vals, float("-inf"))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_logprobs_merge_topk"),
    key=["num_candidates", "topk"],
    # align32 tolerates the num_logprobs == 0 case, where math.log2(0) would
    # blow up the "log" strategy.
    strategy=["align32", "align32"],
)
@triton.jit
def _merge_topk_kernel(
    token_ids_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    sampled_ptr,
    num_candidates,
    vocab_size,
    topk,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    base = row * num_candidates
    live = lane < num_candidates
    vals = tl.load(cand_val_ptr + base + lane, mask=live, other=float("-inf"))
    idxs = tl.load(cand_idx_ptr + base + lane, mask=live, other=vocab_size)
    # Sentinel slots from exhausted blocks never compete. `alive` must be kept
    # separately from the values: once selected, a slot is invalidated in the
    # mask, because its value may legitimately be -inf.
    alive = live & (idxs < vocab_size)

    out = row * (topk + 1)
    tl.store(token_ids_ptr + out, tl.load(sampled_ptr + row).to(tl.int64))
    for j in tl.range(0, topk):
        best = tl.max(tl.where(alive, vals, float("-inf")))
        idx = tl.min(tl.where(alive & (vals == best), idxs, vocab_size))
        tl.store(token_ids_ptr + out + 1 + j, idx.to(tl.int64))
        alive = alive & (idxs != idx)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_logprobs_topk_log_softmax"),
    key=["vocab_size", "topk"],
    strategy=["log", "log"],
)
@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride0,
    logits_stride1,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride0

    full = (vocab_size // BLOCK_SIZE) * BLOCK_SIZE
    max_val = float("-inf")
    se = 0.0
    for i in tl.range(0, full, BLOCK_SIZE, num_stages=2, loop_unroll_factor=2):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block * logits_stride1).to(tl.float32)
        block_max = tl.max(logits)
        new_max = tl.maximum(max_val, block_max)
        # Keep the accumulator at zero until a finite logit appears. An
        # all-masked prefix otherwise evaluates -inf - -inf and poisons se.
        normalizer = tl.where(new_max == float("-inf"), 0.0, new_max)
        se = se * tl.exp(max_val - normalizer) + tl.sum(tl.exp(logits - normalizer))
        max_val = new_max

    if full < vocab_size:
        block = full + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            row_ptr + block * logits_stride1,
            mask=block < vocab_size,
            other=float("-inf"),
        ).to(tl.float32)
        block_max = tl.max(logits)
        new_max = tl.maximum(max_val, block_max)
        normalizer = tl.where(new_max == float("-inf"), 0.0, new_max)
        se = se * tl.exp(max_val - normalizer) + tl.sum(tl.exp(logits - normalizer))
        max_val = new_max

    lse = tl.log(se)

    for j in tl.range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)
        logits = tl.load(row_ptr + ids * logits_stride1, mask=k_mask, other=0.0).to(
            tl.float32
        )
        tl.store(
            output_ptr + req_idx * topk + k_offset,
            logits - max_val - lse,
            mask=k_mask,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_topk_logprobs_ranks"),
    key=["vocab_size"],
    strategy=["log"],
)
@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride0,
    logits_stride1,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride0

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id * logits_stride1)

    full = (vocab_size // BLOCK_SIZE) * BLOCK_SIZE
    n = 0
    for i in tl.range(0, full, BLOCK_SIZE, num_stages=4):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block * logits_stride1)
        n += tl.sum((logits >= x).to(tl.int32))

    if full < vocab_size:
        block = full + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            row_ptr + block * logits_stride1,
            mask=block < vocab_size,
            other=float("-inf"),
        )
        n += tl.sum((logits >= x).to(tl.int32))

    tl.store(output_ptr + req_idx, n.to(tl.int64))


def _validate(logits, num_logprobs, sampled_token_ids):
    if logits.dim() != 2:
        raise ValueError("compute_topk_logprobs requires 2-D [batch, vocab] logits")
    if logits.device.type != "cuda":
        raise NotImplementedError("compute_topk_logprobs requires CUDA tensors")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "compute_topk_logprobs supports FP16/BF16/FP32 logits"
        )
    if any(stride < 0 for stride in logits.stride()):
        raise NotImplementedError("compute_topk_logprobs requires non-negative strides")
    if sampled_token_ids.device != logits.device:
        raise ValueError("sampled_token_ids must be on the same device as logits")
    if sampled_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("sampled_token_ids must be int32 or int64")
    if sampled_token_ids.dim() != 1 or not sampled_token_ids.is_contiguous():
        raise ValueError("sampled_token_ids must be a contiguous 1-D tensor")
    if num_logprobs < 0:
        raise ValueError("num_logprobs must be non-negative")
    if num_logprobs > logits.shape[1]:
        raise ValueError(
            f"num_logprobs ({num_logprobs}) exceeds vocab size ({logits.shape[1]})"
        )


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
):
    """Logprobs of the sampled token plus the ``num_logprobs`` largest logits.

    Mirrors ``vllm.v1.worker.gpu.sample.logprob.compute_topk_logprobs`` from
    vLLM 0.20.2: returns ``[batch, 1 + num_logprobs]`` token ids and fp32
    logprobs, plus the ``>=`` rank of the sampled token in each row.
    """
    _validate(logits, num_logprobs, sampled_token_ids)
    batch_size, vocab_size = logits.shape
    num_cols = num_logprobs + 1
    device = logits.device

    logprob_token_ids = torch.empty(
        (batch_size, num_cols), dtype=torch.int64, device=device
    )
    logprobs = torch.empty((batch_size, num_cols), dtype=torch.float32, device=device)
    token_ranks = torch.empty((batch_size,), dtype=torch.int64, device=device)
    if batch_size == 0:
        return _make_logprobs_tensors(
            logprob_token_ids, logprobs, token_ranks, cu_num_logits
        )

    num_blocks = triton.cdiv(vocab_size, _BLOCK_TOPK_TILE)
    num_candidates = num_blocks * num_logprobs
    # A zero-size candidate buffer would make the masked merge load read a
    # null pointer, so keep at least one (never selected) slot.
    cand_val = torch.empty(
        (batch_size * max(num_candidates, 1),), dtype=torch.float32, device=device
    )
    cand_idx = torch.empty(
        (batch_size * max(num_candidates, 1),), dtype=torch.int32, device=device
    )

    with torch.cuda.device(device):
        if num_logprobs > 0:
            _block_topk_kernel[(batch_size, num_blocks)](
                cand_val,
                cand_idx,
                logits,
                logits.stride(0),
                logits.stride(1),
                vocab_size,
                num_logprobs,
                BLOCK=_BLOCK_TOPK_TILE,
            )
        merge_block = max(
            triton.next_power_of_2(max(num_candidates, 1)), _MIN_MERGE_BLOCK
        )
        _merge_topk_kernel[(batch_size,)](
            logprob_token_ids,
            cand_val,
            cand_idx,
            sampled_token_ids,
            num_candidates,
            vocab_size,
            num_logprobs,
            BLOCK=merge_block,
        )
        _topk_log_softmax_kernel[(batch_size,)](
            logprobs,
            logits,
            logits.stride(0),
            logits.stride(1),
            logprob_token_ids,
            num_cols,
            vocab_size,
        )
        _ranks_kernel[(batch_size,)](
            token_ranks,
            logits,
            logits.stride(0),
            logits.stride(1),
            sampled_token_ids,
            vocab_size,
        )

    return _make_logprobs_tensors(
        logprob_token_ids, logprobs, token_ranks, cu_num_logits
    )


__all__ = ["compute_topk_logprobs", "LogprobsTensors"]
