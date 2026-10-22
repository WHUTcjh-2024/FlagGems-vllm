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

"""Accuracy tests for ``flaggems_vllm.compute_topk_scores``.

The reference is vLLM's own ``compute_topk_scores``.  Ties are expected with
low-precision logits, so token ids are compared as a value multiset; ranks and
the sampled-token column are compared exactly.
"""

import math

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

device = flaggems_vllm.device

try:
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores as _vllm_fn

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    _vllm_fn = None

pytestmark = pytest.mark.skipif(
    not HAS_VLLM or not torch.cuda.is_available(), reason="need vLLM + CUDA"
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

if cfg.QUICK_MODE:
    SHAPES = [(1, 1024), (8, 32768), (64, 131072)]
    KS = [0, 1, 5, 50]
else:
    SHAPES = [
        (1, 999),
        (1, 32768),
        (4, 65536),
        (16, 129280),
        (64, 131072),
        (256, 151936),
    ]
    KS = [0, 1, 5, 10, 20, 50]


def _make_inputs(batch, vocab, dtype, seed=0):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(batch, vocab, dtype=dtype, device=device, generator=generator)
    sampled = torch.randint(
        vocab, (batch,), dtype=torch.int64, device=device, generator=generator
    )
    return logits, sampled


def _row_values(logits, ids):
    return torch.gather(logits, 1, ids.long()).float()


def _sorted_rows(x):
    return torch.sort(x, dim=-1, descending=True).values


def _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids):
    assert out.logprob_token_ids.shape == ref.logprob_token_ids.shape
    assert out.logprobs.shape == ref.logprobs.shape
    assert out.selected_token_ranks.shape == ref.selected_token_ranks.shape
    assert out.logprob_token_ids.dtype == ref.logprob_token_ids.dtype
    assert out.logprobs.dtype == torch.float32
    assert out.selected_token_ranks.dtype == torch.int64

    assert torch.equal(out.logprob_token_ids[:, 0], sampled)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)

    torch.testing.assert_close(
        out.logprobs[:, 0], ref.logprobs[:, 0], atol=2e-3, rtol=1e-5
    )
    if num_logprobs == 0:
        return

    out_ids = out.logprob_token_ids[:, 1:]
    ref_ids = ref.logprob_token_ids[:, 1:]
    out_vals = _sorted_rows(_row_values(logits, out_ids))
    ref_vals = _sorted_rows(_row_values(logits, ref_ids))
    assert torch.equal(out_vals, ref_vals), "top-k values differ"

    torch.testing.assert_close(
        _sorted_rows(out.logprobs[:, 1:]),
        _sorted_rows(ref.logprobs[:, 1:]),
        atol=2e-3,
        rtol=1e-5,
    )
    if strict_ids:
        out_order = torch.argsort(_row_values(logits, out_ids), dim=-1, descending=True)
        ref_order = torch.argsort(_row_values(logits, ref_ids), dim=-1, descending=True)
        assert torch.equal(
            torch.gather(out_ids, 1, out_order), torch.gather(ref_ids, 1, ref_order)
        ), "top-k ids differ"


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_logprobs", KS)
@pytest.mark.parametrize("shape", SHAPES)
def test_compute_topk_scores_matches_vllm(shape, num_logprobs, dtype):
    batch, vocab = shape
    logits, sampled = _make_inputs(batch, vocab, dtype, seed=batch + vocab)
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(
        ref, out, logits, sampled, num_logprobs, strict_ids=(dtype == torch.float32)
    )


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_logprobs", [1, 5, 20])
def test_compute_topk_scores_logits_mode(dtype, num_logprobs):
    batch, vocab = 8, 65536
    logits, sampled = _make_inputs(batch, vocab, dtype, seed=7)
    ref = _vllm_fn(logits, num_logprobs, sampled, logits_mode=True)
    out = flaggems_vllm.compute_topk_scores(
        logits, num_logprobs, sampled, logits_mode=True
    )
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)
    assert torch.equal(out.logprob_token_ids[:, 0], sampled)
    torch.testing.assert_close(out.logprobs[:, 0], ref.logprobs[:, 0])
    out_vals = _sorted_rows(_row_values(logits, out.logprob_token_ids[:, 1:]))
    ref_vals = _sorted_rows(_row_values(logits, ref.logprob_token_ids[:, 1:]))
    assert torch.equal(out_vals, ref_vals)
    torch.testing.assert_close(
        _sorted_rows(out.logprobs[:, 1:]), _sorted_rows(ref.logprobs[:, 1:])
    )


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs", [1, 5, 50])
def test_compute_topk_scores_non_contiguous_rows(num_logprobs):
    batch, vocab, pad = 4, 32768, 128
    base, sampled = _make_inputs(batch, vocab + pad, torch.float32, seed=11)
    logits = base[:, :vocab]
    assert logits.stride(0) == vocab + pad
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids=True)


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs", [1, 10, 50])
def test_compute_topk_scores_ties(num_logprobs):
    batch, vocab = 8, 32768
    generator = torch.Generator(device=device)
    generator.manual_seed(3)
    logits = torch.randint(
        0, 8, (batch, vocab), device=device, generator=generator
    ).float()
    sampled = torch.randint(
        vocab, (batch,), dtype=torch.int64, device=device, generator=generator
    )
    ref = _vllm_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled)
    _assert_matches(ref, out, logits, sampled, num_logprobs, strict_ids=False)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", DTYPES)
def test_compute_topk_scores_masked_logits_keep_valid_ids(dtype):
    vocab, topk = 1024, 16
    logits = torch.full((2, vocab), float("-inf"), dtype=dtype, device=device)
    logits[0, 7] = 3
    logits[0, 19] = 1
    logits[1, 23] = 2
    sampled = torch.tensor([7, 5], dtype=torch.int64, device=device)

    out = flaggems_vllm.compute_topk_scores(logits, topk, sampled)
    ids = out.logprob_token_ids[:, 1:]
    assert torch.all((ids >= 0) & (ids < vocab))
    assert torch.all(
        torch.sort(ids, dim=1).values[:, 1:] != torch.sort(ids, dim=1).values[:, :-1]
    )
    assert torch.equal(ids[:, 0], torch.tensor([7, 23], device=device))
    assert torch.isneginf(out.logprobs[0, 3:]).all()
    assert torch.isneginf(out.logprobs[1, 2:]).all()
    assert torch.equal(
        out.selected_token_ranks, torch.tensor([1, vocab], device=device)
    )


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_masked_logits_custom_token_path():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

    vocab, topk = 1024, 16
    logits = torch.full((1, vocab), float("-inf"), device=device)
    logits[0, 11] = 1
    sampled = torch.tensor([11], dtype=torch.int64, device=device)
    state = LogprobTokenIdsState(max_num_reqs=1, device=torch.device("cuda"))
    state.add_request(0, SamplingParams(logprob_token_ids=[]))
    state.apply_staged_writes()
    out = flaggems_vllm.compute_topk_scores(
        logits,
        topk,
        sampled,
        logprob_token_ids_state=state,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
        max_per_req_token_ids=topk,
    )
    ids = out.logprob_token_ids[0, 1:]
    assert torch.all((ids >= 0) & (ids < vocab))
    assert torch.unique(ids).numel() == topk
    assert ids[0] == 11
    assert torch.isneginf(out.logprobs[0, 2:]).all()


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("spread", ["upper16", "upper24"])
def test_compute_topk_scores_refines_oversized_fp32_group(spread):
    vocab, topk = 32768, 50
    token = torch.arange(vocab, dtype=torch.float32, device=device)
    if spread == "upper24":
        token = torch.floor(token / 128)
    logits = (1.0 + token / (2**23)).reshape(1, vocab)
    sampled = torch.tensor([0], dtype=torch.int64, device=device)

    ref = _vllm_fn(logits, topk, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, topk, sampled)
    _assert_matches(ref, out, logits, sampled, topk, strict_ids=spread == "upper16")


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_refines_oversized_fp16_group():
    vocab, topk = 32768, 50
    token = torch.arange(vocab, dtype=torch.float32, device=device)
    logits = (1.0 + torch.floor(token / 4096) / 1024).to(torch.float16).reshape(1, -1)
    sampled = torch.tensor([0], dtype=torch.int64, device=device)
    ref = _vllm_fn(logits, topk, sampled)
    out = flaggems_vllm.compute_topk_scores(logits, topk, sampled)
    _assert_matches(ref, out, logits, sampled, topk, strict_ids=False)


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [0, 5])
def test_compute_topk_scores_large_common_offset(dtype, topk):
    vocab = 1024
    logits = torch.full((1, vocab), 1e9, dtype=dtype, device=device)
    sampled = torch.tensor([0], dtype=torch.int64, device=device)
    out = flaggems_vllm.compute_topk_scores(logits, topk, sampled)
    torch.testing.assert_close(
        out.logprobs, torch.full_like(out.logprobs, -math.log(vocab)), atol=1e-4, rtol=0
    )


@pytest.mark.compute_topk_scores
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="need two CUDA devices")
def test_compute_topk_scores_uses_logits_device():
    logits = torch.arange(2048, dtype=torch.float32, device="cuda:1").reshape(2, 1024)
    sampled = torch.tensor([100, 900], dtype=torch.int64, device="cuda:1")
    with torch.cuda.device(0):
        out = flaggems_vllm.compute_topk_scores(logits, 5, sampled)
        assert torch.cuda.current_device() == 0
    assert out.logprob_token_ids.device == logits.device
    assert out.logprobs.device == logits.device
    assert torch.equal(
        out.logprob_token_ids[:, 1:],
        torch.tensor([[1023, 1022, 1021, 1020, 1019]] * 2, device="cuda:1"),
    )


@pytest.mark.compute_topk_scores
@pytest.mark.parametrize("num_logprobs,max_custom", [(5, 3), (4, 4), (0, 1)])
def test_compute_topk_scores_custom_token_ids(num_logprobs, max_custom):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

    batch, vocab = 4, 4096
    logits, sampled = _make_inputs(batch, vocab, torch.float32, seed=17)
    per_req = (
        [[1, 2, 3], [], [7], [10, 20]] if max_custom >= 3 else [[1], [], [7], [10]]
    )
    state = LogprobTokenIdsState(max_num_reqs=batch, device=torch.device("cuda"))
    for i, ids in enumerate(per_req):
        state.add_request(i, SamplingParams(logprob_token_ids=ids))
    state.apply_staged_writes()
    expanded_idx_mapping = torch.arange(batch, dtype=torch.int32, device=device)

    kwargs = dict(
        logprob_token_ids_state=state,
        expanded_idx_mapping=expanded_idx_mapping,
        max_per_req_token_ids=max_custom,
    )
    ref = _vllm_fn(logits, num_logprobs, sampled, **kwargs)
    out = flaggems_vllm.compute_topk_scores(logits, num_logprobs, sampled, **kwargs)
    assert torch.equal(out.logprob_token_ids, ref.logprob_token_ids)
    assert torch.equal(out.selected_token_ranks, ref.selected_token_ranks)
    torch.testing.assert_close(out.logprobs, ref.logprobs, atol=2e-3, rtol=1e-5)


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_rejects_bad_inputs():
    logits, sampled = _make_inputs(2, 1024, torch.float32)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_scores(logits, 5, sampled.to(torch.int16))
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_scores(
            torch.empty((2, 0), device=device), 0, sampled
        )
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_scores(logits, 257, sampled)
    sampled_view = torch.tensor([0, 99, 1, 99], dtype=torch.int64, device=device)[::2]
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.compute_topk_scores(logits, 5, sampled_view)


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_empty_batch():
    logits = torch.empty((0, 1024), dtype=torch.float32, device=device)
    sampled = torch.empty((0,), dtype=torch.int64, device=device)
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled)
    assert out.logprob_token_ids.shape == (0, 6)
    assert out.logprobs.shape == (0, 6)
    assert out.selected_token_ranks.shape == (0,)


@pytest.mark.compute_topk_scores
def test_compute_topk_scores_cu_num_logits_passthrough():
    logits, sampled = _make_inputs(4, 4096, torch.float32)
    cu_list = [0, 1, 2, 3, 4]
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled, cu_num_logits=cu_list)
    assert out.cu_num_generated_tokens == cu_list
    assert out.cu_num_generated_tokens_tensor is None

    cu_tensor = torch.tensor(cu_list, dtype=torch.int32, device=device)
    out = flaggems_vllm.compute_topk_scores(logits, 5, sampled, cu_num_logits=cu_tensor)
    assert out.cu_num_generated_tokens is None
    assert out.cu_num_generated_tokens_tensor is cu_tensor
