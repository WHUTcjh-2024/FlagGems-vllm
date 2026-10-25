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

"""Accuracy tests for ``flaggems_vllm.compute_topk_logprobs``.

Reference: vLLM 0.20.2 ``compute_topk_logprobs`` semantics (``torch.topk`` +
``torch.log_softmax``). The selection is compared as a per-row multiset because
ties at the top-k boundary may legitimately pick different equal-valued ids;
logprobs are compared at the ids the kernel actually returned, and the sampled
token rank is compared exactly.
"""

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

device = flaggems_vllm.device

pytestmark = [
    pytest.mark.compute_topk_logprobs,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

if cfg.QUICK_MODE:
    SHAPES = [(1, 1024, 0), (4, 32768, 10), (8, 131072, 50)]
else:
    SHAPES = [
        (0, 128, 5),
        (1, 999, 3),
        (2, 4096, 0),
        (3, 151936, 1),
        (8, 32768, 10),
        (64, 65536, 20),
        (8, 131072, 50),
    ]


def _reference(logits, num_logprobs, sampled):
    token_ids = sampled.unsqueeze(-1).to(torch.int64)
    if num_logprobs > 0:
        topk_ids = torch.topk(logits, num_logprobs, dim=-1).indices
        token_ids = torch.cat((token_ids, topk_ids), dim=1)
    logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1).gather(1, token_ids)
    ranks = (logits >= logits.gather(1, sampled.to(torch.int64)[:, None])).sum(1)
    return token_ids, logprobs, ranks


def _make_logits(batch, vocab, dtype, seed=0, stride1=1):
    torch.manual_seed(seed)
    if stride1 == 1:
        return torch.randn((batch, vocab), device=device, dtype=dtype)
    base = torch.randn((batch, vocab * stride1), device=device, dtype=dtype)
    return base[:, ::stride1]


def _selected_values(logits, ids):
    """The selected logits, sorted, so equal-valued ids compare as a multiset."""
    return torch.sort(logits.to(torch.float32).gather(1, ids), dim=1).values


def _check(out, logits, num_logprobs, sampled):
    expected_ids, _, ranks = _reference(logits, num_logprobs, sampled)

    assert out.logprob_token_ids.shape == expected_ids.shape
    assert out.logprob_token_ids.dtype == torch.int64
    assert out.logprobs.shape == expected_ids.shape
    assert out.logprobs.dtype == torch.float32
    assert out.selected_token_ranks.shape == sampled.shape
    assert out.selected_token_ranks.dtype == torch.int64

    assert torch.equal(
        out.logprob_token_ids[:, 0], sampled.to(torch.int64)
    ), "column 0 must be the sampled token"
    # torch.topk may order equal-valued ids differently, so compare the
    # selected values; the column order itself is checked right below.
    assert torch.equal(
        _selected_values(logits, out.logprob_token_ids),
        _selected_values(logits, expected_ids),
    ), "top-k selection differs from the reference"
    if num_logprobs > 1:
        assert (
            out.logprobs[:, 1:].diff(dim=1) <= 0
        ).all(), "top-k columns must be ordered by descending logprob"
    assert torch.equal(out.selected_token_ranks, ranks)

    if logits.shape[0] == 0:
        return
    expected_logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1).gather(
        1, out.logprob_token_ids
    )
    torch.testing.assert_close(out.logprobs, expected_logprobs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_compute_topk_logprobs(dtype, shape):
    batch, vocab, num_logprobs = shape
    logits = _make_logits(batch, vocab, dtype, seed=batch + vocab + num_logprobs)
    if batch:
        sampled = torch.randint(vocab, (batch,), device=device, dtype=torch.int32)
    else:
        sampled = torch.empty((0,), dtype=torch.int32, device=device)

    out = flaggems_vllm.compute_topk_logprobs(logits, num_logprobs, sampled)
    _check(out, logits, num_logprobs, sampled)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compute_topk_logprobs_strided_logits(dtype):
    batch, vocab, num_logprobs = 4, 4096, 10
    # stride(1) == 2 and a padded row stride exercise both pointer strides.
    logits = _make_logits(batch, vocab, dtype, seed=7, stride1=2)
    sampled = torch.randint(vocab, (batch,), device=device, dtype=torch.int32)
    out = flaggems_vllm.compute_topk_logprobs(logits, num_logprobs, sampled)
    _check(out, logits, num_logprobs, sampled)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_compute_topk_logprobs_ties(dtype):
    batch, vocab, num_logprobs = 8, 8192, 20
    torch.manual_seed(11)
    logits = (torch.randn((batch, vocab), device=device, dtype=dtype) * 4).round() / 4
    sampled = torch.randint(vocab, (batch,), device=device, dtype=torch.int32)
    out = flaggems_vllm.compute_topk_logprobs(logits, num_logprobs, sampled)
    _check(out, logits, num_logprobs, sampled)


def test_compute_topk_logprobs_sampled_neg_inf_rank():
    # 4099 is not a multiple of any tuned BLOCK_SIZE, so the masked tail of
    # _ranks_kernel always runs.
    batch, vocab, num_logprobs = 3, 4099, 5
    logits = _make_logits(batch, vocab, torch.float32, seed=9)
    sampled = torch.randint(vocab, (batch,), device=device, dtype=torch.int32)
    logits[0, sampled[0]] = float("-inf")

    out = flaggems_vllm.compute_topk_logprobs(logits, num_logprobs, sampled)
    expected = torch.log_softmax(logits.to(torch.float32), dim=-1)
    expected_ranks = (logits >= logits.gather(1, sampled.to(torch.int64)[:, None])).sum(
        1
    )
    assert torch.equal(out.selected_token_ranks, expected_ranks)
    torch.testing.assert_close(
        out.logprobs,
        expected.gather(1, out.logprob_token_ids),
        atol=1e-4,
        rtol=1e-4,
    )


def test_compute_topk_logprobs_matches_vllm():
    logprob = pytest.importorskip("vllm.v1.worker.gpu.sample.logprob")
    reference_fn = getattr(logprob, "compute_topk_logprobs", None)
    if reference_fn is None:
        pytest.skip("vllm build without compute_topk_logprobs (0.29+ renamed it)")

    batch, vocab, num_logprobs = 8, 32768, 10
    torch.manual_seed(5)
    logits = torch.randn((batch, vocab), device=device, dtype=torch.bfloat16)
    sampled = torch.randint(vocab, (batch,), device=device, dtype=torch.int32)

    expected = reference_fn(logits, num_logprobs, sampled)
    out = flaggems_vllm.compute_topk_logprobs(logits, num_logprobs, sampled)
    if not hasattr(expected, "logprob_token_ids"):
        pytest.skip("installed vLLM returns a different logprobs container")

    assert torch.equal(out.logprob_token_ids[:, 0], expected.logprob_token_ids[:, 0])
    assert torch.equal(
        _selected_values(logits, out.logprob_token_ids),
        _selected_values(logits, expected.logprob_token_ids),
    )
    # Compare vLLM's logprobs at the ids this kernel returned: ties let the two
    # implementations order equal-valued ids differently.
    expected_at_ours = torch.log_softmax(logits.to(torch.float32), dim=-1).gather(
        1, out.logprob_token_ids
    )
    torch.testing.assert_close(out.logprobs, expected_at_ours, atol=1e-4, rtol=1e-4)
    assert torch.equal(out.selected_token_ranks, expected.selected_token_ranks)


def test_compute_topk_logprobs_metadata_passthrough():
    logits = _make_logits(2, 1024, torch.float32, seed=3)
    sampled = torch.randint(1024, (2,), device=device, dtype=torch.int64)
    cu_num_logits = [1, 2]
    out = flaggems_vllm.compute_topk_logprobs(
        logits, 3, sampled, cu_num_logits=cu_num_logits
    )
    assert out.cu_num_generated_tokens == cu_num_logits
    _check(out, logits, 3, sampled)


def test_compute_topk_logprobs_unsupported():
    logits = _make_logits(2, 1024, torch.float32, seed=3)
    sampled = torch.randint(1024, (2,), device=device, dtype=torch.int32)

    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_logprobs(logits.double(), 3, sampled)
    with pytest.raises(ValueError):
        flaggems_vllm.compute_topk_logprobs(logits, 1025, sampled)
    with pytest.raises(ValueError):
        flaggems_vllm.compute_topk_logprobs(logits, -1, sampled)
    with pytest.raises(ValueError):
        flaggems_vllm.compute_topk_logprobs(logits[0], 3, sampled)
    with pytest.raises(ValueError):
        flaggems_vllm.compute_topk_logprobs(logits, 3, sampled.to(torch.float32))
    with pytest.raises(ValueError):
        flaggems_vllm.compute_topk_logprobs(logits, 3, sampled[:1])
    with pytest.raises(NotImplementedError):
        flaggems_vllm.compute_topk_logprobs(logits, 257, sampled)
