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

"""Performance benchmark for ``flaggems_vllm.compute_topk_logprobs``.

Baseline is vLLM 0.20.2's own ``compute_topk_logprobs`` when it is importable,
otherwise an equivalent torch reference (``torch.topk`` + ``log_softmax``).
"""

import pytest
import torch

import flaggems_vllm

from . import base, consts

try:
    from vllm.v1.worker.gpu.sample.logprob import (
        compute_topk_logprobs as _vllm_compute_topk_logprobs,
    )
except Exception:  # noqa: BLE001 - vLLM absent/broken, or 0.29+ renamed it
    _vllm_compute_topk_logprobs = None


def reference(logits, num_logprobs, sampled):
    token_ids = sampled.unsqueeze(-1).to(torch.int64)
    if num_logprobs > 0:
        token_ids = torch.cat(
            (token_ids, torch.topk(logits, num_logprobs, dim=-1).indices), dim=1
        )
    logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1).gather(1, token_ids)
    ranks = (logits >= logits.gather(1, sampled.to(torch.int64)[:, None])).sum(1)
    return token_ids, logprobs, ranks


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 32768, 10),
            (1, 131072, 10),
            (8, 131072, 50),
            (64, 65536, 10),
            (64, 131072, 10),
            (256, 32768, 0),
            (256, 151936, 20),
        ]
        self.shape_desc = "batch, vocab, num_logprobs"

    def get_input_iter(self, dtype):
        for batch, vocab, num_logprobs in self.shapes:
            yield (
                torch.randn((batch, vocab), device=self.device, dtype=dtype),
                num_logprobs,
                torch.randint(vocab, (batch,), device=self.device, dtype=torch.int32),
            )


@pytest.mark.compute_topk_logprobs
def test_perf_compute_topk_logprobs():
    torch_op = _vllm_compute_topk_logprobs or reference
    bench = OperatorBenchmark(
        op_name="compute_topk_logprobs",
        torch_op=torch_op,
        gems_op=flaggems_vllm.compute_topk_logprobs,
        dtypes=[torch.bfloat16, torch.float16, torch.float32],
    )
    bench.run()
