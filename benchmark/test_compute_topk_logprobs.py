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

The baseline is vLLM 0.20.2's own ``compute_topk_logprobs``. When it is not
importable the benchmark skips instead of silently timing against a different
reference, so a reported SpeedUp always means "vs vLLM".
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
    if _vllm_compute_topk_logprobs is None:
        pytest.skip(
            "vLLM's compute_topk_logprobs is unavailable; refusing to report a "
            "SpeedUp against a different baseline"
        )
    bench = OperatorBenchmark(
        op_name="compute_topk_logprobs",
        torch_op=_vllm_compute_topk_logprobs,
        gems_op=flaggems_vllm.compute_topk_logprobs,
        dtypes=[torch.bfloat16, torch.float16, torch.float32],
    )
    bench.run()
