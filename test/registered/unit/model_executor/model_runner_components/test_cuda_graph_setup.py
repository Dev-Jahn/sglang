import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_decode_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, CustomGraphRunner)
    assert capture.runner.model_runner is model_runner


def test_qwen4_exp_sm120_jit_prewarm_uses_model_specializations(monkeypatch):
    calls = []

    from sglang.kernels.ops.attention import qsa_indexer
    from sglang.kernels.ops.elementwise import fast_topk, hc_combine
    from sglang.kernels.ops.gemm import fp8_blockwise_gemm
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm

    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(
        fast_topk, "_jit_fast_topk_module", lambda *args: calls.append(("topk", args))
    )
    monkeypatch.setattr(
        qsa_indexer,
        "_jit_qsa_indexer_module",
        lambda *args: calls.append(("qsa", args)),
    )
    monkeypatch.setattr(
        hc_combine,
        "_jit_hc_combine_module",
        lambda *args: calls.append(("hc", args)),
    )
    monkeypatch.setattr(
        grouped_gemma_rmsnorm,
        "_jit_grouped_gemma_rmsnorm_module",
        lambda *args: calls.append(("rms", args)),
    )
    monkeypatch.setattr(
        fp8_blockwise_gemm,
        "_jit_fp8_blockwise_module",
        lambda: calls.append(("fp8", ())),
    )
    runner = SimpleNamespace(
        device="cuda",
        dtype=torch.bfloat16,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen4ExpForConditionalGeneration"]
            ),
            hf_text_config=SimpleNamespace(
                indexer_budget=2048,
                indexer_compress_ratio=4,
                indexer_head_dim=128,
                hc_count=4,
                hidden_size=2560,
            ),
            quantization="fp8",
        ),
    )

    cuda_graph_setup._prewarm_qwen4_exp_sm120_jit_kernels(runner)

    assert calls == [
        ("topk", (512,)),
        ("qsa", (torch.bfloat16, 128, True)),
        ("hc", (4, 2560, torch.bfloat16)),
        ("rms", (2560, torch.bfloat16)),
        ("fp8", ()),
    ]


@pytest.mark.parametrize(
    ("is_sm120", "architecture"),
    [
        (False, "Qwen4ExpForConditionalGeneration"),
        (True, "LlamaForCausalLM"),
    ],
)
def test_qwen4_exp_jit_prewarm_is_sm120_and_architecture_gated(
    monkeypatch, is_sm120, architecture
):
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: is_sm120)
    runner = SimpleNamespace(
        device="cuda",
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        ),
    )
    cuda_graph_setup._prewarm_qwen4_exp_sm120_jit_kernels(runner)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
