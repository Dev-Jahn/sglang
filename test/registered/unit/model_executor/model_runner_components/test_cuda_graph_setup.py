import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_decode_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_cuda_graph_prewarm_skips_non_qwen_model_off_sm120(monkeypatch):
    class TransformerOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2)

    runner = SimpleNamespace(
        device="cuda",
        model=TransformerOnlyModel(),
        server_args=SimpleNamespace(
            ple_storage="gpu",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="full"),
            ),
        ),
    )
    resolve = MagicMock()
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", resolve)
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: False)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    resolve.assert_not_called()


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


def test_cuda_graph_prewarm_delegates_to_the_language_model(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    language_model = SimpleNamespace(prewarm_cuda_graphs=prewarm)
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        server_args=SimpleNamespace(
            ple_storage="pinned",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            ),
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "resolve_language_model", lambda _: language_model
    )
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: True)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_called_once_with(runner, capture_decode_cuda_graph=True)


def test_sm120_qwen_prewarm_runs_with_gpu_ple_storage(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    language_model = SimpleNamespace(prewarm_cuda_graphs=prewarm)
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        server_args=SimpleNamespace(
            ple_storage="gpu",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="full"),
            ),
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "resolve_language_model", lambda _: language_model
    )
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(cuda_graph_setup, "is_sm121", lambda: False)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_called_once_with(runner, capture_decode_cuda_graph=True)


@pytest.mark.parametrize("offloaded", [False, True])
def test_qwen_prewarm_allocates_staging_only_for_offloaded_ple(monkeypatch, offloaded):
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.models import qwen4_exp
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpModel,
        Qwen4ExpPinnedHostEmbedding,
        Qwen4ExpPLELayer,
    )

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    if offloaded:
        embedding = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
        torch.nn.Module.__init__(embedding)
    else:
        embedding = torch.nn.Embedding(1, 1)
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=embedding, gather_dp_tokens=False
    )
    layer.reset_cuda_graph_capture_buffers = MagicMock()
    layer.prepare_cuda_graph_prefetch_buffer = MagicMock()
    model.layer = layer
    runner = SimpleNamespace(
        device="cpu",
        max_decode_logits_rows=lambda: 7,
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED, bs=None, max_bs=0),
                decode=SimpleNamespace(backend=Backend.FULL),
            )
        ),
    )
    monkeypatch.setattr(qwen4_exp, "is_sm120_supported", lambda: False)

    model.prewarm_cuda_graphs(runner, capture_decode_cuda_graph=True)

    if offloaded:
        layer.reset_cuda_graph_capture_buffers.assert_called_once_with()
        layer.prepare_cuda_graph_prefetch_buffer.assert_called_once_with(
            7, torch.device("cpu")
        )
    else:
        layer.reset_cuda_graph_capture_buffers.assert_not_called()
        layer.prepare_cuda_graph_prefetch_buffer.assert_not_called()


def test_pinned_prewarm_uses_prefill_token_bucket_extent(monkeypatch):
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.models import qwen4_exp
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpModel,
        Qwen4ExpPinnedHostEmbedding,
        Qwen4ExpPLELayer,
    )

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    embedding = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
    torch.nn.Module.__init__(embedding)
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=embedding, gather_dp_tokens=False
    )
    layer.reset_cuda_graph_capture_buffers = MagicMock()
    layer.prepare_cuda_graph_prefetch_buffer = MagicMock()
    model.layer = layer
    runner = SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.FULL, bs=(13,), max_bs=0),
                decode=SimpleNamespace(backend=Backend.DISABLED),
            )
        ),
    )
    monkeypatch.setattr(qwen4_exp, "is_sm120_supported", lambda: False)

    model.prewarm_cuda_graphs(runner, capture_decode_cuda_graph=False)

    layer.prepare_cuda_graph_prefetch_buffer.assert_called_once_with(
        13, torch.device("cpu")
    )


def test_pinned_prewarm_rejects_zero_capture_extent(monkeypatch):
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.models import qwen4_exp
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpModel,
        Qwen4ExpPinnedHostEmbedding,
        Qwen4ExpPLELayer,
    )

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    embedding = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
    torch.nn.Module.__init__(embedding)
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=embedding, gather_dp_tokens=False
    )
    layer.reset_cuda_graph_capture_buffers = MagicMock()
    model.layer = layer
    runner = SimpleNamespace(
        device="cpu",
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.FULL, bs=(), max_bs=0),
                decode=SimpleNamespace(backend=Backend.DISABLED),
            )
        ),
    )
    monkeypatch.setattr(qwen4_exp, "is_sm120_supported", lambda: False)

    with pytest.raises(
        RuntimeError,
        match=r"--ple-storage pinned.*zero-token.*--cuda-graph-backend-prefill",
    ):
        model.prewarm_cuda_graphs(runner, capture_decode_cuda_graph=False)


@pytest.mark.parametrize("ple_storage", ["pinned", "disk"])
def test_cuda_graph_prewarm_is_required_for_ple_offload(monkeypatch, ple_storage):
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_storage=ple_storage)
        ),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            )
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup,
        "resolve_language_model",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: True)

    with pytest.raises(RuntimeError, match="PLE offload.*prewarm_cuda_graphs"):
        cuda_graph_setup._prewarm_model_cuda_graphs(
            runner, capture_decode_cuda_graph=True
        )


def test_disk_cuda_graph_requires_replay_hook_on_resolved_model():
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_storage="disk")
        ),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="full"),
            )
        ),
        _decode_cuda_graph_runner_cls=lambda: type(
            "HookedRunner", (), {"routes_model_replay_hook": True}
        ),
    )
    with pytest.raises(RuntimeError, match="supports_cuda_graph_replay_hook"):
        cuda_graph_setup._validate_ple_disk_cuda_graph_replay(
            runner,
            SimpleNamespace(prewarm_cuda_graphs=lambda *args, **kwargs: None),
        )


def test_disk_cuda_graph_rejects_unlisted_replay_runner():
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_storage="disk")
        ),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="full"),
            )
        ),
        _decode_cuda_graph_runner_cls=lambda: type("UnlistedRunner", (), {}),
    )
    with pytest.raises(RuntimeError, match="UnlistedRunner"):
        cuda_graph_setup._validate_ple_disk_cuda_graph_replay(
            runner,
            SimpleNamespace(
                prewarm_cuda_graphs=lambda *args, **kwargs: None,
                supports_cuda_graph_replay_hook=True,
            ),
        )


def test_draft_runner_constructors_explicitly_disable_model_replay_hooks():
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )
    from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
        FrozenKVMTPCudaGraphRunner,
    )
    from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
        MultiLayerEagleDraftExtendCudaGraphRunner,
    )

    for runner_cls in (
        EAGLEDraftCudaGraphRunner,
        EAGLEDraftExtendCudaGraphRunner,
        FrozenKVMTPCudaGraphRunner,
        MultiLayerEagleDraftExtendCudaGraphRunner,
    ):
        assert runner_cls.__dict__["routes_model_replay_hook"] is False


def test_capture_decode_graph_uses_resolved_hook_model(monkeypatch):
    language_model = SimpleNamespace(supports_cuda_graph_replay_hook=True)

    class HookedRunner:
        routes_model_replay_hook = True

        def __init__(self, model_runner):
            self.model_runner = model_runner

    runner = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_storage="disk")
        ),
        server_args=SimpleNamespace(
            model_impl="auto",
            disaggregation_mode="none",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="full"),
            ),
        ),
        _decode_cuda_graph_runner_cls=lambda: HookedRunner,
        decode_num_tokens_per_req=lambda: 1,
    )
    resolve = MagicMock(return_value=language_model)
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", resolve)
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    capture = cuda_graph_setup.capture_decode_graph(model_runner=runner)

    resolve.assert_called_once_with(runner.model)
    assert isinstance(capture.runner, HookedRunner)


def test_ple_staging_prewarm_reaches_non_sm120_models(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
        server_args=SimpleNamespace(
            ple_storage="pinned",
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            ),
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: False)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_called_once_with(runner, capture_decode_cuda_graph=True)


def test_capture_cuda_graphs_prewarms_before_prefill_capture(monkeypatch):
    runner = SimpleNamespace(
        device="cpu",
        model=object(),
        model_config=SimpleNamespace(quantization=None),
        is_draft_worker=False,
        server_args=SimpleNamespace(
            moe_runner_backend="cutlass",
            moe_a2a_backend="none",
            forward_hooks=None,
            enable_symm_mem=False,
        ),
        forward_stream=None,
        canary_manager=None,
    )
    eager_runner = object()
    calls = MagicMock()
    prewarm = calls.prewarm
    capture_prefill = calls.capture_prefill
    prefill = cuda_graph_setup.GraphCapture(
        runner=eager_runner,
        memory_phase="prefill",
        memory_usage_gb=0,
        capture_time=0,
    )
    monkeypatch.setattr(
        cuda_graph_setup.GraphSharedOutput,
        "create_for_model_runner",
        lambda _: object(),
    )
    monkeypatch.setattr(cuda_graph_setup, "EagerRunner", lambda _: eager_runner)
    monkeypatch.setattr(cuda_graph_setup, "_prewarm_model_cuda_graphs", prewarm)
    capture_prefill.return_value = prefill
    monkeypatch.setattr(cuda_graph_setup, "capture_prefill_graph", capture_prefill)
    monkeypatch.setattr(
        cuda_graph_setup, "prealloc_symmetric_memory_pool", lambda **_: None
    )

    cuda_graph_setup.capture_cuda_graphs(
        model_runner=runner, capture_decode_cuda_graph=False
    )

    assert calls.mock_calls[:2] == [
        call.prewarm(runner, capture_decode_cuda_graph=False),
        call.capture_prefill(model_runner=runner, eager_runner=eager_runner),
    ]


def test_cuda_graph_prewarm_skips_when_both_phases_are_disabled(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="disabled"),
            )
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(cuda_graph_setup, "is_sm120_supported", lambda: True)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
