import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import LlamaConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.weight_cache.daemon import WeightCacheDaemon, _model_config_for_loading
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODEL_PATH = Path(__file__).parent / "fixtures" / "qwen4_exp_nvfp4"
_PLE_RUNTIME_FIELDS = (
    "ple_disk_dir",
    "ple_disk_hot_cache_gb",
    "ple_disk_hot_frequency_file",
    "ple_disk_dynamic_cache_gb",
    "ple_disk_prefill_buffer_tokens",
    "ple_disk_prefill_read_pages",
    "ple_disk_max_read_pages",
    "ple_disk_stats_log_interval",
    "ple_disk_cleanup_generations",
)


@pytest.mark.parametrize(
    ("requested_storage", "expected_storage"),
    (("disk", "disk"), (None, "pinned")),
)
def test_qwen4_scheduler_config_carries_resolved_ple_settings(
    tmp_path, requested_storage, expected_storage
):
    kwargs = {
        "model_path": str(_MODEL_PATH),
        "ple_storage": requested_storage,
        "tp_size": 2,
        "device": "cuda",
        "chunked_prefill_size": 3072,
        "ple_disk_hot_cache_gb": 1.25,
        "ple_disk_dynamic_cache_gb": 0.75,
        "ple_disk_prefill_buffer_tokens": 4096,
        "ple_disk_prefill_read_pages": 64,
        "ple_disk_max_read_pages": 96,
        "ple_disk_stats_log_interval": 17,
    }
    if requested_storage == "disk":
        kwargs["ple_disk_dir"] = str(tmp_path / "images")

    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(**kwargs)
        scheduler_config = ModelConfig.from_server_args(server_args)

    launcher_text_config = server_args.get_model_config().hf_text_config
    for text_config in (launcher_text_config, scheduler_config.hf_text_config):
        assert text_config.ple_storage == expected_storage
        for name in _PLE_RUNTIME_FIELDS:
            assert getattr(text_config, name) == getattr(server_args, name)
        assert (
            text_config.ple_disk_max_prefill_chunk_tokens
            == server_args.chunked_prefill_size
        )


def test_qwen4_text_checkpoint_carries_runtime_settings(tmp_path):
    document = json.loads((_MODEL_PATH / "config.json").read_text())
    text_document = document["text_config"]
    text_document["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    model_path = tmp_path / "text-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(text_document))
    image_dir = tmp_path / "images"

    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(model_path),
            ple_storage="disk",
            ple_disk_dir=str(image_dir),
            device="cuda",
        )
        scheduler_config = ModelConfig.from_server_args(server_args)

    assert scheduler_config.hf_text_config.ple_storage == "disk"
    assert scheduler_config.hf_text_config.ple_disk_dir == str(image_dir)


def test_prefill_chunk_bound_uses_final_mis_rewrite(tmp_path):
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path / "images"),
            device="cuda",
            attention_backend="flashinfer",
            enable_mis=True,
        )
        child_config = ModelConfig.from_server_args(server_args)

    assert server_args.chunked_prefill_size == -1
    assert (
        server_args.get_model_config().hf_text_config.ple_disk_max_prefill_chunk_tokens
        == server_args.max_prefill_tokens
    )
    assert (
        child_config.hf_text_config.ple_disk_max_prefill_chunk_tokens
        == server_args.max_prefill_tokens
    )


def test_qwen4_draft_config_is_stamped(tmp_path):
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path / "images"),
            device="cuda",
        )
        draft_config = ModelConfig.from_server_args(
            server_args, model_path=str(_MODEL_PATH), is_draft_model=True
        )

    assert draft_config.hf_text_config.ple_storage == "disk"
    assert draft_config.hf_text_config.ple_disk_dir == str(tmp_path / "images")


def test_non_qwen_draft_config_ignores_target_ple_storage(tmp_path):
    draft_path = tmp_path / "eagle3-draft"
    draft_config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
    )
    draft_config.architectures = ["LlamaForCausalLMEagle3"]
    draft_config.save_pretrained(draft_path)

    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            ple_storage="pinned",
            device="cuda",
        )
        resolved_draft = ModelConfig.from_server_args(
            server_args,
            model_path=str(draft_path),
            is_draft_model=True,
        )

    assert resolved_draft.hf_config.architectures == ["LlamaForCausalLMEagle3"]


def test_runtime_config_hook_runs_once_during_server_args_launch(monkeypatch):
    calls = []
    original = Qwen4ExpConfig.apply_sglang_runtime_config

    def counted_hook(config, server_args):
        calls.append(config)
        return original(config, server_args)

    monkeypatch.setattr(Qwen4ExpConfig, "apply_sglang_runtime_config", counted_hook)
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            ple_storage="pinned",
            device="cuda",
        )

    assert calls == [server_args.get_model_config().hf_config]


def test_weight_cache_daemon_warns_once_for_non_gpu_engine_storage(caplog):
    daemon = WeightCacheDaemon(
        model_path=str(_MODEL_PATH),
        gpu_id=0,
        ple_storage="disk",
    )

    with caplog.at_level("WARNING"):
        selected = [daemon._ple_storage_for_cuda_ipc() for _ in range(2)]

    assert selected == ["gpu", "gpu"]
    warnings = [
        record.message
        for record in caplog.records
        if "--ple-storage disk" in record.message
    ]
    assert len(warnings) == 1


def test_weight_cache_daemon_stamps_bf16_qwen4_loading_on_gpu_storage(tmp_path):
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            dtype="bfloat16",
            device="cuda",
            ple_storage="gpu",
        )

    assert server_args.ple_storage == "gpu"
    model_config = _model_config_for_loading(server_args)
    assert model_config.hf_text_config.ple_storage == "gpu"


def test_weight_cache_daemon_command_receives_resolved_storage(monkeypatch):
    from sglang.srt.entrypoints import engine as engine_module
    from sglang.srt.weight_cache import protocol

    commands = []

    class Process:
        pid = 17
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        engine_module.subprocess,
        "Popen",
        lambda command: commands.append(command) or Process(),
    )
    monkeypatch.setattr(engine_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(protocol, "cleanup_stale_daemon_files", lambda rank: None)
    args = SimpleNamespace(
        dp_size=1,
        nnodes=1,
        dist_init_addr="127.0.0.1:23456",
        tp_size=1,
        pp_size=1,
        node_rank=0,
        ep_size=1,
        base_gpu_id=0,
        gpu_id_step=1,
        model_path=str(_MODEL_PATH),
        load_format="auto",
        dtype="bfloat16",
        quantization=None,
        model_loader_extra_config="{}",
        trust_remote_code=False,
        revision=None,
        weight_cache_timeout=1,
        ple_storage="disk",
    )

    processes = engine_module.Engine._launch_weight_cache_daemons(args)

    assert len(processes) == 1
    storage_index = commands[0].index("--ple-storage")
    assert commands[0][storage_index + 1] == "disk"


def test_launcher_skips_runtime_config_stamping_for_instance_connector():
    model_config = ModelConfig(str(_MODEL_PATH))
    hook = MagicMock(side_effect=AssertionError("launcher stamped instance config"))
    model_config.hf_config.apply_sglang_runtime_config = hook
    with patch.object(ServerArgs, "get_model_config", return_value=model_config), patch(
        "sglang.srt.arg_groups.overrides.is_cuda", return_value=True
    ), patch("sglang.srt.server_args.is_cuda", return_value=True):
        args = ServerArgs(
            model_path="instance://127.0.0.1:8000/qwen",
            device="cuda",
            ple_storage="pinned",
        )

    assert args.ple_storage == "pinned"
    hook.assert_not_called()


def test_loaded_ple_embedding_must_match_requested_storage():
    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(ple_storage="disk"))
    ngram = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(ngram)
    pinned = Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)
    torch.nn.Module.__init__(pinned)
    ngram.ngram_embedding = pinned
    model.ple = ngram

    with pytest.raises(RuntimeError, match="requested disk.*PinnedHostEmbedding"):
        model._assert_ple_storage_matches_config()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
