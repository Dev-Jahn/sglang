import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.weight_cache.daemon import _model_config_for_loading
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


def test_weight_cache_daemon_rejects_non_gpu_ple_storage(tmp_path):
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        server_args = ServerArgs(
            model_path=str(_MODEL_PATH),
            ple_storage="pinned",
            device="cuda",
        )

    with pytest.raises(ValueError, match="weight-cache daemon.*--ple-storage gpu"):
        _model_config_for_loading(server_args)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
