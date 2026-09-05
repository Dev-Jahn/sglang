"""PLE disk server argument validation."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import LlamaConfig

from sglang.srt import server_args as server_args_module
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils.ple_disk import IORING_MAX_ENTRIES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODEL_PATH = Path(__file__).parents[1] / "configs/fixtures/qwen4_exp_nvfp4"


def _cuda_server_args(**kwargs):
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ):
        return server_args_module.ServerArgs(device="cuda", **kwargs)


def _checkpoint_with_disk_storage(tmp_path):
    document = json.loads((_MODEL_PATH / "config.json").read_text())
    document["text_config"]["ple_storage"] = "disk"
    model_path = tmp_path / "disk-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(document))
    return model_path


def test_offload_compatibility_writes_nothing_after_resolution():
    args = _cuda_server_args(model_path=str(_MODEL_PATH), ple_storage="gpu")
    before = vars(args).copy()
    args._handle_offload_compatibility(resolved=True)
    assert vars(args) == before
    assert args.ple_disk_max_read_pages is None
    assert args.ple_disk_prefill_read_pages > 0


def test_unused_disk_options_warn_only_after_resolution(caplog):
    args = _cuda_server_args(
        model_path=str(_MODEL_PATH),
        ple_storage="gpu",
        ple_disk_hot_cache_gb=1.0,
    )
    caplog.clear()
    with caplog.at_level("WARNING"):
        args._handle_offload_compatibility(resolved=True)
        args._handle_offload_compatibility()
    assert caplog.text.count("are unused with --ple-storage") == 1


def test_explicit_max_read_pages_still_validated():
    with pytest.raises(ValueError):
        _cuda_server_args(model_path="dummy", ple_disk_max_read_pages=0)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("ple_disk_hot_cache_gb", -0.1, "hot-cache-gb"),
        ("ple_disk_dynamic_cache_gb", -0.1, "dynamic-cache-gb"),
        ("ple_disk_prefill_buffer_tokens", -1, "prefill-buffer-tokens"),
        ("ple_disk_prefill_read_pages", 0, "prefill-read-pages"),
        ("ple_disk_max_read_pages", 0, "max-read-pages"),
        ("ple_disk_stats_log_interval", -1, "stats-log-interval"),
    ],
)
def test_disk_argument_bounds_are_validated(option, value, message):
    with pytest.raises(ValueError, match=message):
        _cuda_server_args(model_path="dummy", **{option: value})


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        (
            "ple_disk_prefill_read_pages",
            IORING_MAX_ENTRIES + 1,
            str(IORING_MAX_ENTRIES),
        ),
        (
            "ple_disk_prefill_buffer_tokens",
            server_args_module.PLE_DISK_MAX_PREFILL_BUFFER_TOKENS + 1,
            "pinned bytes",
        ),
    ],
)
def test_prefill_argument_upper_bounds_are_validated(option, value, message):
    with pytest.raises(ValueError, match=message):
        _cuda_server_args(model_path="dummy", **{option: value})


def test_disk_storage_requires_an_image_directory():
    with pytest.raises(ValueError, match="requires --ple-disk-dir"):
        _cuda_server_args(model_path=str(_MODEL_PATH), ple_storage="disk")


def test_disk_storage_accepts_a_creatable_directory(tmp_path):
    target = tmp_path / "new" / "images"
    args = _cuda_server_args(
        model_path=str(_MODEL_PATH), ple_storage="disk", ple_disk_dir=str(target)
    )
    assert args.ple_disk_dir == str(target)


def test_disk_storage_rejects_an_unreadable_hot_file(tmp_path):
    with pytest.raises(ValueError, match="readable file"):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
            ple_disk_hot_frequency_file=str(tmp_path / "missing.bin"),
        )


def test_disk_storage_accepts_a_readable_hot_file_template(tmp_path):
    (tmp_path / "hot-0.bin").touch()
    args = _cuda_server_args(
        model_path=str(_MODEL_PATH),
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        ple_disk_hot_frequency_file=str(tmp_path / "hot-{layer}.bin"),
    )
    assert args.ple_disk_hot_frequency_file.endswith("hot-{layer}.bin")


def test_disk_storage_accepts_literal_braces_in_a_hot_file_path(tmp_path):
    archive = tmp_path / "{archive}"
    archive.mkdir()
    hot_file = archive / "hot.bin"
    hot_file.touch()
    args = _cuda_server_args(
        model_path=str(_MODEL_PATH),
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        ple_disk_hot_frequency_file=str(hot_file),
    )
    assert args.ple_disk_hot_frequency_file == str(hot_file)


def test_max_read_pages_rejects_io_uring_entry_overflow():
    with pytest.raises(ValueError, match="32768"):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir="/tmp/ple",
            ple_disk_max_read_pages=32769,
        )


def test_disk_storage_rejects_json_prefill_cuda_graph(tmp_path):
    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
            cuda_graph_config={"prefill": {"backend": Backend.FULL}},
        )


def test_disk_storage_rejects_prefill_cuda_graph_convenience_flag(tmp_path):
    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
            cuda_graph_backend_prefill=Backend.FULL,
        )


def test_disk_storage_disables_default_prefill_cuda_graph(tmp_path, caplog):
    with caplog.at_level("INFO"):
        args = _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
        )

    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED
    assert args.disable_prefill_cuda_graph
    assert "prefill CUDA graphs" in caplog.text
    assert "prefill].backend='full' is experimental" not in caplog.text


def test_disk_storage_rejects_legacy_prefill_cuda_graph_flag(tmp_path):
    parser = server_args_module.argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(
        [
            "--model-path",
            str(_MODEL_PATH),
            "--ple-storage",
            "disk",
            "--ple-disk-dir",
            str(tmp_path),
            "--device",
            "cuda",
            "--enable-breakable-cuda-graph",
        ]
    )
    with patch("sglang.srt.arg_groups.overrides.is_cuda", return_value=True), patch(
        "sglang.srt.server_args.is_cuda", return_value=True
    ), pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        server_args_module.ServerArgs.from_cli_args(namespace)


def test_checkpoint_disk_storage_rejects_prefill_graph_during_post_init(tmp_path):
    model_path = _checkpoint_with_disk_storage(tmp_path)
    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        _cuda_server_args(
            model_path=str(model_path),
            ple_disk_dir=str(tmp_path / "images"),
            cuda_graph_backend_prefill=Backend.BREAKABLE,
        )


def test_checkpoint_disk_storage_disables_prefill_graph_during_post_init(tmp_path):
    model_path = _checkpoint_with_disk_storage(tmp_path)
    args = _cuda_server_args(
        model_path=str(model_path),
        ple_disk_dir=str(tmp_path / "images"),
    )

    assert args.ple_storage == "disk"
    assert args.disable_prefill_cuda_graph
    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED


def test_non_qwen_storage_rejection_names_the_architecture(tmp_path):
    model_path = tmp_path / "llama"
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
    )
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(model_path)

    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        _cuda_server_args(model_path=str(model_path), ple_storage="pinned")


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("enable_dp_attention", "--enable-dp-attention"),
        ("enable_multi_layer_eagle", "--enable-multi-layer-eagle"),
    ],
)
def test_disk_storage_rejects_graph_hook_bypass_modes(tmp_path, option, message):
    with pytest.raises(ValueError, match=message):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
            tp_size=2,
            dp_size=2,
            **{option: True},
        )


def test_disk_storage_rejects_pdmux_after_resolution(tmp_path):
    with pytest.raises(ValueError, match="enable-pdmux"):
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            ple_storage="disk",
            ple_disk_dir=str(tmp_path),
            enable_pdmux=True,
        )


def test_auto_selected_pinned_storage_names_the_explicit_escape():
    with pytest.raises(ValueError) as exc_info:
        _cuda_server_args(
            model_path=str(_MODEL_PATH),
            cpu_offload_gb=1.0,
        )

    message = str(exc_info.value)
    assert "selected automatically" in message
    assert "--ple-storage gpu" in message


def test_deprecated_ple_offload_embedding_alias_maps_to_pinned(caplog):
    parser = server_args_module.argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)

    with caplog.at_level("WARNING"):
        namespace = parser.parse_args(
            ["--model-path", "dummy", "--ple-offload-embedding"]
        )

    assert namespace.ple_storage == "pinned"
    assert "--ple-offload-embedding" in caplog.text
    assert "--ple-storage pinned" in caplog.text


def test_deprecated_no_ple_offload_embedding_alias_maps_to_gpu(caplog):
    parser = server_args_module.argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)

    with caplog.at_level("WARNING"):
        namespace = parser.parse_args(
            ["--model-path", "dummy", "--no-ple-offload-embedding"]
        )

    assert namespace.ple_storage == "gpu"
    assert "--no-ple-offload-embedding" in caplog.text
    assert "--ple-storage gpu" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        (
            ["--ple-offload-embedding", "--ple-storage", "gpu"],
            "--ple-storage conflicts with --ple-offload-embedding",
        ),
        (
            ["--ple-storage", "gpu", "--ple-offload-embedding"],
            "--ple-offload-embedding conflicts with --ple-storage",
        ),
        (
            ["--no-ple-offload-embedding", "--ple-storage", "pinned"],
            "--ple-storage conflicts with --no-ple-offload-embedding",
        ),
        (
            ["--ple-storage", "pinned", "--no-ple-offload-embedding"],
            "--no-ple-offload-embedding conflicts with --ple-storage",
        ),
    ],
)
def test_deprecated_ple_alias_conflict_names_both_flags(
    arguments, expected_error, capsys
):
    parser = server_args_module.argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--model-path", "dummy", *arguments])

    message = capsys.readouterr().err
    assert message.rstrip().endswith(f"error: {expected_error}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
