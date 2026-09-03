"""PLE disk server argument validation."""

import sys

import pytest

from sglang.srt import server_args as server_args_module
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils.ple_disk import IORING_MAX_ENTRIES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _server_args(**overrides):
    values = {
        "ple_storage": "gpu",
        "ple_disk_dir": "/tmp/ple",
        "ple_disk_hot_cache_gb": 0.0,
        "ple_disk_hot_frequency_file": None,
        "ple_disk_dynamic_cache_gb": 0.0,
        "ple_disk_prefill_buffer_tokens": 16,
        "ple_disk_prefill_read_pages": 2048,
        "ple_disk_max_read_pages": None,
        "ple_disk_stats_log_interval": 0,
        "cpu_offload_gb": 0.0,
        "offload_group_size": 0,
        "pp_size": 1,
        "dllm_algorithm": None,
        "enable_dp_attention": False,
        "enable_multi_layer_eagle": False,
        "enable_pdmux": False,
    }
    values.update(overrides)
    args = object.__new__(server_args_module.ServerArgs)
    for name, value in values.items():
        object.__setattr__(args, name, value)
    return args


def test_offload_compatibility_writes_nothing_after_resolution():
    args = _server_args()
    before = vars(args).copy()
    args._handle_offload_compatibility(resolved=True)
    assert vars(args) == before
    assert args.ple_disk_max_read_pages is None
    assert args.ple_disk_prefill_read_pages == 2048


def test_unused_disk_options_warn_only_after_resolution(caplog):
    args = _server_args()
    with caplog.at_level("WARNING"):
        args._handle_offload_compatibility(resolved=True)
        args._handle_offload_compatibility()
    assert caplog.text.count("are unused with --ple-storage") == 1


def test_explicit_max_read_pages_still_validated():
    args = _server_args(ple_disk_max_read_pages=0)
    with pytest.raises(ValueError):
        args._handle_offload_compatibility()


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
    args = _server_args(**{option: value})
    with pytest.raises(ValueError, match=message):
        args._validate_ple_disk_args()


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
    args = _server_args(**{option: value})
    with pytest.raises(ValueError, match=message):
        args._validate_ple_disk_args()


def test_disk_storage_requires_an_image_directory():
    args = _server_args(ple_storage="disk", ple_disk_dir=None)
    with pytest.raises(ValueError, match="requires --ple-disk-dir"):
        args._handle_offload_compatibility()


def test_disk_storage_accepts_a_creatable_directory(tmp_path):
    target = tmp_path / "new" / "images"
    args = _server_args(ple_storage="disk", ple_disk_dir=str(target))
    args._handle_offload_compatibility()


def test_disk_storage_rejects_an_unreadable_hot_file(tmp_path):
    args = _server_args(
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        ple_disk_hot_frequency_file=str(tmp_path / "missing.bin"),
    )
    with pytest.raises(ValueError, match="readable file"):
        args._handle_offload_compatibility()


def test_disk_storage_accepts_a_readable_hot_file_template(tmp_path):
    (tmp_path / "hot-0.bin").touch()
    args = _server_args(
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        ple_disk_hot_frequency_file=str(tmp_path / "hot-{layer}.bin"),
    )
    args._handle_offload_compatibility()


def test_disk_storage_accepts_literal_braces_in_a_hot_file_path(tmp_path):
    archive = tmp_path / "{archive}"
    archive.mkdir()
    hot_file = archive / "hot.bin"
    hot_file.touch()
    args = _server_args(
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        ple_disk_hot_frequency_file=str(hot_file),
    )
    args._handle_offload_compatibility()


def test_max_read_pages_rejects_io_uring_entry_overflow():
    args = _server_args(ple_disk_max_read_pages=32769)
    with pytest.raises(ValueError, match="32768"):
        args._handle_offload_compatibility()


def test_disk_storage_rejects_json_prefill_cuda_graph(tmp_path):
    args = server_args_module.ServerArgs(
        model_path="dummy",
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        cuda_graph_config={"prefill": {"backend": Backend.FULL}},
    )

    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        args._handle_cuda_graph_config()


def test_disk_storage_rejects_prefill_cuda_graph_convenience_flag(tmp_path):
    args = server_args_module.ServerArgs(
        model_path="dummy",
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        cuda_graph_backend_prefill=Backend.FULL,
    )

    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        args._handle_cuda_graph_config()


def test_disk_storage_disables_default_prefill_cuda_graph(tmp_path):
    args = server_args_module.ServerArgs(
        model_path="dummy",
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
    )

    args._handle_cuda_graph_config()

    assert args.cuda_graph_config.prefill.backend == Backend.DISABLED


def test_disk_storage_rejects_legacy_prefill_cuda_graph_flag(tmp_path):
    parser = server_args_module.argparse.ArgumentParser()
    server_args_module.ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--ple-storage",
            "disk",
            "--ple-disk-dir",
            str(tmp_path),
            "--enable-breakable-cuda-graph",
        ]
    )
    args = server_args_module.ServerArgs.from_cli_args(namespace)

    with pytest.raises(ValueError, match="cuda-graph-backend-prefill"):
        args._handle_cuda_graph_config()


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("enable_dp_attention", "--enable-dp-attention"),
        ("enable_multi_layer_eagle", "--enable-multi-layer-eagle"),
    ],
)
def test_disk_storage_rejects_graph_hook_bypass_modes(tmp_path, option, message):
    args = _server_args(
        ple_storage="disk",
        ple_disk_dir=str(tmp_path),
        **{option: True},
    )
    with pytest.raises(ValueError, match=message):
        args._handle_offload_compatibility(resolved=True)


def test_disk_storage_rejects_pdmux_after_resolution(tmp_path):
    args = _server_args(
        ple_storage="disk", ple_disk_dir=str(tmp_path), enable_pdmux=True
    )
    args._handle_offload_compatibility()
    with pytest.raises(ValueError, match="enable-pdmux"):
        args._handle_offload_compatibility(resolved=True)


def test_auto_selected_pinned_storage_names_the_explicit_escape():
    args = _server_args(ple_storage="pinned", cpu_offload_gb=1.0)
    args._resolved_overrides = [("qwen4 automatic storage", {"ple_storage": "pinned"})]

    with pytest.raises(ValueError) as exc_info:
        args._handle_offload_compatibility(resolved=True)

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
