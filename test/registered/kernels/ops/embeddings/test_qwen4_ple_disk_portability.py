import ctypes
import errno
import importlib.util
import json
import os
import struct
import sys
import threading
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models import qwen4_ple_disk as disk
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpDiskEmbedding,
    Qwen4ExpModel,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.models.qwen4_ple_hash import (
    PLEMetadata,
    hash_contexts_numpy,
    hash_token_stream_numpy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def replace_pinned_allocators_without_cuda(monkeypatch):
    if torch.cuda.is_available():
        return

    def cpu_host_tensor(*args, pin_memory=True, **kwargs):
        return torch.empty(*args, **kwargs)

    monkeypatch.setattr(disk, "_allocate_host_tensor", cpu_host_tensor)
    monkeypatch.setattr(qwen4_exp_module, "_allocate_host_tensor", cpu_host_tensor)


@pytest.mark.parametrize("rows", [4, 8])
def test_gpu_mode_ngram_ids_match_the_pre_disk_stream(monkeypatch, rows):
    expected = torch.arange(rows * 16, dtype=torch.long).view(rows, 16)
    valid_tokens = torch.tensor([True] + [False] * (rows - 1))
    batch = SimpleNamespace(
        ngram_context=torch.arange(rows * 3, dtype=torch.long).view(rows, 3),
        use_decode_fast_path=True,
        valid_tokens=valid_tokens,
        mode=ForwardMode.DECODE,
    )
    pool = SimpleNamespace(ple_window_cache=None)
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)

    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module._mask_invalid_ngram_ids = False
    module._hash_contexts = lambda contexts, decode_sized=False: expected.clone()

    actual = module.compute_ngram_ids(batch)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[~valid_tokens], expected[~valid_tokens])

    module._mask_invalid_ngram_ids = True
    disk_ids = module.compute_ngram_ids(batch)
    assert torch.equal(disk_ids[valid_tokens], expected[valid_tokens])
    assert torch.all(disk_ids[~valid_tokens] == -1)


def _fp8_rows(count: int) -> torch.Tensor:
    raw = torch.arange(count * disk.ROW_BYTES, dtype=torch.uint8).reshape(
        count, disk.ROW_BYTES
    )
    raw[(raw & 0x7F) == 0x7F] = 0
    return raw.view(torch.float8_e4m3fn)


class _FakeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FakeFetcherLibrary:
    def __init__(
        self,
        image_bytes: bytes,
        *,
        create_errno=0,
        failure_stage=0,
        read_results=(),
        last_error=None,
    ):
        self.image_bytes = image_bytes
        self.create_errno = create_errno
        self.failure_stage = failure_stage
        self.read_results = iter(read_results)
        self.last_error = last_error
        self.create_args = None
        self.read_buffer = None
        self.read_buffer_bytes = None
        self.ple_fetcher_create = _FakeFunction(self._create)
        self.ple_fetcher_read = _FakeFunction(self._read)
        self.ple_fetcher_last_error = _FakeFunction(self._last_error)
        self.ple_fetcher_destroy = _FakeFunction(lambda handle: 0)

    def _create(
        self,
        file_fd,
        buffer,
        buffer_bytes,
        max_pages,
        register_buffer,
        failure_stage,
    ):
        self.create_args = (
            file_fd,
            buffer,
            buffer_bytes,
            max_pages,
            register_buffer,
        )
        failure_stage._obj.value = self.failure_stage
        if self.create_errno:
            ctypes.set_errno(self.create_errno)
            return None
        return 1

    def _read(self, handle, offsets, count, buffer, buffer_bytes):
        self.read_buffer = buffer
        self.read_buffer_bytes = buffer_bytes
        if count * disk.PAGE_BYTES > buffer_bytes:
            return -errno.EFAULT
        result = next(self.read_results, 0)
        if result:
            return result
        for index in range(count):
            offset = offsets[index]
            page = self.image_bytes[offset : offset + disk.PAGE_BYTES]
            ctypes.memmove(buffer + index * disk.PAGE_BYTES, page, len(page))
        return 0

    def _last_error(self, handle, index, result):
        if self.last_error is None:
            return 0
        index._obj.value, result._obj.value = self.last_error
        return 1


def _patch_fetcher_library(monkeypatch, image: disk.PLEImage, **kwargs):
    library = _FakeFetcherLibrary(image.path.read_bytes(), **kwargs)
    monkeypatch.setattr(disk, "_find_helper_library", lambda: Path("fake-fetcher.so"))
    monkeypatch.setattr(disk.ctypes, "CDLL", lambda *args, **opts: library)
    return library


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
    }
    values.update(overrides)
    args = object.__new__(server_args_module.ServerArgs)
    for name, value in values.items():
        object.__setattr__(args, name, value)
    return args


@pytest.mark.parametrize(
    ("uses_host_tables", "expected"), [(True, 1024), (False, 4096)]
)
def test_max_read_pages_arch_default(monkeypatch, uses_host_tables, expected):
    properties = SimpleNamespace(
        pageableMemoryAccessUsesHostPageTables=uses_host_tables
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device=0: properties
    )
    assert disk.resolve_max_read_pages(None) == expected
    assert disk.resolve_max_read_pages(73) == 73


def test_max_read_pages_queries_the_current_device(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: seen.append(device)
        or SimpleNamespace(pageableMemoryAccessUsesHostPageTables=True),
    )
    assert disk.resolve_max_read_pages(None) == 1024
    assert seen == [3]


def test_cudart_fallback_uses_current_device_and_versioned_library(monkeypatch):
    calls = []
    attributes = []

    def get_attribute(value, attribute, device):
        value._obj.value = 1
        attributes.append((attribute, device))
        return 0

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 5)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setattr(
        disk.ctypes,
        "CDLL",
        lambda name, **kwargs: calls.append(name)
        or SimpleNamespace(cudaDeviceGetAttribute=_FakeFunction(get_attribute)),
    )
    assert disk.pageable_memory_access_uses_host_page_tables()
    assert calls[0].startswith("libcudart.so.")
    assert attributes == [
        (disk.CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, 5)
    ]


def test_cudart_query_failure_is_reported(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setattr(
        disk.ctypes,
        "CDLL",
        lambda name, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    with pytest.raises(RuntimeError, match="Could not query CUDA"):
        disk.pageable_memory_access_uses_host_page_tables(2)


def test_helper_is_loaded_from_the_sgl_kernel_package(tmp_path, monkeypatch):
    package_dir = tmp_path / "sgl_kernel"
    package_dir.mkdir()
    helper = package_dir / "qwen4_ple_disk_fetcher.so"
    helper.touch()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post2")
    assert disk._find_helper_library() == helper


def test_missing_helper_names_the_required_sgl_kernel_version(monkeypatch):
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=disk.REQUIRED_SGL_KERNEL_VERSION):
        disk._find_helper_library()


def test_missing_helper_distinguishes_old_and_disabled_wheels(tmp_path, monkeypatch):
    package_dir = tmp_path / "sgl_kernel"
    package_dir.mkdir()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post1")
    with pytest.raises(RuntimeError, match="found 0.4.6.post1"):
        disk._find_helper_library()

    (package_dir / "qwen4_ple_disk_fetcher.build").write_text("disabled\n")
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post2")
    with pytest.raises(RuntimeError, match="built without.*io_uring"):
        disk._find_helper_library()


def test_helper_rejects_target_post_release_development_wheel(tmp_path, monkeypatch):
    package_dir = tmp_path / "sgl_kernel"
    package_dir.mkdir()
    helper = package_dir / "qwen4_ple_disk_fetcher.so"
    helper.touch()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(
        disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post2.dev0"
    )
    with pytest.raises(RuntimeError, match="found 0.4.6.post2.dev0"):
        disk._find_helper_library()


def test_helper_accepts_a_higher_epoch_version(tmp_path, monkeypatch):
    package_dir = tmp_path / "sgl_kernel"
    package_dir.mkdir()
    helper = package_dir / "qwen4_ple_disk_fetcher.so"
    helper.touch()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "1!0.4.6.post1")
    assert disk._find_helper_library() == helper


def test_poisoned_fetcher_error_requires_restart(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        read_results=(-errno.EIO, -disk.FETCHER_ERR_POISONED),
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        with pytest.raises(OSError) as first_error:
            reader.read(np.array([0], dtype=np.int64))
        assert first_error.value.errno == errno.EIO
        with pytest.raises(RuntimeError, match="poisoned.*restart"):
            reader.read(np.array([0], dtype=np.int64))
    finally:
        reader.close()


def test_fetcher_error_reports_page_and_short_read_size(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        read_results=(-errno.EIO,),
        last_error=(0, 123),
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        with pytest.raises(OSError, match="short read of 123 bytes on PLE page 0"):
            reader.read(np.array([0], dtype=np.int64))
    finally:
        reader.close()


def test_native_short_read_keeps_later_reads_quiescent(tmp_path, monkeypatch):
    rows = _fp8_rows(50)
    image = disk.build_test_image(tmp_path, rows)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    try:
        reader = disk.DirectPageReader(image, max_pages=2)
    except RuntimeError as exc:
        message = str(exc)
        if (
            "requires sglang-kernel" in message
            or "qwen4_ple_disk_fetcher" in message
            or "io_uring support" in message
        ):
            pytest.skip(str(exc))
        raise
    except OSError as exc:
        if exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
        }:
            pytest.skip(f"native PLE reader is unavailable: {exc}")
        raise

    try:
        os.truncate(image.path, image.path.stat().st_size - disk.PAGE_BYTES // 2)
        with pytest.raises(OSError) as short_read:
            reader.read(np.array([1], dtype=np.int64))
        assert short_read.value.errno == errno.EIO

        page = reader.read(np.array([0], dtype=np.int64))[0]
        expected = rows.view(torch.uint8)[: disk.ROWS_PER_PAGE].reshape(-1)
        assert np.array_equal(
            page[: disk.ROWS_PER_PAGE * disk.ROW_BYTES], expected.numpy()
        )
    finally:
        reader.close()


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


def test_max_read_pages_rejects_io_uring_entry_overflow():
    args = _server_args(ple_disk_max_read_pages=32769)
    with pytest.raises(ValueError, match="32768"):
        args._handle_offload_compatibility()


@pytest.mark.parametrize("max_pages", [0, disk.IORING_MAX_ENTRIES + 1])
def test_direct_reader_rejects_invalid_io_uring_entry_count(tmp_path, max_pages):
    image = disk.build_test_image(tmp_path, _fp8_rows(1))
    with pytest.raises(ValueError, match="max_pages must be between"):
        disk.DirectPageReader(image, max_pages=max_pages)


def test_resolved_page_limit_reaches_fetcher_registration(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(50))
    library = _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=37)
    try:
        assert reader.max_pages == 37
        assert library.create_args[2] == 37 * disk.PAGE_BYTES
        assert library.create_args[3] == 37
    finally:
        reader.close()


def test_read_abi_passes_the_staging_buffer_length(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(50))
    library = _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=2)
    try:
        reader.read(np.array([0, 1], dtype=np.int64))
        assert library.read_buffer_bytes == reader.staging.numel()
    finally:
        reader.close()


def test_memlock_error_names_limit_bytes_and_flag(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=errno.ENOMEM,
        failure_stage=disk.FETCHER_FAILURE_REGISTER_BUFFER,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: False
    )
    torch_empty = torch.empty
    monkeypatch.setattr(
        disk.torch,
        "empty",
        lambda size, **kwargs: torch_empty(size, dtype=kwargs["dtype"]),
    )
    with pytest.raises(OSError) as exc_info:
        disk.DirectPageReader(image, max_pages=19)
    message = str(exc_info.value)
    assert "RLIMIT_MEMLOCK" in message
    assert str(19 * disk.PAGE_BYTES) in message
    assert "--ple-disk-max-read-pages" in message


@pytest.mark.parametrize("blocked_errno", [errno.EPERM, errno.EACCES, errno.ENOSYS])
def test_blocked_io_uring_error_has_operator_actions(
    tmp_path, monkeypatch, blocked_errno
):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=blocked_errno,
        failure_stage=disk.FETCHER_FAILURE_SETUP,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    with pytest.raises(OSError) as exc_info:
        disk.DirectPageReader(image, max_pages=2)
    message = str(exc_info.value)
    assert "io_uring is blocked" in message
    assert "container seccomp" in message
    assert "kernel.io_uring_disabled" in message
    assert "--ple-storage pinned" in message


def test_registration_permission_error_is_not_reported_as_blocked_setup(
    tmp_path, monkeypatch
):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=errno.EPERM,
        failure_stage=disk.FETCHER_FAILURE_REGISTER_BUFFER,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    with pytest.raises(OSError) as exc_info:
        disk.DirectPageReader(image, max_pages=2)
    assert "io_uring is blocked" not in str(exc_info.value)


def test_manifest_records_ranges_and_rejects_missing_ranges(tmp_path):
    rows = _fp8_rows(100)
    builder = disk.PLEImageBuilder(tmp_path, "ranges", 0, 1, 0, 100)
    builder.add_shard("shard_1", rows[50:], 50, 100)
    builder.add_shard("shard_0", rows[:50], 0, 50)
    image, _, _ = builder.finalize(0.5)
    manifest_path = image.path.parent / "manifest.json"
    document = json.loads(manifest_path.read_text())
    assert [(item["row_start"], item["row_end"]) for item in document["shards"]] == [
        (0, 50),
        (50, 100),
    ]

    del document["shards"][0]["row_start"]
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="rebuild"):
        disk.open_ple_image(image.path)
    with pytest.raises(ValueError, match="rebuild"):
        disk.PLEImageBuilder(tmp_path, "ranges", 0, 1, 0, 100)


def test_manifest_from_newer_install_rejects_older_image(tmp_path):
    rows = _fp8_rows(25)
    image = disk.build_test_image(
        tmp_path, rows, config_sha256="torn-install", weight_scale=0.5
    )
    manifest_path = image.path.parent / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["fingerprint"] = "newer-image-fingerprint"
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="delete.*before rebuilding"):
        disk.PLEImageBuilder(tmp_path, "torn-install", 0, 1, 0, 25)


@pytest.mark.parametrize("failure_call", [2, 3])
def test_finalize_crash_before_image_install_is_recoverable(
    tmp_path, monkeypatch, failure_call
):
    rows = _fp8_rows(25)
    real_replace = os.replace
    destinations = []

    def fail_between_installs(source, destination):
        destinations.append(Path(destination).name)
        if len(destinations) == failure_call:
            raise OSError(errno.EIO, "injected install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(disk.os, "replace", fail_between_installs)
    builder = disk.PLEImageBuilder(tmp_path, "crash-order", 0, 1, 0, 25)
    builder.add_shard("shard", rows, 0, 25)
    with pytest.raises(OSError, match="injected install failure"):
        builder.finalize(0.5)

    assert (
        destinations[:failure_call]
        == ["rank0.crc32", "manifest.json", "rank0.bin"][:failure_call]
    )
    assert not list(tmp_path.rglob("*.tmp"))

    monkeypatch.setattr(disk.os, "replace", real_replace)
    retry = disk.PLEImageBuilder(tmp_path, "crash-order", 0, 1, 0, 25)
    retry.add_shard("shard", rows, 0, 25)
    image, reused, _ = retry.finalize(0.5)
    assert not reused
    assert disk.open_ple_image(image.path).path == image.path


def test_builder_cleans_raw_and_packed_temporaries_after_enospc(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    builder = disk.PLEImageBuilder(tmp_path, "enospc", 0, 1, 0, 25)
    builder.add_shard("shard", rows, 0, 25)
    monkeypatch.setattr(
        disk.os,
        "pread",
        lambda *args: (_ for _ in ()).throw(OSError(errno.ENOSPC, "injected")),
    )
    with pytest.raises(OSError) as error:
        builder.finalize(0.5)
    assert error.value.errno == errno.ENOSPC
    assert not list(tmp_path.rglob("*.tmp"))


def test_builder_cleans_raw_temporary_after_later_shard_validation_error(tmp_path):
    rows = _fp8_rows(50)
    builder = disk.PLEImageBuilder(tmp_path, "invalid-shard", 0, 1, 0, 50)
    builder.add_shard("first", rows[:25], 0, 25)
    assert list(tmp_path.rglob("*.tmp"))

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        builder.add_shard("second", rows[25:].to(torch.bfloat16), 25, 50)
    assert not list(tmp_path.rglob("*.tmp"))


def test_transfer_buffers_keep_only_largest_shape_per_device():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    embedding.embedding_dim = disk.ROW_BYTES
    embedding._transfer_buffers = {}

    small = torch.empty((2, 16), dtype=torch.long)
    medium = torch.empty((4, 16), dtype=torch.long)
    tiny = torch.empty((1, 16), dtype=torch.long)
    first = embedding._get_transfer_buffers(small)
    second = embedding._get_transfer_buffers(medium)
    third = embedding._get_transfer_buffers(tiny)

    assert len(embedding._transfer_buffers) == 1
    assert first[0].shape == small.shape
    assert second[0].shape == medium.shape
    assert third[0].shape == tiny.shape
    assert (
        third[0].untyped_storage().data_ptr() == second[0].untyped_storage().data_ptr()
    )


def test_transfer_buffer_retains_the_configured_prefill_working_set():
    retain_rows = qwen4_exp_module._ple_transfer_buffer_retain_rows(8192, 16)
    assert retain_rows == 131072


def test_rank_executor_initializes_its_cuda_device(monkeypatch):
    captured = {}
    selected = []

    class RecordingExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(qwen4_exp_module, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(
        qwen4_exp_module.torch.cuda,
        "set_device",
        lambda device: selected.append(device),
    )
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    embedding._rank = 7
    embedding._cuda_device = 5

    embedding._new_executor()
    captured["initializer"](*captured["initargs"])

    assert captured["thread_name_prefix"] == "ple-disk-rank7"
    assert selected == [5]


def test_current_cuda_device_propagates_initialization_failure(monkeypatch):
    monkeypatch.setattr(
        disk.torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(RuntimeError("CUDA is not initialized")),
    )

    with pytest.raises(RuntimeError, match="CUDA is not initialized"):
        disk._current_cuda_device()


def test_prefill_executor_initializes_the_current_cuda_device(tmp_path, monkeypatch):
    captured = {}
    selected = []

    class RecordingExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self, wait=True):
            pass

    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    monkeypatch.setattr(disk, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(disk.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(disk, "_current_cuda_device", lambda: 4)
    monkeypatch.setattr(
        disk.torch.cuda, "set_device", lambda device: selected.append(device)
    )

    fetcher = disk.DiskRowFetcher(
        image,
        hot_cache_gb=0,
        prefill_buffer_tokens=1,
        prefill_read_pages=1,
        max_pages=1,
    )
    try:
        captured["initializer"](*captured["initargs"])
    finally:
        fetcher.close()

    assert captured["thread_name_prefix"] == "ple-prefill"
    assert selected == [4]


def test_disk_graph_replay_wait_is_a_noop_without_staged_work():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    waits = []
    embedding.wait_for_graph_step = lambda generation: waits.append(generation)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._graph_replay_generation = None
    layer._graph_replay_stage_expected = False

    layer.wait_cuda_graph_replay()

    assert waits == []


def test_disk_capture_retains_the_graph_updated_lookup_buffer(monkeypatch):
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.gather = lambda input_ids, out: out

    captured_ids = torch.arange(64, dtype=torch.long).view(4, 16)
    ngram_embedding = SimpleNamespace(
        ngram_embedding=offloaded,
        ngram_heads=16,
        gather_dp_tokens=False,
        compute_ngram_ids=lambda batch: captured_ids,
        _prepare_embedding_lookup=lambda ids, forward_batch, physical_tokens: (
            ids,
            physical_tokens,
        ),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = ngram_embedding
    layer._prefetch_stream = object()
    layer._prefetch_state = None
    layer._future_lookup_contexts = None
    layer._graph_lookup_id_buffers = {}
    layer._graph_lookup_validation_due = set()
    layer._is_capturing = lambda: True
    layer._get_prefetch_buffer = lambda tokens, ids: torch.empty(
        (tokens, 16, 10), dtype=torch.bfloat16
    )

    batch = SimpleNamespace(physical_tokens=4, processed_tokens=4)
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(4),
        global_dp_buffer_len=None,
    )
    layer.start_prefetch(batch, forward_batch)
    assert layer._prefetch_state[0].shape == (4, 16, 10)
    captured_ids[0, 0] = -1
    assert layer._graph_lookup_id_buffers[4].data_ptr() == captured_ids.data_ptr()
    assert layer._graph_lookup_id_buffers[4][0, 0].item() == -1


def test_graph_replay_prepares_shared_batch_once(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import CudaGraphReplayInput

    prepared_batch = SimpleNamespace(physical_tokens=2)
    forward_batch = SimpleNamespace(input_ids=torch.arange(1))
    prepare_calls = []
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)) or prepared_batch,
    )

    disk_embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(disk_embedding)
    received = []

    def make_layer(value):
        ngram = SimpleNamespace(
            ngram_embedding=disk_embedding,
            ngram_size=3,
            eos_token_id=2,
            compute_ngram_ids=lambda batch: torch.full((2, 1), value),
            _prepare_embedding_lookup=lambda ids, batch, tokens: (ids + 10, tokens),
        )
        return SimpleNamespace(
            ple_embedding=ngram,
            prepare_cuda_graph_replay=lambda batch, lookup_ids: received.append(
                (batch, lookup_ids.clone())
            ),
        )

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([make_layer(1), make_layer(2)])
    replay = CudaGraphReplayInput(
        padded_num_tokens=2,
        input_ids=torch.arange(2),
        req_pool_indices=torch.arange(2),
        seq_lens=torch.ones(2),
        seq_lens_sum=2,
        out_cache_loc=torch.ones(2),
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=None,
        runtime_forward_batch=forward_batch,
    )

    model.prepare_cuda_graph_replay(replay)

    assert len(prepare_calls) == 1
    assert [item[0] for item in received] == [prepared_batch, prepared_batch]
    assert [item[1].tolist() for item in received] == [
        [[11], [11]],
        [[12], [12]],
    ]


def test_graph_replay_prepare_rolls_back_every_disk_layer(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import CudaGraphReplayInput

    disk_embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(disk_embedding)
    events = []

    def make_layer(index, fail=False):
        ngram = SimpleNamespace(
            ngram_embedding=disk_embedding,
            compute_ngram_ids=lambda batch: torch.zeros((1, 1), dtype=torch.long),
            _prepare_embedding_lookup=lambda ids, batch, tokens: (ids, tokens),
        )

        def prepare(batch, lookup_ids):
            events.append(("prepare", index))
            if fail:
                raise OSError("injected layer failure")

        return SimpleNamespace(
            ple_embedding=ngram,
            prepare_cuda_graph_replay=prepare,
            reset_cuda_graph_replay=lambda: events.append(("reset", index)),
        )

    layers = [make_layer(0), make_layer(1, fail=True), make_layer(2)]
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter(layers)
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *args, **kwargs: SimpleNamespace(physical_tokens=1),
    )

    replay = CudaGraphReplayInput(
        padded_num_tokens=1,
        input_ids=torch.zeros(1, dtype=torch.long),
        req_pool_indices=torch.zeros(1, dtype=torch.long),
        seq_lens=torch.ones(1, dtype=torch.long),
        seq_lens_sum=1,
        out_cache_loc=torch.ones(1, dtype=torch.long),
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=None,
        runtime_forward_batch=SimpleNamespace(),
    )

    with pytest.raises(OSError, match="injected layer failure"):
        model.prepare_cuda_graph_replay(replay)

    assert events == [
        ("prepare", 0),
        ("prepare", 1),
        ("reset", 0),
        ("reset", 1),
        ("reset", 2),
    ]


def test_failed_embedding_future_does_not_block_the_next_graph_step(monkeypatch):
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.embedding_dim = 4
    embedding._active_transfer_device = "cuda:0"
    embedding._graph_generation = 0
    embedding._active_graph_generation = None
    failed = Future()
    failed.set_exception(OSError("injected fetch failure"))
    embedding._future = failed

    with pytest.raises(OSError, match="injected fetch failure"):
        embedding.wait_for_prefetch()

    monkeypatch.setattr(embedding, "_launch_fetch", lambda *args, **kwargs: None)
    generation = embedding.stage_graph_step(
        torch.zeros((1,), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.bfloat16),
    )
    assert generation == 1


def test_layer_wait_resets_graph_state_after_fetch_failure():
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.wait_for_graph_step = lambda generation: (_ for _ in ()).throw(
        OSError("injected layer wait")
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=offloaded)
    layer._graph_replay_generation = 4
    layer._graph_replay_stage_expected = True
    layer._graph_replay_lookup_tokens = 1
    layer._graph_replay_prefetch_buffer = torch.zeros(1)
    layer._prefetch_stream = object()
    layer._validate_graph_staging = False

    with pytest.raises(OSError, match="injected layer wait"):
        layer.wait_cuda_graph_replay()

    assert layer._graph_replay_generation is None
    assert not layer._graph_replay_stage_expected
    assert layer._graph_replay_lookup_tokens is None
    assert layer._graph_replay_prefetch_buffer is None


def test_model_wait_keeps_replay_validation_pending_until_finish():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    events = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(ngram_embedding=embedding),
        wait_cuda_graph_replay=lambda: events.append("wait"),
        reset_cuda_graph_replay=lambda: events.append("reset"),
    )
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model._ple_layers = lambda: iter([layer])

    model.wait_cuda_graph_replay()

    assert events == ["wait"]


def test_graph_lookup_validation_defaults_to_every_replay():
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_replay_steps = 0
    layer._graph_lookup_validation_due = {4}
    layer._graph_lookup_validation_interval = 1

    assert layer._graph_lookup_validation_required(4)
    assert all(layer._graph_lookup_validation_required(4) for _ in range(8))

    layer._graph_lookup_validation_interval = 256
    layer._graph_replay_steps = 0
    layer._graph_lookup_validation_due = {4}
    assert layer._graph_lookup_validation_required(4)
    assert all(not layer._graph_lookup_validation_required(4) for _ in range(254))
    assert layer._graph_lookup_validation_required(4)

    layer._graph_lookup_validation_interval = 0
    with pytest.raises(ValueError, match="must be positive"):
        layer._graph_lookup_validation_required(4)


def test_graph_replay_shared_buffer_requires_a_captured_size(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_prefetch_buffer = torch.empty((8, 16))
    layer._graph_lookup_id_buffers = {4: torch.empty((4, 1), dtype=torch.long)}
    monkeypatch.setattr(qwen4_exp_module, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp_module, "is_sm121", lambda: False)

    assert layer._select_graph_prefetch_buffer(4).shape == (4, 16)
    with pytest.raises(RuntimeError, match="no captured staging buffer"):
        layer._select_graph_prefetch_buffer(2)


def test_capture_start_drops_references_from_the_previous_graph():
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_prefetch_buffers = {1: object()}
    layer._graph_lookup_id_buffers = {1: object()}
    layer._graph_embedding_snapshot_buffers = {1: object()}
    layer._graph_lookup_validation_due = {1}

    layer.reset_cuda_graph_capture_buffers()

    assert layer._graph_prefetch_buffers == {}
    assert layer._graph_lookup_id_buffers == {}
    assert layer._graph_embedding_snapshot_buffers == {}
    assert layer._graph_lookup_validation_due == set()


def test_aborted_graph_replay_reset_discards_pending_validation(monkeypatch):
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    monkeypatch.setattr(embedding, "reset_graph_step", lambda: None)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._graph_replay_generation = 1
    layer._graph_replay_stage_expected = True
    layer._graph_replay_lookup_tokens = 4
    layer._graph_replay_prefetch_buffer = object()
    layer._pending_graph_lookup_validation = object()
    layer._pending_graph_embedding_validation = object()

    layer.reset_cuda_graph_replay()

    assert layer._pending_graph_lookup_validation is None
    assert layer._pending_graph_embedding_validation is None


def test_graph_lookup_validation_checks_the_current_replay(monkeypatch):
    class ReadyEvent:
        def record(self, stream):
            self.stream = stream

        def query(self):
            return True

    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "Event", ReadyEvent)
    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "current_stream", lambda: object())
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._pending_graph_embedding_validation = None
    layer._completed_graph_embedding_validation = deque()
    layer._pending_graph_lookup_validation = (
        2,
        torch.tensor([[3], [5]], dtype=torch.long),
    )
    layer._completed_graph_lookup_validation = deque()
    layer._graph_validation_free_slots = deque()
    layer._graph_lookup_id_buffers = {2: torch.tensor([[3], [7]], dtype=torch.long)}

    layer.finish_cuda_graph_replay()
    assert len(layer._completed_graph_lookup_validation) == 1
    with pytest.raises(RuntimeError, match="lookup IDs differ"):
        layer.validate_cuda_graph_replay()


def test_graph_replay_uses_the_explicit_padded_token_extent(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import (
        CudaGraphReplayInput,
        ForwardMode,
    )

    runtime = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.ones(2, dtype=torch.int32),
        seq_lens_sum=2,
        spec_info=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        global_num_tokens_cpu=None,
        global_num_tokens_gpu=None,
        dp_padding_mode=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        _original_forward_mode=None,
        num_token_non_padded_cpu=1,
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
        extend_seq_lens_cpu=[1],
        extend_prefix_lens_cpu=[0],
        extend_num_tokens=1,
    )
    replay = CudaGraphReplayInput(
        padded_num_tokens=2,
        input_ids=torch.arange(2),
        req_pool_indices=torch.arange(2, dtype=torch.int32),
        seq_lens=runtime.seq_lens,
        seq_lens_sum=runtime.seq_lens_sum,
        out_cache_loc=torch.ones(2, dtype=torch.int64),
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=None,
        runtime_forward_batch=runtime,
    )
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
        get_ngram_context=lambda indices: torch.zeros(
            (indices.numel(), 2), dtype=torch.long
        ),
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    batch = qwen4_exp_module._prepare_ple_batch(
        replay.input_ids,
        runtime,
        ngram_size=3,
        ngram_eos_token_id=2,
        replay=replay,
    )
    assert batch.physical_tokens == 2
    assert batch.processed_tokens == 2
    assert batch.lengths.tolist() == [1, 1]


def test_image_reuse_is_scoped_to_the_ple_module_prefix(tmp_path):
    rows = _fp8_rows(25)
    for prefix in ("model.layers.1.ple", "model.layers.9.ple"):
        builder = disk.PLEImageBuilder(
            tmp_path,
            "two-ple-layers",
            0,
            1,
            0,
            25,
            module_prefix=prefix,
        )
        builder.add_shard(f"{prefix}.shard_0.weight", rows, 0, 25)
        _, reused, _ = builder.finalize(0.5)
        assert not reused

    for prefix in ("model.layers.1.ple", "model.layers.9.ple"):
        builder = disk.PLEImageBuilder(
            tmp_path,
            "two-ple-layers",
            0,
            1,
            0,
            25,
            module_prefix=prefix,
        )
        builder.add_shard(f"{prefix}.shard_0.weight", rows, 0, 25)
        image, reused, _ = builder.finalize(0.5)
        assert reused
        assert image.header["module_prefix"] == prefix


def test_hot_cache_deduplicates_before_applying_capacity(tmp_path, monkeypatch):
    source_rows = _fp8_rows(25)
    image = disk.build_test_image(tmp_path, source_rows)
    real_empty = torch.empty

    def cpu_empty(*shape, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*shape, **kwargs)

    monkeypatch.setattr(disk.torch, "empty", cpu_empty)
    two_rows_gb = (2 * disk.ROW_BYTES + 1) / (1 << 30)
    cache = disk.RankSelectHotCache(
        image, np.array([2, 2, 1, 3], dtype=np.int64), two_rows_gb
    )
    assert cache.rows.shape[0] == 2
    requested = np.array([1, 2, 3], dtype=np.int64)
    hit, slots = cache.lookup(requested)
    assert hit.tolist() == [True, True, False]
    assert np.array_equal(
        cache.rows.numpy()[slots[hit]],
        source_rows.view(torch.uint8).numpy()[requested[hit]],
    )


@pytest.mark.parametrize("seed", range(5))
def test_hot_file_frequency_order_round_trips_through_rank_select_cache(tmp_path, seed):
    rng = np.random.default_rng(seed)
    source_rows = _fp8_rows(64)
    image = disk.build_test_image(tmp_path, source_rows, tp_size=2)
    rank_zero = rng.permutation(64).astype(np.uint32)
    rank_one = (64 + rng.permutation(64)).astype(np.uint32)
    path = tmp_path / "hot.bin"
    disk.write_hot_frequency_file(
        path,
        {0: rank_zero, 1: rank_one},
        fingerprint=image.header["fingerprint"],
        total_rows=128,
        tp_size=2,
        padding_divisor=64,
    )

    loaded = disk.read_hot_frequency_file(
        path,
        0,
        expected_fingerprint=image.header["fingerprint"],
        expected_tp_size=2,
        expected_vocab_start=0,
        expected_vocab_end=64,
    )
    assert np.array_equal(loaded, rank_zero)

    keep = 7
    cache = disk.RankSelectHotCache(
        image, loaded, (keep * disk.ROW_BYTES + 1) / (1 << 30)
    )
    requested = np.arange(64, dtype=np.int64)
    hit, slots = cache.lookup(requested)
    expected_ids = np.sort(rank_zero[:keep].astype(np.int64))
    assert np.array_equal(requested[hit], expected_ids)
    assert np.array_equal(
        cache.rows.numpy()[slots[hit]],
        source_rows.view(torch.uint8).numpy()[expected_ids],
    )


def test_hot_frequency_template_selects_two_ple_layer_files(tmp_path):
    rows = _fp8_rows(25)
    second_rows = rows.clone()
    second_rows.view(torch.uint8)[0, 0] = 1
    images = [
        disk.build_test_image(tmp_path, layer_rows, module_prefix=f"ple.{layer}")
        for layer, layer_rows in enumerate((rows, second_rows))
    ]
    assert images[0].header["fingerprint"] != images[1].header["fingerprint"]
    template = str(tmp_path / "hot-{layer}.bin")

    for layer, image in enumerate(images):
        path = disk.resolve_hot_frequency_file(template, layer, len(images))
        disk.write_hot_frequency_file(
            path,
            {0: np.array([layer], dtype=np.uint32)},
            fingerprint=image.header["fingerprint"],
            total_rows=25,
            tp_size=1,
            padding_divisor=1,
        )

    for layer, image in enumerate(images):
        path = disk.resolve_hot_frequency_file(template, layer, len(images))
        loaded = disk.read_hot_frequency_file(
            path,
            0,
            expected_fingerprint=image.header["fingerprint"],
        )
        assert loaded.tolist() == [layer]


def test_multiple_ple_layers_require_a_hot_frequency_template():
    with pytest.raises(ValueError, match=r"contain \{layer\}"):
        disk.resolve_hot_frequency_file("hot.bin", 0, 2)


def test_hot_file_writer_requires_the_image_fingerprint(tmp_path):
    path = tmp_path / "hot.bin"
    with pytest.raises(ValueError, match="require an image fingerprint"):
        disk.write_hot_frequency_file(
            path,
            {0: np.array([1], dtype=np.uint32)},
            fingerprint="",
            total_rows=4,
            tp_size=1,
            padding_divisor=1,
        )


def test_hot_file_rejects_a_rank_range_past_eof(tmp_path):
    path = tmp_path / "hot.bin"
    disk.write_hot_frequency_file(
        path,
        {0: np.array([1], dtype=np.uint32)},
        fingerprint="image",
        total_rows=4,
        tp_size=1,
        padding_divisor=1,
    )
    header = disk._read_metadata_page(path, disk.HOT_MAGIC)
    header["ranks"][0]["count"] = 1 << 40
    with path.open("r+b") as handle:
        handle.write(disk._write_metadata_page(disk.HOT_MAGIC, header))
    with pytest.raises(ValueError, match="file size"):
        disk.read_hot_frequency_file(path, 0, expected_fingerprint="image")


def test_direct_reader_uses_device_block_alignment(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    monkeypatch.setattr(disk, "_logical_block_size", lambda path: 8192)
    with pytest.raises(RuntimeError, match="4096-byte logical blocks"):
        disk.DirectPageReader(image, max_pages=1)


def test_unknown_device_block_alignment_stops_before_direct_io(tmp_path, monkeypatch):
    path = tmp_path / "image.bin"
    path.touch()
    monkeypatch.setattr(
        disk.os,
        "stat",
        lambda target: (_ for _ in ()).throw(OSError("sysfs unavailable")),
    )
    with pytest.raises(RuntimeError, match="logical block size"):
        disk._logical_block_size(path)


def test_direct_reader_closes_fd_when_native_destroy_fails(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    fd = reader.fd
    reader.lib.ple_fetcher_destroy = _FakeFunction(lambda handle: -errno.EBUSY)
    with pytest.raises(OSError, match="shutdown failed"):
        reader.close()
    with pytest.raises(OSError) as exc_info:
        os.fstat(fd)
    assert exc_info.value.errno == errno.EBADF


def test_fetcher_constructor_closes_decode_reader_when_prefill_reader_fails(
    tmp_path, monkeypatch
):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    readers = []

    class FakeReader:
        def __init__(self, image, max_pages):
            if readers:
                raise OSError(errno.ENOMEM, "injected prefill registration failure")
            self.closed = False
            readers.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(disk, "DirectPageReader", FakeReader)
    monkeypatch.setattr(
        disk, "RankSelectHotCache", lambda *args, **kwargs: SimpleNamespace()
    )
    with pytest.raises(OSError, match="injected prefill"):
        disk.DiskRowFetcher(
            image,
            hot_cache_gb=0,
            prefill_buffer_tokens=1,
            prefill_read_pages=1,
            max_pages=1,
        )
    assert readers[0].closed


def test_prefill_pipeline_failure_disables_lookahead_and_decode_continues(
    tmp_path, monkeypatch, caplog
):
    image = disk.build_test_image(tmp_path, _fp8_rows(50))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(
        image,
        hot_cache_gb=0,
        prefill_buffer_tokens=1,
        prefill_read_pages=1,
        max_pages=1,
    )
    fetcher.prefill_reader.locked_pages = lambda page_ids: (_ for _ in ()).throw(
        OSError(errno.EIO, "injected prefill failure")
    )
    try:
        assert fetcher.submit_prefill(np.array([1], dtype=np.int64))
        fetcher.wait_prefill()
        actual = fetcher.fetch(np.array([26], dtype=np.int64))
        assert torch.equal(actual[0], _fp8_rows(50).view(torch.uint8)[26])
        assert fetcher.prefill_reader is None
        assert "prefill look-ahead failed" in caplog.text
    finally:
        fetcher.close()


def test_prefill_dynamic_hits_do_not_enter_the_admission_queue(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    image = disk.build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(
        image,
        hot_cache_gb=0,
        dynamic_capacity_rows=8,
        prefill_buffer_tokens=1,
        prefill_read_pages=1,
        max_pages=1,
    )
    try:
        fetcher.fetch(np.array([3], dtype=np.int64))
        fetcher.dynamic.flush()
        queued = []
        monkeypatch.setattr(
            fetcher.dynamic,
            "record",
            lambda ids, exact_rows: queued.append((ids.copy(), exact_rows)),
        )

        actual = fetcher.fetch(
            np.array([3], dtype=np.int64),
            priority="prefill",
            use_prefill=False,
            admit_dynamic=False,
        )

        assert torch.equal(actual[0], rows.view(torch.uint8)[3])
        assert fetcher.last_fetch_stats.dynamic_hits == 1
        assert queued == []
    finally:
        fetcher.close()


def test_prefill_disable_waits_for_the_teardown_owner(monkeypatch):
    entered_close = threading.Event()
    release_close = threading.Event()
    waiter_done = threading.Event()
    executor_calls = []

    class BlockingReader:
        def close(self):
            entered_close.set()
            assert release_close.wait(1.0)

    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    fetcher._prefill_lock = threading.Lock()
    fetcher._prefill_disabled = False
    fetcher._prefill_disable_done = threading.Event()
    fetcher._prefill_slots = [{"state": "ready", "count": 1}]
    fetcher.prefill_reader = BlockingReader()
    fetcher._prefill_executor = SimpleNamespace(
        shutdown=lambda wait: executor_calls.append(wait)
    )
    monkeypatch.setattr(disk.logger, "error", lambda *args, **kwargs: None)

    error = OSError(errno.EIO, "injected prefill failure")
    owner = threading.Thread(target=fetcher._disable_prefill, args=(error,))
    owner.start()
    assert entered_close.wait(1.0)

    waiter = threading.Thread(
        target=lambda: (fetcher._disable_prefill(error), waiter_done.set())
    )
    waiter.start()
    assert not waiter_done.wait(0.05)

    release_close.set()
    owner.join(1.0)
    waiter.join(1.0)
    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert waiter_done.is_set()
    assert executor_calls == [False]


def test_prefill_submission_uses_the_executor_selected_with_its_slot():
    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    future = Future()
    submissions = []
    executor = SimpleNamespace(
        submit=lambda fn, *args: submissions.append((fn, args)) or future
    )
    fetcher.image = SimpleNamespace(vocab_start=0, vocab_end=10)
    fetcher._prefill_disabled = False
    fetcher._prefill_max_rows = 4
    fetcher._prefill_truncated_submissions = 0
    fetcher._prefill_sequence = 0
    fetcher._prefill_slots = [{"state": "empty", "sequence": 0, "count": 0}]
    fetcher._prefill_executor = executor
    fetcher._prefill_futures = set()

    class DisableAfterSelection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            fetcher._prefill_disabled = True
            fetcher._prefill_executor = None
            fetcher._prefill_slots = []

    fetcher._prefill_lock = DisableAfterSelection()

    assert fetcher.submit_prefill(np.array([1], dtype=np.int64))
    assert len(submissions) == 1
    assert future in fetcher._prefill_futures


def test_prefill_truncation_keeps_the_earliest_requested_rows(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(
        image,
        hot_cache_gb=0,
        prefill_buffer_tokens=1,
        prefill_read_pages=1,
        max_pages=1,
    )
    requested = np.arange(20, 0, -1, dtype=np.int64)
    try:
        assert fetcher.submit_prefill(requested)
        fetcher.wait_prefill()
        ready = next(
            slot for slot in fetcher._prefill_slots if slot["state"] == "ready"
        )
        assert ready["ids"][: ready["count"]].tolist() == list(range(5, 21))
    finally:
        fetcher.close()


def test_numpy_hash_matches_production_on_seeded_token_stream(monkeypatch):
    metadata = PLEMetadata(
        multipliers=np.array([1000003, 1000033, 1000037], dtype=np.int64),
        vocab_sizes=np.array([101 + 2 * index for index in range(16)], dtype=np.int64),
        offsets=np.cumsum(
            np.r_[np.int64(0), np.array([101 + 2 * index for index in range(15)])]
        ),
        eos_token_id=2,
    )
    rng = np.random.default_rng(20260830)
    tokens = rng.integers(3, 32000, size=4096, dtype=np.int64)
    tokens[::127] = 2
    tokens[1::509] = 2
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.enable_ple_fusion = False
    module.ngram_size = 3
    module.heads_per_ngram = 8
    module.ngram_heads = 16
    module.eos_token_id = metadata.eos_token_id
    module.layer_multipliers = torch.from_numpy(metadata.multipliers)
    module.ngram_heads_vocab_sizes = torch.from_numpy(metadata.vocab_sizes)
    module.ngram_heads_offsets = torch.from_numpy(metadata.offsets)
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_req_to_token_pool",
        lambda: SimpleNamespace(ple_window_cache=None),
    )

    contexts = np.full((tokens.size, 3), metadata.eos_token_id, dtype=np.int64)
    contexts[:, 2] = tokens
    contexts[1:, 1] = tokens[:-1]
    contexts[2:, 0] = tokens[:-2]
    actual = module._hash_contexts(torch.from_numpy(contexts)).numpy()
    expected = hash_contexts_numpy(contexts, metadata)
    assert np.array_equal(actual, expected)
    assert np.array_equal(hash_token_stream_numpy(tokens, metadata), expected)


def test_hit_sim_selects_accessed_rows_and_splits_tp_ranks(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[5] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_test", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)

    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([4, 3], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        eos_token_id=2,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "multipliers": metadata.multipliers.tolist(),
                "vocab_sizes": metadata.vocab_sizes.tolist(),
                "offsets": metadata.offsets.tolist(),
                "eos_token_id": metadata.eos_token_id,
                "ngram_size": metadata.ngram_size,
            }
        )
    )
    tokens = np.array([1, 2, 3, 1, 0, 2], dtype="<i4")
    token_path = tmp_path / "tokens.i32"
    tokens.tofile(token_path)
    count_dir = tmp_path / "counts"
    output_path = tmp_path / "hot.bin"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--tokens",
            str(token_path),
            "--metadata",
            str(metadata_path),
            "--work-dir",
            str(count_dir),
            "--output",
            str(output_path),
            "--fingerprint",
            "test-image",
            "--tp-size",
            "2",
            "--budget-gib",
            "0.00001",
            "--padding-divisor",
            "4",
            "--chunk-tokens",
            "2",
        ],
    )

    hit_sim.main()

    count_files = [
        np.memmap(
            count_dir / f"head{head:02d}.u64",
            mode="r",
            dtype=np.uint64,
            shape=(int(size),),
        )
        for head, size in enumerate(metadata.vocab_sizes)
    ]
    assert all(array.dtype == np.uint64 for array in count_files)
    ids, frequencies = hit_sim.select_rows(count_files, metadata, capacity=10)
    ranks = hit_sim.split_ranks(ids, frequencies, total_rows=7, tp_size=2, divisor=4)
    for rank, expected in ranks.items():
        assert np.array_equal(
            disk.read_hot_frequency_file(
                output_path, rank, expected_fingerprint="test-image"
            ),
            expected,
        )


def test_image_builder_reserves_space_for_all_tp_ranks(tmp_path, monkeypatch):
    per_rank = 25 * disk.ROW_BYTES + 2 * disk.PAGE_BYTES
    monkeypatch.setattr(
        disk.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=per_rank + 1),
    )
    with pytest.raises(OSError, match="all 2 tensor-parallel ranks"):
        disk.PLEImageBuilder(tmp_path, "space", 0, 2, 0, 25)


def test_image_builder_reserves_space_for_all_ple_layers(tmp_path, monkeypatch):
    per_image = 25 * disk.ROW_BYTES + 2 * disk.PAGE_BYTES
    monkeypatch.setattr(
        disk.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=2 * per_image + 1),
    )
    with pytest.raises(OSError, match="3 PLE layers"):
        disk.PLEImageBuilder(
            tmp_path,
            "space-layers",
            0,
            1,
            0,
            25,
            image_count=3,
        )


def test_config_digest_rejects_non_json_values():
    with pytest.raises(ValueError, match="not JSON-serializable"):
        disk.config_digest({"opaque": object()})


def test_prefill_priority_requires_its_own_reader(tmp_path, monkeypatch):
    image = disk.build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(image, hot_cache_gb=0, max_pages=1)
    try:
        with pytest.raises(RuntimeError, match="prefill reader"):
            fetcher.fetch(np.array([0]), priority="prefill", use_prefill=False)
    finally:
        fetcher.close()


def test_dynamic_cache_reports_a_full_admission_queue(monkeypatch, caplog):
    monkeypatch.setattr(
        threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    cache = disk.WTinyLFURowCache(capacity_rows=8, queue_batches=1)
    cache._queue.put_nowait(object())
    with caplog.at_level("WARNING"):
        cache.record(np.array([1]), np.zeros((1, disk.ROW_BYTES), dtype=np.uint8))
    assert "dropped 1 admission batches" in caplog.text


def test_dynamic_cache_lookup_does_not_wait_for_admission_batch_lock():
    cache = disk.WTinyLFURowCache(capacity_rows=8)
    row = np.arange(disk.ROW_BYTES, dtype=np.uint8)
    cache._insert(3, row)
    output = np.zeros((1, disk.ROW_BYTES), dtype=np.uint8)
    finished = threading.Event()

    def lookup():
        hit = cache.lookup_into(
            np.array([3], dtype=np.int64), output, record_hits=False
        )
        assert hit.tolist() == [True]
        finished.set()

    cache._lock.acquire()
    thread = threading.Thread(target=lookup)
    thread.start()
    try:
        assert finished.wait(1.0)
    finally:
        cache._lock.release()
        thread.join(timeout=1.0)
        cache.close()
    assert np.array_equal(output[0], row)


def test_dynamic_cache_lookup_bounds_retries_on_a_stalled_set(monkeypatch):
    cache = disk.WTinyLFURowCache(capacity_rows=8)
    output = np.full((1, disk.ROW_BYTES), 0xA5, dtype=np.uint8)
    set_index = int(cache._set_indices(np.array([3], dtype=np.int64))[0])
    cache._versions[set_index] = np.uint64(1)
    yields = []
    monkeypatch.setattr(disk.time, "sleep", lambda seconds: yields.append(seconds))
    try:
        hit = cache.lookup_into(
            np.array([3], dtype=np.int64), output, record_hits=False
        )
    finally:
        cache._versions[set_index] = np.uint64(2)
        cache.close()

    assert hit.tolist() == [False]
    assert np.all(output == 0xA5)
    assert len(yields) == cache._LOOKUP_MAX_RETRIES - 1


def test_dynamic_cache_worker_failure_is_reported_and_close_is_bounded(
    monkeypatch,
):
    real_empty = torch.empty

    def cpu_empty(*shape, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*shape, **kwargs)

    monkeypatch.setattr(disk.torch, "empty", cpu_empty)
    cache = disk.WTinyLFURowCache(capacity_rows=8)
    monkeypatch.setattr(
        cache,
        "_insert",
        lambda row_id, exact_row: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    cache.record(np.array([1]), np.zeros((1, disk.ROW_BYTES), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="admission failed"):
        cache.flush(timeout=1.0)
    with pytest.raises(RuntimeError, match="admission failed"):
        cache.close()
    assert not cache._worker.is_alive()


def test_disk_embedding_close_releases_fetcher_builder_and_executor():
    closed = []

    class Closeable:
        def close(self):
            closed.append(type(self).__name__)

    class Executor:
        def shutdown(self, wait=True):
            closed.append("Executor")

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._future = None
    embedding._prefill_submit_future = None
    embedding._fetcher = Closeable()
    embedding._image_builder = Closeable()
    embedding._executor = Executor()
    embedding._transfer_buffers = {"device": object()}

    embedding.close()
    embedding.close()
    assert closed == ["Closeable", "Closeable", "Executor"]
    assert embedding._transfer_buffers == {}


def test_weight_reload_without_ple_rows_keeps_the_live_image():
    class Closeable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Executor:
        def __init__(self):
            self.closed = False

        def shutdown(self, wait=True):
            self.closed = True

    fetcher = Closeable()
    executor = Executor()
    image = object()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = fetcher
    embedding._image = image
    embedding._image_builder = None
    embedding._executor = executor
    embedding._future = None
    embedding._prefill_submit_future = None
    embedding._transfer_buffers = {}
    embedding._active_transfer_device = None
    embedding._prefill_host_ids = None
    embedding._builder_args = {}

    embedding.prepare_weight_reload()
    embedding.finalize_image()

    assert embedding._fetcher is fetcher
    assert embedding._image is image
    assert embedding._executor is executor
    assert not fetcher.closed
    assert not executor.closed


def test_weight_reload_tears_down_when_the_first_ple_row_arrives(monkeypatch):
    created = []

    class Builder:
        def __init__(self, **kwargs):
            created.append(self)
            self.shards = []

        def add_shard(self, *args):
            self.shards.append(args)

        def close(self):
            pass

    class Closeable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Executor:
        def __init__(self):
            self.closed = False

        def shutdown(self, wait=True):
            self.closed = True

    monkeypatch.setattr(disk, "PLEImageBuilder", Builder)
    old_fetcher = Closeable()
    old_executor = Executor()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = old_fetcher
    embedding._image = object()
    embedding._image_builder = None
    embedding._executor = old_executor
    embedding._future = None
    embedding._prefill_submit_future = None
    embedding._transfer_buffers = {}
    embedding._active_transfer_device = None
    embedding._prefill_host_ids = None
    embedding._builder_args = {"root": "unused"}
    embedding._new_executor = lambda: Executor()

    embedding.prepare_weight_reload()
    assert not old_fetcher.closed
    embedding.add_checkpoint_shard("shard_0.weight", _fp8_rows(1), 0, 1)

    assert old_fetcher.closed
    assert old_executor.closed
    assert embedding._image is None
    assert len(created) == 1
    assert created[0].shards[0][0] == "shard_0.weight"


def test_future_contexts_follow_configured_ngram_size():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._prefill_buffer_tokens = 2
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._future_lookup_contexts = None
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 4
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([layer])
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.arange(3),
    )
    request = SimpleNamespace(
        origin_input_ids=[10, 11, 12, 13, 14, 15],
        extend_range=SimpleNamespace(end=3),
    )
    batch = SimpleNamespace(reqs=[request])
    model.prepare_model_batch(batch, forward_batch)
    assert torch.equal(
        layer._future_lookup_contexts,
        torch.tensor([[10, 11, 12, 13], [11, 12, 13, 14]]),
    )


def test_model_resume_storage_reopens_every_disk_embedding():
    resumed = []

    def make_layer(name):
        embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.resume_storage = lambda: resumed.append(name)
        return SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=embedding))

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model._ple_layers = lambda: iter([make_layer("first"), make_layer("second")])

    model.resume_storage()

    assert resumed == ["first", "second"]


def test_manifest_rejects_noncontiguous_ranges(tmp_path):
    rows = _fp8_rows(100)
    image = disk.build_test_image(
        tmp_path, rows, config_sha256="manifest-gap", weight_scale=0.5
    )
    manifest_path = image.path.parent / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["shards"][0]["row_start"] = 1
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="contiguous.*rebuild"):
        disk.open_ple_image(image.path)
    with pytest.raises(ValueError, match="delete.*rebuild"):
        disk.PLEImageBuilder(tmp_path, "manifest-gap", 0, 1, 0, 100)


def test_old_image_format_requests_rebuild(tmp_path):
    image = disk.build_test_image(tmp_path, _fp8_rows(25), config_sha256="old")
    with image.path.open("r+b") as handle:
        block = handle.read(disk.PAGE_BYTES)
        magic, length = struct.unpack_from("<8sI", block)
        header = json.loads(block[12 : 12 + length])
        header["format_version"] = disk.FORMAT_VERSION - 1
        handle.seek(0)
        handle.write(disk._write_metadata_page(magic, header))
    with pytest.raises(ValueError, match="rebuild"):
        disk.PLEImageBuilder(tmp_path, "old", 0, 1, 0, 25)


def test_weight_scale_stale_image_is_still_rejected(tmp_path):
    rows = _fp8_rows(25)
    disk.build_test_image(tmp_path, rows, config_sha256="scale", weight_scale=0.25)
    builder = disk.PLEImageBuilder(tmp_path, "scale", 0, 1, 0, 25)
    builder.add_shard("test.shard_0.weight", rows, 0, 25)
    with pytest.raises(ValueError, match="weight_scale mismatch.*delete"):
        builder.finalize(0.5)


def test_sampled_payload_change_rejects_same_shape_reuse(tmp_path):
    rows = _fp8_rows(25)
    disk.build_test_image(tmp_path, rows, config_sha256="payload", weight_scale=0.5)
    changed = rows.view(torch.uint8).clone()
    changed[12, 7] ^= 1
    builder = disk.PLEImageBuilder(tmp_path, "payload", 0, 1, 0, 25)
    builder.add_shard("test.shard_0.weight", changed.view(torch.float8_e4m3fn), 0, 25)
    with pytest.raises(ValueError, match="manifest mismatch.*delete"):
        builder.finalize(0.5)


def test_unregistered_staging_reads_with_the_same_pointer(tmp_path, monkeypatch):
    raw = _fp8_rows(50).view(torch.uint8)
    image = disk.build_test_image(tmp_path, raw.view(torch.float8_e4m3fn))
    library = _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=2)
    try:
        pages = reader.read(np.array([0, 1], dtype=np.int64))
        assert library.create_args[4] is False
        assert library.create_args[1] == reader.staging.data_ptr()
        assert library.read_buffer == reader.staging.data_ptr()
        assert np.array_equal(
            pages[0, : 25 * disk.ROW_BYTES], raw[:25].numpy().reshape(-1)
        )
        assert np.array_equal(
            pages[1, : 25 * disk.ROW_BYTES], raw[25:].numpy().reshape(-1)
        )
    finally:
        reader.close()


def test_direct_reader_results_survive_the_next_read(tmp_path, monkeypatch):
    raw = _fp8_rows(50).view(torch.uint8)
    image = disk.build_test_image(tmp_path, raw.view(torch.float8_e4m3fn))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        first = reader.read(np.array([0], dtype=np.int64))
        saved = first.copy()
        reader.read(np.array([1], dtype=np.int64))
        assert np.array_equal(first, saved)
    finally:
        reader.close()


def test_gather_rejects_missing_fetcher_before_forward_work():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    embedding._fetcher = None
    with pytest.raises(RuntimeError, match="not finalized after weight loading"):
        embedding.gather(torch.tensor([0], dtype=torch.long))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
