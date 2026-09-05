import ast
import ctypes
import errno
import gc
import importlib.metadata
import importlib.util
import json
import os
import queue
import runpy
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch
from packaging.version import Version

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models import qwen4_ple_disk as disk
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpDiskEmbedding,
    Qwen4ExpModel,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    _prepare_ple_batch,
)
from sglang.srt.utils.ple_disk import IORING_MAX_ENTRIES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.ple_disk_utils import (
    attach_checkpoint_source,
    build_test_image,
    fetcher_header_constants,
    native_reader_unavailable_reason,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_REAL_ALLOCATE_HOST_TENSOR = disk._allocate_host_tensor
_FETCHER_HEADER_CONSTANTS = fetcher_header_constants()


@pytest.fixture(autouse=True)
def replace_pinned_allocators_without_cuda(monkeypatch):
    if torch.cuda.is_available():
        return

    def cpu_host_tensor(*args, pin_memory=True, **kwargs):
        return torch.empty(*args, **kwargs)

    monkeypatch.setattr(disk, "_allocate_host_tensor", cpu_host_tensor)
    monkeypatch.setattr(qwen4_exp_module, "_allocate_host_tensor", cpu_host_tensor)


def test_host_tensor_allocator_pins_the_cpu_device(monkeypatch):
    call = {}

    def record_empty(*args, **kwargs):
        call.update(kwargs)
        return object()

    monkeypatch.setattr(torch, "empty", record_empty)
    _REAL_ALLOCATE_HOST_TENSOR(1, pin_memory=False)
    assert call["device"] == "cpu"


def test_sgl_kernel_version_survives_missing_distribution_metadata(monkeypatch):
    version_file = (
        Path(__file__).parents[4]
        / "python/sglang/kernels/aot/python/sgl_kernel/version.py"
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError),
    )
    required = Version(disk.MIN_SGL_KERNEL_VERSION_FOR_PLE_DISK)
    fallback = Version(runpy.run_path(version_file)["__version__"])
    assert disk._installed_sgl_kernel_version() is None
    assert fallback >= required


def test_metadata_page_reader_accepts_short_reads():
    block = disk._write_metadata_page(disk.HOT_MAGIC, {"value": 17})

    class ShortReader:
        def __init__(self):
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            size = min(size, 37)
            chunk = block[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    fake_path = SimpleNamespace(
        parent=Path("/tmp/metadata-parent"),
        open=lambda *args, **kwargs: ShortReader(),
    )
    assert disk._read_metadata_page(fake_path, disk.HOT_MAGIC) == {"value": 17}


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


@pytest.mark.parametrize("storage", ["gpu", "pinned"])
def test_non_disk_target_verify_keeps_base_valid_tokens(monkeypatch, storage):
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    forward_batch = SimpleNamespace(
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=SimpleNamespace(topk=1, draft_token_num=4),
        forward_mode=ForwardMode.TARGET_VERIFY,
        _original_forward_mode=None,
        extend_seq_lens=torch.tensor([3], dtype=torch.int32),
        out_cache_loc=torch.tensor([7, 0, 9, 0], dtype=torch.long),
        req_pool_indices=torch.tensor([1], dtype=torch.long),
        num_token_non_padded_cpu=4,
    )

    batch = _prepare_ple_batch(
        torch.arange(4),
        forward_batch,
        ngram_size=None,
        ngram_eos_token_id=None,
        mask_invalid_tokens=storage == "disk",
    )

    assert batch.valid_tokens.tolist() == [True, True, True, False]


def test_disk_target_verify_masks_invalid_graph_slots(monkeypatch):
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    forward_batch = SimpleNamespace(
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=SimpleNamespace(topk=1, draft_token_num=4),
        forward_mode=ForwardMode.TARGET_VERIFY,
        _original_forward_mode=None,
        extend_seq_lens=torch.tensor([3], dtype=torch.int32),
        out_cache_loc=torch.tensor([7, 0, 9, 0], dtype=torch.long),
        req_pool_indices=torch.tensor([1], dtype=torch.long),
        num_token_non_padded_cpu=4,
    )

    batch = _prepare_ple_batch(
        torch.arange(4),
        forward_batch,
        ngram_size=None,
        ngram_eos_token_id=None,
        mask_invalid_tokens=True,
    )

    assert batch.valid_tokens.tolist() == [True, False, True, False]


def test_disk_decode_masks_padding_when_fusion_is_disabled(monkeypatch):
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    monkeypatch.setattr(
        qwen4_exp_module.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION,
        "get",
        lambda: False,
    )
    forward_batch = SimpleNamespace(
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        forward_mode=ForwardMode.DECODE,
        _original_forward_mode=None,
        extend_seq_lens=None,
        out_cache_loc=torch.tensor([7, 0], dtype=torch.long),
        req_pool_indices=torch.tensor([1, 2], dtype=torch.long),
        num_token_non_padded_cpu=2,
    )

    batch = _prepare_ple_batch(
        torch.arange(2),
        forward_batch,
        ngram_size=None,
        ngram_eos_token_id=None,
        mask_invalid_tokens=True,
    )

    assert batch.valid_tokens.tolist() == [True, False]


def test_disk_padding_slot_uses_each_real_allocator_contract():
    from sglang.srt.mem_cache.allocator.base import allocator_reserves_padding_slot
    from sglang.srt.mem_cache.allocator.hisparse import (
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
        HiSparseTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.multi_ended_allocator import (
        MultiEndedAllocator,
        UnifiedMambaTokenToKVPoolAllocator,
        UnifiedSWATokenToKVPoolAllocator,
    )

    token = TokenToKVPoolAllocator(8, torch.bfloat16, "cpu", None, False)
    paged = PagedTokenToKVPoolAllocator(8, 2, torch.bfloat16, "cpu", None, False)
    multi = MultiEndedAllocator.__new__(MultiEndedAllocator)
    multi.page_size = 2
    multi.num_pages = 4
    multi.virtual_to_physical = torch.tensor([0, -1, -1, -1])
    multi.free_virtual_ids = torch.tensor([1, 2, 3])
    multi._peer = None

    swa = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    swa.full_attn_allocator = token
    hisparse = HiSparseTokenToKVPoolAllocator.__new__(HiSparseTokenToKVPoolAllocator)
    hisparse.logical_attn_allocator = paged
    deepseek_hisparse = DeepSeekV4HiSparseTokenToKVPoolAllocator.__new__(
        DeepSeekV4HiSparseTokenToKVPoolAllocator
    )
    deepseek_hisparse.logical_attn_allocator = paged
    unified_mamba = UnifiedMambaTokenToKVPoolAllocator.__new__(
        UnifiedMambaTokenToKVPoolAllocator
    )
    unified_mamba.full_attn_allocator = multi
    unified_swa = UnifiedSWATokenToKVPoolAllocator.__new__(
        UnifiedSWATokenToKVPoolAllocator
    )
    unified_swa.full_attn_allocator = multi

    for allocator in (
        token,
        paged,
        multi,
        swa,
        hisparse,
        deepseek_hisparse,
        unified_mamba,
        unified_swa,
    ):
        assert allocator_reserves_padding_slot(allocator)

    token.free_pages = torch.tensor([0, 1])
    assert not allocator_reserves_padding_slot(token)
    assert multi.is_slot_allocated(0)
    multi.free_virtual_ids = None
    multi._peer = SimpleNamespace(free_virtual_ids=torch.tensor([1, 2, 3]))
    assert allocator_reserves_padding_slot(multi)
    multi._peer.free_virtual_ids = torch.tensor([0, 1, 2, 3])
    assert not allocator_reserves_padding_slot(multi)
    multi._peer = None
    multi.free_virtual_ids = torch.tensor([0, 1, 2, 3])
    assert not allocator_reserves_padding_slot(multi)


def test_missing_padding_slot_contract_names_the_allocator_and_disk_flag():
    class MissingAllocator:
        pass

    with pytest.raises(
        NotImplementedError,
        match=r"MissingAllocator\.reserves_padding_slot.*--ple-storage disk",
    ):
        BaseTokenToKVPoolAllocator.reserves_padding_slot(MissingAllocator())


def test_layer_multipliers_follow_the_default_device_context():
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.unigram_vocab_size = 32
    module.config = SimpleNamespace(seed=1234)
    module.ple_layer_index = 0

    with torch.device("meta"):
        multipliers = module._build_layer_multipliers(3)
        offsets = torch.tensor([0, 11], dtype=torch.long)

    assert multipliers.device == offsets.device
    assert multipliers.dtype == torch.long


def test_fused_hash_capability_accepts_cuda_resident_contract():
    from sglang.kernels.ops.qwen4_ple import can_fuse_qwen4_ngram_hash

    contexts = torch.zeros((2, 3), dtype=torch.long)
    multipliers = torch.ones(3, dtype=torch.long)
    vocab_sizes = torch.ones(16, dtype=torch.long)
    offsets = torch.zeros(16, dtype=torch.long)
    with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda:
        is_cuda.return_value = True
        assert can_fuse_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets)


def test_checkpoint_ple_offload_embedding_maps_to_storage_with_warning():
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig

    for legacy, expected in ((True, "pinned"), (False, "gpu")):
        with pytest.warns(FutureWarning, match="--ple-storage"):
            config = Qwen4ExpTextConfig(ple_offload_embedding=legacy)
        assert config.ple_storage == expected


@pytest.mark.parametrize(("legacy", "expected"), [(True, "pinned"), (False, "gpu")])
def test_engine_legacy_ple_keyword_translates_from_json_dict(legacy, expected):
    from sglang.srt.entrypoints.engine import _translate_legacy_ple_storage_kwargs

    kwargs = json.loads(json.dumps({"ple_offload_embedding": legacy}))
    with pytest.warns(FutureWarning, match="ple_storage"):
        _translate_legacy_ple_storage_kwargs(kwargs)
    assert kwargs == {"ple_storage": expected}


def test_engine_legacy_ple_keyword_rejects_explicit_conflict():
    from sglang.srt.entrypoints.engine import _translate_legacy_ple_storage_kwargs

    kwargs = {"ple_offload_embedding": True, "ple_storage": "gpu"}
    with pytest.raises(ValueError, match="ple_offload_embedding.*ple_storage"):
        _translate_legacy_ple_storage_kwargs(kwargs)


def test_engine_none_legacy_ple_keyword_leaves_resolution_unset():
    from sglang.srt.entrypoints.engine import _translate_legacy_ple_storage_kwargs

    kwargs = {"ple_offload_embedding": None}
    _translate_legacy_ple_storage_kwargs(kwargs)
    assert kwargs == {}


def test_qwen4_model_has_no_top_level_disk_engine_import():
    model_path = Path(disk.__file__).with_name("qwen4_exp.py")
    tree = ast.parse(model_path.read_text())
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "sglang.srt.models.qwen4_ple_disk" not in imported_modules


def test_checkpoint_config_uses_server_disk_defaults():
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
    from sglang.srt.server_args import ServerArgs

    config = Qwen4ExpTextConfig()

    assert config.ple_disk_hot_cache_gb == ServerArgs.ple_disk_hot_cache_gb
    assert config.ple_disk_dynamic_cache_gb == ServerArgs.ple_disk_dynamic_cache_gb
    assert (
        config.ple_disk_prefill_buffer_tokens
        == ServerArgs.ple_disk_prefill_buffer_tokens
    )
    assert config.ple_disk_prefill_read_pages == ServerArgs.ple_disk_prefill_read_pages
    assert config.ple_disk_max_prefill_chunk_tokens == 0


def _fp8_rows(count: int) -> torch.Tensor:
    raw = torch.arange(count * disk.ROW_BYTES, dtype=torch.uint8).reshape(
        count, disk.ROW_BYTES
    )
    raw[(raw & 0x7F) == 0x7F] = 0
    return attach_checkpoint_source(raw.view(torch.float8_e4m3fn))


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
        failure_stage=None,
        read_results=(),
        last_error=None,
        abi_version=None,
    ):
        self.image_bytes = image_bytes
        self.create_errno = create_errno
        self.failure_stage = (
            _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_FAILURE_NONE"]
            if failure_stage is None
            else failure_stage
        )
        self.read_results = iter(read_results)
        self.last_error = last_error
        self.abi_version = (
            _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_ABI_VERSION"]
            if abi_version is None
            else abi_version
        )
        self.create_args = None
        self.created_buffer = None
        self.created_buffer_bytes = 0
        self.max_pages = 0
        self.read_buffer = None
        self.read_buffer_bytes = None
        self.ple_fetcher_create = _FakeFunction(self._create)
        self.ple_fetcher_abi_version = _FakeFunction(lambda: self.abi_version)
        self.ple_fetcher_lock_budget_ms = _FakeFunction(
            lambda: _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_LOCK_BUDGET_MS"]
        )
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
        if (
            file_fd < 0
            or not buffer
            or int(buffer) & (disk.PAGE_BYTES - 1)
            or max_pages <= 0
            or buffer_bytes < max_pages * disk.PAGE_BYTES
        ):
            ctypes.set_errno(errno.EINVAL)
            return None
        if self.create_errno:
            ctypes.set_errno(self.create_errno)
            return None
        self.created_buffer = int(buffer)
        self.created_buffer_bytes = int(buffer_bytes)
        self.max_pages = int(max_pages)
        return 1

    def _read(self, handle, offsets, count, buffer, buffer_bytes):
        self.read_buffer = buffer
        self.read_buffer_bytes = buffer_bytes
        if (
            handle != 1
            or not buffer
            or int(buffer) & (disk.PAGE_BYTES - 1)
            or count > self.max_pages
            or (count and not offsets)
        ):
            return -errno.EINVAL
        for index in range(count):
            if offsets[index] & (disk.PAGE_BYTES - 1):
                return -errno.EINVAL
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
    monkeypatch.setattr(disk, "_logical_block_size", lambda path: disk.PAGE_BYTES)
    monkeypatch.setattr(
        disk, "_open_direct_file", lambda path: os.open(path, os.O_RDONLY)
    )
    return library


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
    (package_dir / "qwen4_ple_disk_fetcher.build").write_text("enabled\n")
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post1")
    assert disk._find_helper_library() == helper


def test_missing_helper_names_the_required_sgl_kernel_version(monkeypatch):
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="sgl_kernel is not importable"):
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
    (package_dir / "qwen4_ple_disk_fetcher.build").write_text("enabled\n")
    assert disk._find_helper_library() == helper


def test_helper_rejects_an_absent_build_marker(tmp_path, monkeypatch):
    package_dir = tmp_path / "sgl_kernel"
    package_dir.mkdir()
    (package_dir / "qwen4_ple_disk_fetcher.so").touch()
    spec = SimpleNamespace(submodule_search_locations=[str(package_dir)])
    monkeypatch.setattr(disk.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(disk, "_installed_sgl_kernel_version", lambda: "0.4.6.post2")

    with pytest.raises(RuntimeError, match="built without.*io_uring"):
        disk._find_helper_library()


def test_poisoned_fetcher_error_requires_restart(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
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


def test_poisoned_reader_retains_staging_when_native_drain_expires(
    tmp_path, monkeypatch, caplog
):
    image = build_test_image(tmp_path, _fp8_rows(25))
    library = _patch_fetcher_library(
        monkeypatch,
        image,
        read_results=(-disk.FETCHER_ERR_POISONED,),
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    retained = []
    monkeypatch.setattr(disk, "_RETAINED_POISONED_STAGING", retained)
    reader = disk.DirectPageReader(image, max_pages=1)
    fd = reader.fd
    staging = weakref.ref(reader._staging_allocation)
    library.ple_fetcher_destroy = _FakeFunction(lambda handle: -errno.ETIMEDOUT)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="poisoned"):
        reader.read(np.array([0], dtype=np.int64))
    assert retained == []

    with caplog.at_level("ERROR"), pytest.raises(OSError, match="shutdown failed"):
        reader.close()
    assert "retained 8191 staging bytes" in caplog.text
    with pytest.raises(OSError) as exc_info:
        os.fstat(fd)
    assert exc_info.value.errno == errno.EBADF
    assert reader.handle is None

    del reader
    gc.collect()
    assert staging() is not None
    assert retained == [staging()]


def test_fetcher_error_reports_page_and_short_read_size(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
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
    image = build_test_image(tmp_path, rows)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    try:
        reader = disk.DirectPageReader(image, max_pages=2)
    except (OSError, RuntimeError) as exc:
        reason = native_reader_unavailable_reason(exc)
        if reason is None:
            raise
        pytest.skip(reason)

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


@pytest.mark.parametrize("max_pages", [0, IORING_MAX_ENTRIES + 1])
def test_direct_reader_rejects_invalid_io_uring_entry_count(tmp_path, max_pages):
    image = build_test_image(tmp_path, _fp8_rows(1))
    with pytest.raises(ValueError, match="ple-disk-max-read-pages must be between"):
        disk.DirectPageReader(image, max_pages=max_pages)


def test_resolved_page_limit_reaches_fetcher_registration(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(50))
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
    image = build_test_image(tmp_path, _fp8_rows(50))
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


def test_locked_pages_returns_the_crc_checked_staging_view(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        with reader.locked_pages(np.array([0], dtype=np.int64)) as pages:
            assert np.shares_memory(pages, reader.staging.numpy())
    finally:
        reader.close()


def test_direct_reader_waits_for_a_concurrent_caller_with_a_budget(monkeypatch):
    reader = disk.DirectPageReader.__new__(disk.DirectPageReader)
    reader._read_lock = threading.Lock()
    reader._read_lock_timeout_seconds = 0.01
    reader.max_pages = 1
    reader._read_lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="timed out waiting"):
            with reader.locked_pages(np.array([], dtype=np.int64)):
                pass
    finally:
        reader._read_lock.release()


def test_direct_reader_rejects_a_fetcher_abi_mismatch(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image, abi_version=0)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )

    with pytest.raises(RuntimeError, match="ABI mismatch: expected 1, found 0"):
        disk.DirectPageReader(image, max_pages=1)


def test_installed_fetcher_create_rejects_an_invalid_buffer():
    spec = importlib.util.find_spec("sgl_kernel")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    helper = next(
        (
            Path(location) / "qwen4_ple_disk_fetcher.so"
            for location in locations
            if (Path(location) / "qwen4_ple_disk_fetcher.so").is_file()
        ),
        None,
    )
    if helper is None:
        pytest.skip("installed wheel has no qwen4_ple_disk_fetcher.so")

    library = ctypes.CDLL(str(helper), use_errno=True)
    library.ple_fetcher_create.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.ple_fetcher_create.restype = ctypes.c_void_p
    failure_stage = ctypes.c_int(-1)
    ctypes.set_errno(0)
    handle = library.ple_fetcher_create(-1, None, 0, 1, 0, ctypes.byref(failure_stage))
    assert not handle
    assert ctypes.get_errno() == errno.EINVAL
    assert failure_stage.value == 0


def test_installed_fetcher_abi_matches_python_constant():
    installed = disk._installed_sgl_kernel_version()
    if installed is None or Version(installed) < Version(
        disk.MIN_SGL_KERNEL_VERSION_FOR_PLE_DISK
    ):
        pytest.skip("the pinned sglang-kernel wheel is not installed")

    library = disk._load_helper_library()

    assert library.ple_fetcher_abi_version() == disk.PLE_FETCHER_ABI_VERSION


def test_fake_fetcher_constants_come_from_the_native_header():
    assert (
        disk.PLE_FETCHER_ABI_VERSION
        == _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_ABI_VERSION"]
    )
    assert (
        disk.FETCHER_FAILURE_SETUP
        == _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_FAILURE_SETUP"]
    )
    assert (
        disk.FETCHER_FAILURE_REGISTER_BUFFER
        == _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_FAILURE_REGISTER_BUFFER"]
    )
    assert (
        disk.FETCHER_FAILURE_REGISTER_FILE
        == _FETCHER_HEADER_CONSTANTS["PLE_FETCHER_FAILURE_REGISTER_FILE"]
    )


def test_fake_fetcher_rejects_real_abi_contract_violations():
    allocation = ctypes.create_string_buffer(3 * disk.PAGE_BYTES)
    base = ctypes.addressof(allocation)
    aligned = (base + disk.PAGE_BYTES - 1) & ~(disk.PAGE_BYTES - 1)
    failure_stage = ctypes.c_int(-1)
    library = _FakeFetcherLibrary(bytes(3 * disk.PAGE_BYTES))

    assert not library.ple_fetcher_create(
        -1, aligned, 2 * disk.PAGE_BYTES, 2, 1, ctypes.byref(failure_stage)
    )
    assert ctypes.get_errno() == errno.EINVAL
    handle = library.ple_fetcher_create(
        3, aligned, 2 * disk.PAGE_BYTES, 2, 1, ctypes.byref(failure_stage)
    )
    assert handle
    offsets = (ctypes.c_uint64 * 3)(
        disk.PAGE_BYTES, 2 * disk.PAGE_BYTES, disk.PAGE_BYTES + 1
    )
    assert (
        library.ple_fetcher_read(handle, offsets, 3, aligned, 2 * disk.PAGE_BYTES)
        == -errno.EINVAL
    )
    assert (
        library.ple_fetcher_read(
            handle,
            ctypes.cast(offsets, ctypes.POINTER(ctypes.c_uint64)),
            1,
            aligned + 1,
            2 * disk.PAGE_BYTES,
        )
        == -errno.EINVAL
    )
    assert (
        library.ple_fetcher_read(
            handle,
            ctypes.pointer(ctypes.c_uint64(disk.PAGE_BYTES + 1)),
            1,
            aligned,
            2 * disk.PAGE_BYTES,
        )
        == -errno.EINVAL
    )


def test_capability_filter_does_not_skip_native_einval():
    assert native_reader_unavailable_reason(OSError(errno.EINVAL, "bad buffer")) is None


def test_memlock_error_names_limit_bytes_and_flag(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=errno.ENOMEM,
        failure_stage=disk.FETCHER_FAILURE_REGISTER_BUFFER,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: False
    )
    monkeypatch.setattr(
        disk, "_RETAINED_POISONED_STAGING", [torch.empty(13, dtype=torch.uint8)]
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
    assert "13 bytes remain in retained staging" in message


@pytest.mark.parametrize("blocked_errno", [errno.EPERM, errno.EACCES, errno.ENOSYS])
def test_blocked_io_uring_error_has_operator_actions(
    tmp_path, monkeypatch, blocked_errno
):
    image = build_test_image(tmp_path, _fp8_rows(25))
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


def test_old_kernel_io_uring_error_names_the_minimum_version(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=errno.EOPNOTSUPP,
        failure_stage=disk.FETCHER_FAILURE_SETUP,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )

    with pytest.raises(OSError, match="Linux kernel 5.11 or later"):
        disk.DirectPageReader(image, max_pages=2)


def test_registration_permission_error_is_not_reported_as_blocked_setup(
    tmp_path, monkeypatch
):
    image = build_test_image(tmp_path, _fp8_rows(25))
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


def test_file_registration_permission_error_names_seccomp(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(
        monkeypatch,
        image,
        create_errno=errno.EPERM,
        failure_stage=disk.FETCHER_FAILURE_REGISTER_FILE,
    )
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )

    with pytest.raises(OSError) as exc_info:
        disk.DirectPageReader(image, max_pages=2)

    assert "file registration" in str(exc_info.value)
    assert "seccomp" in str(exc_info.value)


def test_manifest_records_ranges_and_rejects_missing_ranges(tmp_path):
    rows = _fp8_rows(100)
    builder = disk.PLEImageBuilder(tmp_path, "ranges", 0, 1, 0, 100)
    builder.add_shard("shard_1", attach_checkpoint_source(rows[50:]), 50, 100)
    builder.add_shard("shard_0", attach_checkpoint_source(rows[:50]), 0, 50)
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


def test_manifest_install_is_read_back_and_compared(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    real_replace = disk.os.replace

    def corrupt_manifest_after_install(source, destination):
        real_replace(source, destination)
        destination = Path(destination)
        if destination.name == "manifest.json":
            destination.write_text('{"corrupt": true}\n')

    monkeypatch.setattr(disk.os, "replace", corrupt_manifest_after_install)
    builder = disk.PLEImageBuilder(tmp_path, "manifest-readback", 0, 1, 0, 25)
    builder.add_shard("shard", rows, 0, 25)

    with pytest.raises(RuntimeError, match="manifest changed after install"):
        builder.finalize(0.5)


def test_manifest_from_newer_install_rejects_older_image(tmp_path):
    rows = _fp8_rows(25)
    image = build_test_image(
        tmp_path, rows, config_sha256="torn-install", weight_scale=0.5
    )
    manifest_path = image.path.parent / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["fingerprint"] = "newer-image-fingerprint"
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="delete.*before rebuilding"):
        disk.PLEImageBuilder(tmp_path, "torn-install", 0, 1, 0, 25)


def test_manifest_source_identity_controls_reuse(tmp_path, caplog):
    source = tmp_path / "checkpoint.safetensors"
    source.write_bytes(b"checkpoint-v1")
    rows = _fp8_rows(25)

    def set_source_identity(tensor):
        stat = source.stat()
        tensor._sglang_checkpoint_source = {
            "file": source.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        return tensor

    first = disk.PLEImageBuilder(tmp_path, "source-identity", 0, 1, 0, 25)
    first.add_shard("shard", set_source_identity(rows), 0, 25)
    first_image, reused, _ = first.finalize(0.5)
    assert not reused

    same = disk.PLEImageBuilder(tmp_path, "source-identity", 0, 1, 0, 25)
    same.add_shard("shard", set_source_identity(rows), 0, 25)
    _, reused, _ = same.finalize(0.5)
    assert reused

    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    touched = disk.PLEImageBuilder(tmp_path, "source-identity", 0, 1, 0, 25)
    touched._materialize_reuse_rows = lambda: pytest.fail(
        "mtime-only changes must not materialize the image"
    )
    touched.add_shard("shard", set_source_identity(rows), 0, 25)
    touched_image, reused, _ = touched.finalize(0.5)
    assert reused
    assert touched_image.path == first_image.path
    assert "identity changed" not in caplog.text


@pytest.mark.parametrize("source", [None, {}])
def test_missing_checkpoint_source_identity_is_an_error(tmp_path, source):
    rows = _fp8_rows(25).clone()
    if source is not None:
        rows._sglang_checkpoint_source = source
    builder = disk.PLEImageBuilder(tmp_path, "missing-source", 0, 1, 0, 25)
    try:
        with pytest.raises(ValueError, match="requires checkpoint source identity"):
            builder.add_shard("shard", rows, 0, 25)
    finally:
        builder.close()


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
    builder.add_shard("first", attach_checkpoint_source(rows[:25]), 0, 25)
    assert list(tmp_path.rglob("*.tmp"))

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        builder.add_shard(
            "second",
            attach_checkpoint_source(rows[25:].to(torch.bfloat16)),
            25,
            50,
        )
    assert not list(tmp_path.rglob("*.tmp"))


def test_builder_error_closes_once(tmp_path, monkeypatch):
    builder = disk.PLEImageBuilder(tmp_path, "write-error", 0, 1, 0, 25)
    close_calls = 0
    real_close = builder.close

    def counted_close():
        nonlocal close_calls
        close_calls += 1
        real_close()

    monkeypatch.setattr(builder, "close", counted_close)
    monkeypatch.setattr(
        disk.os,
        "pwrite",
        lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "injected")),
    )
    with pytest.raises(OSError, match="injected"):
        builder.add_shard("shard", _fp8_rows(25), 0, 25)
    assert close_calls == 1


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


def test_transfer_buffer_retains_the_largest_prefill_chunk():
    retain_rows = qwen4_exp_module._ple_transfer_buffer_retain_rows(
        8192, 16, max_prefill_chunk_tokens=16384
    )
    assert retain_rows == 262144


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

    image = build_test_image(tmp_path, _fp8_rows(25))
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
    image = build_test_image(tmp_path, source_rows)
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


def test_static_hot_cache_checks_direct_reader_page_crc(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    with image.path.open("r+b") as handle:
        handle.seek(disk.PAGE_BYTES + 7)
        value = handle.read(1)
        handle.seek(disk.PAGE_BYTES + 7)
        handle.write(bytes([value[0] ^ 1]))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        with pytest.raises(IOError, match="checksum mismatch"):
            disk.RankSelectHotCache(
                image,
                np.array([0], dtype=np.int64),
                (disk.ROW_BYTES + 1) / (1 << 30),
                reader=reader,
            )
    finally:
        reader.close()


@pytest.mark.parametrize("seed", range(5))
def test_hot_file_frequency_order_round_trips_through_rank_select_cache(tmp_path, seed):
    rng = np.random.default_rng(seed)
    source_rows = _fp8_rows(64)
    image = build_test_image(tmp_path, source_rows, tp_size=2)
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
        build_test_image(tmp_path, layer_rows, module_prefix=f"ple.{layer}")
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


def test_hot_frequency_template_requires_each_resolved_layer_file(tmp_path):
    template = str(tmp_path / "hot-{layer}.bin")
    (tmp_path / "hot-0.bin").touch()

    with pytest.raises(ValueError, match=rf"layer 1.*{tmp_path / 'hot-1.bin'}"):
        disk.resolve_hot_frequency_file(template, 1, 2, require_exists=True)


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


def test_hot_file_reader_requires_the_image_fingerprint(tmp_path):
    path = tmp_path / "hot.bin"
    disk.write_hot_frequency_file(
        path,
        {0: np.array([1], dtype=np.uint32)},
        fingerprint="image",
        total_rows=4,
        tp_size=1,
        padding_divisor=1,
    )
    with pytest.raises(TypeError, match="expected_fingerprint"):
        disk.read_hot_frequency_file(path, 0)


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


def test_hot_file_rejects_a_missing_rank_entry_with_its_path(tmp_path):
    path = tmp_path / "hot-rank.bin"
    disk.write_hot_frequency_file(
        path,
        {0: np.array([1], dtype=np.uint32)},
        fingerprint="image",
        total_rows=4,
        tp_size=1,
        padding_divisor=1,
    )
    header = disk._read_metadata_page(path, disk.HOT_MAGIC)
    header["ranks"] = []
    with path.open("r+b") as handle:
        handle.write(disk._write_metadata_page(disk.HOT_MAGIC, header))

    with pytest.raises(ValueError, match=rf"{path}.*rank 0.*regenerate"):
        disk.read_hot_frequency_file(path, 0, expected_fingerprint="image")


def test_direct_reader_uses_device_block_alignment(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
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


def test_direct_reader_retries_native_destroy_after_busy(tmp_path, monkeypatch, caplog):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    fd = reader.fd
    handle = reader.handle
    results = iter((-errno.EBUSY, 0))
    reader.lib.ple_fetcher_destroy = _FakeFunction(lambda value: next(results))
    with caplog.at_level("ERROR"), pytest.raises(OSError, match="shutdown failed"):
        reader.close()
    assert reader.handle == handle
    os.fstat(fd)
    assert "busy" in caplog.text
    assert "could not drain" not in caplog.text

    reader.close()
    with pytest.raises(OSError) as exc_info:
        os.fstat(fd)
    assert exc_info.value.errno == errno.EBADF
    assert reader.handle is None


def test_closed_direct_reader_raises_before_native_call():
    reader = disk.DirectPageReader.__new__(disk.DirectPageReader)
    reader.handle = None
    reader.image = SimpleNamespace(num_pages=1)

    with pytest.raises(RuntimeError, match="reader is closed"):
        reader._read_locked(np.array([0], dtype=np.int64))


def test_direct_reader_close_has_a_lock_wait_bound():
    class BusyLock:
        def acquire(self, *, timeout):
            assert timeout == 0.25
            return False

        def release(self):
            pytest.fail("busy lock was released")

    reader = disk.DirectPageReader.__new__(disk.DirectPageReader)
    reader._read_lock = BusyLock()
    reader._read_lock_timeout_seconds = 0.25

    with pytest.raises(TimeoutError, match="shutdown.*reader lock"):
        reader.close()


def test_partial_reader_destructor_swallows_busy_cleanup_error():
    reader = disk.DirectPageReader.__new__(disk.DirectPageReader)
    reader.close = lambda: (_ for _ in ()).throw(OSError(errno.EBUSY, "busy"))

    disk.DirectPageReader.__del__(reader)


def test_busy_reader_gc_retains_registered_staging(tmp_path, monkeypatch, caplog):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    retained = []
    monkeypatch.setattr(disk, "_RETAINED_POISONED_STAGING", retained)
    reader = disk.DirectPageReader(image, max_pages=1)
    fd = reader.fd
    staging = reader._staging_allocation
    reader.lib.ple_fetcher_destroy = _FakeFunction(lambda handle: -errno.EBUSY)

    with caplog.at_level("ERROR"):
        del reader
        gc.collect()

    assert retained == [staging]
    assert "garbage collection" in caplog.text
    os.close(fd)


def test_disk_fetcher_calls_raise_after_close():
    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    fetcher._closed = True
    fetcher._prefill_executor = None

    with pytest.raises(RuntimeError, match="PLE disk fetcher is closed"):
        fetcher.fetch(np.array([0], dtype=np.int64))
    with pytest.raises(RuntimeError, match="PLE disk fetcher is closed"):
        fetcher.submit_prefill(np.array([0], dtype=np.int64))
    with pytest.raises(RuntimeError, match="PLE disk fetcher is closed"):
        fetcher.wait_prefill()


def test_disk_fetcher_close_waits_for_an_inflight_fetch():
    entered = threading.Event()
    release = threading.Event()
    fetch_done = threading.Event()
    close_done = threading.Event()
    errors = []

    class BlockingHot:
        rows = torch.zeros((1, disk.ROW_BYTES), dtype=torch.uint8)

        def lookup(self, local):
            entered.set()
            release.wait(5.0)
            return np.ones(local.size, dtype=np.bool_), np.zeros(
                local.size, dtype=np.int64
            )

    class Closeable:
        def close(self):
            pass

    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    fetcher._closed = False
    fetcher._fetch_lock = threading.Lock()
    fetcher._prefill_executor = None
    fetcher._prefill_slots = []
    fetcher._prefill_lock = threading.Lock()
    fetcher._thread_stats = threading.local()
    fetcher.image = SimpleNamespace(vocab_start=0, vocab_end=1)
    fetcher.hot = BlockingHot()
    fetcher.dynamic = Closeable()
    fetcher.prefill_reader = None
    fetcher.reader = Closeable()

    def run_fetch():
        try:
            fetcher.fetch(np.array([0], dtype=np.int64))
        except BaseException as exc:
            errors.append(exc)
        finally:
            fetch_done.set()

    def run_close():
        try:
            fetcher.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_done.set()

    fetch_thread = threading.Thread(target=run_fetch, daemon=True)
    close_thread = threading.Thread(target=run_close, daemon=True)
    try:
        fetch_thread.start()
        assert entered.wait(5.0)
        close_thread.start()
        assert not close_done.wait(0.05)
        release.set()
        assert fetch_done.wait(5.0)
        assert close_done.wait(5.0)
        assert errors == []
    finally:
        release.set()
        fetch_thread.join(5.0)
        close_thread.join(5.0)


def test_fetcher_constructor_closes_decode_reader_when_prefill_reader_fails(
    tmp_path, monkeypatch
):
    image = build_test_image(tmp_path, _fp8_rows(25))
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
    image = build_test_image(tmp_path, _fp8_rows(50))
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


def test_fetcher_close_drains_queued_prefill_before_marking_closed(caplog):
    release = threading.Event()
    worker_started = threading.Event()
    close_waiting = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    closer = None
    try:
        executor.submit(lambda: (worker_started.set(), release.wait()))
        assert worker_started.wait(1.0)

        fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
        fetcher._closed = False
        fetcher._prefill_executor = executor
        fetcher._prefill_lock = threading.Lock()
        queued = executor.submit(
            lambda: (
                (_ for _ in ()).throw(RuntimeError("closed during drain"))
                if fetcher._closed
                else None
            )
        )

        class TrackedFuture:
            def result(self):
                close_waiting.set()
                return queued.result()

        fetcher._prefill_futures = {TrackedFuture()}
        fetcher.hot = None
        fetcher.dynamic = None
        fetcher.prefill_reader = None
        fetcher.reader = None
        close_result = {}

        def close_fetcher():
            try:
                fetcher.close()
            except BaseException as exc:
                close_result["error"] = exc

        closer = threading.Thread(target=close_fetcher, daemon=True)
        with caplog.at_level("ERROR", logger=disk.__name__):
            closer.start()
            assert close_waiting.wait(1.0)
            assert not fetcher._closed
            release.set()
            closer.join(2.0)

        assert not closer.is_alive()
        assert "error" not in close_result
        assert fetcher._closed
        assert not any(
            record.levelno >= 40 and record.name == disk.__name__
            for record in caplog.records
        )
    finally:
        release.set()
        if closer is not None:
            closer.join(2.0)
        executor.shutdown(wait=True)


def test_prefill_dynamic_hits_do_not_enter_the_admission_queue(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    image = build_test_image(tmp_path, rows)
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
    waiter_started = threading.Event()
    executor_calls = []
    results = {}

    class TrackedEvent:
        def __init__(self):
            self.event = threading.Event()

        def wait(self):
            waiter_started.set()
            return self.event.wait()

        def set(self):
            self.event.set()

    class BlockingReader:
        def close(self):
            entered_close.set()
            results["reader_released"] = release_close.wait(1.0)

    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    fetcher._prefill_lock = threading.Lock()
    fetcher._prefill_disabled = False
    fetcher._prefill_disable_done = TrackedEvent()
    fetcher._prefill_slots = [{"state": "ready", "count": 1}]
    fetcher.prefill_reader = BlockingReader()
    fetcher._prefill_executor = SimpleNamespace(
        shutdown=lambda wait, cancel_futures=False: executor_calls.append(
            (wait, cancel_futures)
        )
    )
    monkeypatch.setattr(disk.logger, "error", lambda *args, **kwargs: None)

    error = OSError(errno.EIO, "injected prefill failure")
    owner = threading.Thread(target=fetcher._disable_prefill, args=(error,))
    owner.start()
    assert entered_close.wait(1.0)

    waiter = threading.Thread(
        target=lambda: (fetcher._disable_prefill(error), waiter_done.set())
    )
    try:
        waiter.start()
        assert waiter_started.wait(1.0)
        assert not waiter_done.is_set()
    finally:
        release_close.set()
        owner.join(1.0)
        waiter.join(1.0)
    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert waiter_done.is_set()
    assert results == {"reader_released": True}
    assert executor_calls == [(False, True)]


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
    image = build_test_image(tmp_path, _fp8_rows(25))
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


def test_image_builder_reserves_space_for_all_tp_ranks(tmp_path, monkeypatch):
    all_ranks = 25 * disk.ROW_BYTES + 4 * disk.PAGE_BYTES + 2 * (16 + 4)
    monkeypatch.setattr(
        disk.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=all_ranks - 1),
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
    assert disk.config_digest({"unrelated": object()}) == disk.config_digest({})
    with pytest.raises(ValueError, match="not JSON-serializable"):
        disk.config_digest({"ngram_size": object()})


def test_cache_budget_divisor_accounts_for_classic_dp_replicas():
    assert qwen4_exp_module._ple_cache_budget_divisor(2, 3, 4) == 24
    assert qwen4_exp_module._ple_cache_budget_divisor(8, 3, 1) == 24


def test_image_fingerprint_uses_only_explicit_identity_fields():
    config = {
        "model_type": "qwen4_exp_text",
        "vocab_size": 100,
        "seed": 1234,
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 20_000_000,
        "make_ngram_vocab_size_divisible_by": 128,
        "ple_embedding_dtype": "float8_e4m3fn",
        "eos_token_id": 2,
    }
    digest = disk.config_digest(config)
    assert digest == disk.config_digest({**config, "unrelated_server_arg": 9})
    assert digest != disk.config_digest({**config, "ngram_size": 4})
    assert digest != disk.config_digest({**config, "eos_token_id": 3})
    assert digest == disk.config_digest({**config, "eos_token_id": 2})

    manifest = [{"name": "weight", "sample_sha256": "a"}]
    identity = {
        "config_sha256": digest,
        "tp_size": 2,
        "padded_vocab_size": 320_000_128,
        "valid_vocab_size": 320_000_016,
        "dtype": "float8_e4m3fn",
        "module_prefix": "model.layers.1.ple.ple_embedding.ngram_embedding",
        "manifest": manifest,
    }
    fingerprint = disk.checkpoint_fingerprint(**identity)
    for name, value in (
        ("config_sha256", "different"),
        ("tp_size", 4),
        ("padded_vocab_size", 320_000_256),
        ("valid_vocab_size", 320_000_017),
        ("dtype", "float8_e5m2"),
        ("module_prefix", "model.layers.2.ple.ple_embedding.ngram_embedding"),
    ):
        assert fingerprint != disk.checkpoint_fingerprint(**{**identity, name: value})


def test_prefill_priority_requires_its_own_reader(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
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
        assert fetcher.hot is None


def test_fetch_clears_only_rows_outside_the_local_shard(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(image, hot_cache_gb=0, max_pages=1)

    class Output:
        def __init__(self):
            self.values = np.full((2, disk.ROW_BYTES), 77, dtype=np.uint8)
            self.shape = self.values.shape
            self.dtype = torch.uint8
            self.device = SimpleNamespace(type="cpu")

        def is_contiguous(self):
            return True

        def numpy(self):
            return self.values

    output = Output()
    try:
        returned = fetcher.fetch(np.array([-1, 0]), out=output)
        assert returned is output
        assert not output.values[0].any()
        assert np.array_equal(output.values[1], rows.view(torch.uint8).numpy()[0])
    finally:
        fetcher.close()


def test_dynamic_cache_reports_a_full_admission_queue(caplog):
    cache = disk.WTinyLFURowCache(capacity_rows=8, queue_batches=1, start_worker=False)
    try:
        cache._queue.put_nowait(object())
        with caplog.at_level("WARNING"):
            cache.record(np.array([1]), np.zeros((1, disk.ROW_BYTES), dtype=np.uint8))
        assert "dropped 1 admission batches" in caplog.text
    finally:
        cache.close()


def test_disk_stats_export_admission_drops_and_queue_depth():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    dynamic = SimpleNamespace(
        _lock=threading.Lock(),
        _dropped_batches=7,
        _queue=queue.Queue(),
    )
    dynamic._queue.put(object())
    embedding._fetcher = SimpleNamespace(dynamic=dynamic)
    embedding._stats = {"steps": 3}

    snapshot = embedding.stats_snapshot()

    assert snapshot["dynamic_admission_dropped"] == 7
    assert snapshot["dynamic_admission_queue_depth"] == 1


def test_dynamic_cache_hits_do_not_displace_an_admission(monkeypatch):
    worker_entered = threading.Event()
    release_worker = threading.Event()
    original_loop = disk.WTinyLFURowCache._admission_loop

    def blocked_loop(cache):
        worker_entered.set()
        release_worker.wait()
        original_loop(cache)

    monkeypatch.setattr(disk.WTinyLFURowCache, "_admission_loop", blocked_loop)
    cache = disk.WTinyLFURowCache(capacity_rows=8, queue_batches=1)
    row = np.arange(disk.ROW_BYTES, dtype=np.uint8)
    cache._insert(3, row)
    output = np.zeros((1, disk.ROW_BYTES), dtype=np.uint8)
    try:
        assert worker_entered.wait(1.0)
        assert cache.lookup_into(np.array([3]), output).tolist() == [True]
        cache.record(np.array([4]), row.reshape(1, -1))

        queued = list(cache._queue.queue)
        assert cache._dropped_batches == 0
        assert len(queued) == 1
        assert queued[0][1] is not None
    finally:
        release_worker.set()
        cache.close()


def test_dynamic_cache_lookup_waits_for_admission_batch_lock():
    cache = disk.WTinyLFURowCache(capacity_rows=8)
    row = np.arange(disk.ROW_BYTES, dtype=np.uint8)
    cache._insert(3, row)
    output = np.zeros((1, disk.ROW_BYTES), dtype=np.uint8)
    finished = threading.Event()
    lock_attempted = threading.Event()
    result = {}
    main_thread = threading.get_ident()

    class SignalingLock:
        def __init__(self):
            self.lock = threading.Lock()

        def acquire(self, *args, **kwargs):
            if threading.get_ident() != main_thread:
                lock_attempted.set()
            return self.lock.acquire(*args, **kwargs)

        def release(self):
            self.lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *args):
            self.release()

    cache._lock = SignalingLock()

    def lookup():
        try:
            result["hit"] = cache.lookup_into(
                np.array([3], dtype=np.int64), output, record_hits=False
            )
        except BaseException as exc:
            result["error"] = exc
        finally:
            finished.set()

    cache._lock.acquire()
    thread = threading.Thread(target=lookup)
    thread.start()
    try:
        assert lock_attempted.wait(1.0)
        assert not finished.is_set()
    finally:
        cache._lock.release()
        thread.join(timeout=1.0)
        cache.close()
    assert "error" not in result
    assert result["hit"].tolist() == [True]
    assert np.array_equal(output[0], row)


def test_dynamic_cache_admission_bounds_each_lock_acquisition():
    cache = disk.WTinyLFURowCache.__new__(disk.WTinyLFURowCache)
    cache._queue = queue.Queue()
    cache._queue.put((np.arange(145, dtype=np.int64), None))
    cache._queue.put(None)
    cache._pending_condition = threading.Condition()
    cache._pending_batches = 1
    cache._worker_error = None

    class TrackingLock:
        def __init__(self):
            self.current = 0
            self.work = []

        def __enter__(self):
            self.current = 0
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.work.append(self.current)

    lock = TrackingLock()
    cache._lock = lock
    cache._increment = lambda row_id: setattr(lock, "current", lock.current + 1)
    cache._admission_loop()
    assert len(lock.work) > 1
    assert max(lock.work) <= 1


def test_large_reader_request_scales_lock_wait_by_chunk_count():
    reader = disk.DirectPageReader.__new__(disk.DirectPageReader)
    reader.max_pages = 2
    reader._read_lock_timeout_seconds = 1.5
    waits = []

    class Lock:
        def acquire(self, *, timeout):
            waits.append(timeout)
            return True

        def release(self):
            pass

    reader._read_lock = Lock()
    reader._read_locked = lambda page_ids, return_staging: np.empty(
        (page_ids.size, disk.PAGE_BYTES), dtype=np.uint8
    )

    reader.read(np.arange(5))

    assert waits == [4.5]


def test_dynamic_cache_record_checks_closed_under_the_condition():
    cache = disk.WTinyLFURowCache.__new__(disk.WTinyLFURowCache)
    cache.capacity = 1
    cache._closed = False
    cache._pending_batches = 0
    cache._queue = queue.Queue()

    class ClosingCondition:
        def __enter__(self):
            cache._closed = True

        def __exit__(self, exc_type, exc, traceback):
            pass

        def notify_all(self):
            pass

    cache._pending_condition = ClosingCondition()
    cache.record(np.array([1]), None)
    assert cache._pending_batches == 0
    assert cache._queue.empty()


def test_dynamic_cache_budget_includes_rows_and_metadata():
    budget_bytes = 8 * (disk.ROW_BYTES + 16) + 4 * 1024
    cache = disk.WTinyLFURowCache(budget_gb=budget_bytes / (1 << 30))
    try:
        allocated = (
            cache.rows.numel()
            + cache.tags.nbytes
            + cache.recency.nbytes
            + cache.sketch.nbytes
        )
        assert cache.capacity == 8
        assert allocated <= budget_bytes
    finally:
        cache.close()


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
        "_insert_locked",
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


def test_disk_embedding_close_synchronizes_transfer_before_clear():
    events = []

    class Completion:
        def synchronize(self):
            events.append("synchronize")

    class Buffers(dict):
        def clear(self):
            events.append("clear")
            super().clear()

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._future = None
    embedding._completion_event = Completion()
    embedding._prefill_submit_future = None
    embedding._fetcher = None
    embedding._image_builder = None
    embedding._executor = None
    embedding._transfer_buffers = Buffers(device=object())

    embedding.close()

    assert events == ["synchronize", "clear"]


def test_disk_embedding_close_continues_after_fetcher_error():
    events = []

    class Closeable:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    class Executor:
        def shutdown(self, wait=True):
            events.append("executor")

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._future = None
    embedding._prefill_submit_future = None
    embedding._fetcher = Closeable("fetcher", RuntimeError("fetcher close"))
    embedding._image_builder = Closeable("builder")
    embedding._executor = Executor()
    embedding._transfer_buffers = {"device": object()}
    embedding._active_transfer_device = object()
    embedding._prefill_host_ids = object()
    embedding._active_graph_generation = object()

    with pytest.raises(RuntimeError, match="fetcher close"):
        embedding.close()
    embedding.close()
    assert events == ["fetcher", "builder", "executor"]
    assert embedding._fetcher is None
    assert embedding._image_builder is None
    assert embedding._executor is None
    assert embedding._transfer_buffers == {}


def test_disk_fetcher_close_continues_after_dynamic_cache_error():
    events = []

    class Closeable:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    fetcher = disk.DiskRowFetcher.__new__(disk.DiskRowFetcher)
    fetcher._closed = False
    fetcher._prefill_executor = None
    fetcher.hot = object()
    fetcher.dynamic = Closeable("dynamic", RuntimeError("dynamic close"))
    fetcher.prefill_reader = Closeable("prefill")
    fetcher.reader = Closeable("decode")

    with pytest.raises(RuntimeError, match="dynamic close"):
        fetcher.close()
    fetcher.close()
    assert events == ["dynamic", "prefill", "decode"]
    assert fetcher.dynamic is None
    assert fetcher.prefill_reader is None
    assert fetcher.reader is None


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


def test_weight_reload_builds_replacement_while_live_image_serves(monkeypatch):
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
    old_image = object()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = old_fetcher
    embedding._image = old_image
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

    assert not old_fetcher.closed
    assert not old_executor.closed
    assert embedding._image is old_image
    assert embedding._fetcher is old_fetcher
    assert len(created) == 1
    assert created[0].shards[0][0] == "shard_0.weight"


def test_real_builder_weight_reload_replaces_the_reusable_image(tmp_path):
    rows = _fp8_rows(25)
    builder_args = {
        "root": tmp_path,
        "config_sha256": "real-reload",
        "rank": 0,
        "tp_size": 1,
        "vocab_start": 0,
        "vocab_end": 25,
    }
    first = disk.PLEImageBuilder(**builder_args)
    first.add_shard("shard_0.weight", rows, 0, 25)
    old_image, _, _ = first.finalize(0.5)

    class Fetcher:
        def __init__(self):
            self.closed = False
            self.hot = SimpleNamespace(
                rows=torch.empty(0, dtype=torch.uint8),
                bitmap=np.empty(0, dtype=np.uint64),
                rank_prefix=np.empty(0, dtype=np.uint32),
            )
            self.reader = SimpleNamespace(staging=torch.empty(0, dtype=torch.uint8))
            self.dynamic = SimpleNamespace(rows=torch.empty(0, dtype=torch.uint8))
            self._prefill_slots = []

        def close(self):
            self.closed = True

    old_fetcher = Fetcher()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._weight_reload_pending = False
    embedding._image_builder = None
    embedding._builder_args = builder_args
    embedding._image = old_image
    embedding._fetcher = old_fetcher
    embedding._build_fetcher = lambda image: Fetcher()
    embedding.weight_scale = torch.tensor(0.5)
    embedding._rank = 0
    embedding._hot_frequency_file = None
    embedding._hot_cache_gb = 0.0

    changed = rows.clone().view(torch.uint8)
    changed[0, 0] = (changed[0, 0] + 1) % 0x7F
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2
    embedding.prepare_weight_reload()
    embedding.add_checkpoint_shard("shard_0.weight", changed, 0, 25)
    embedding.finalize_image()

    assert embedding._image.path != old_image.path
    assert old_fetcher.closed
    assert list(tmp_path.glob("*/rank0.bin")) == [embedding._image.path]
    startup = disk.PLEImageBuilder(**builder_args)
    try:
        assert startup._reuse.path == embedding._image.path
    finally:
        startup.close()


def test_generation_cleanup_logs_removed_path_and_size(tmp_path, caplog):
    rows = _fp8_rows(25)
    first = disk.PLEImageBuilder(tmp_path, "cleanup-log", 0, 1, 0, 25)
    first.add_shard("shard", rows, 0, 25)
    old_image, _, _ = first.finalize(0.5)
    changed = rows.clone().view(torch.uint8)
    changed[0, 0] = (changed[0, 0] + 1) % 0x7F
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2

    with caplog.at_level("INFO"):
        replacement = disk.PLEImageBuilder(tmp_path, "cleanup-log", 0, 1, 0, 25)
        replacement.add_shard("shard", changed, 0, 25)
        replacement.finalize(0.5)

    assert not old_image.path.parent.exists()
    assert str(old_image.path.parent) in caplog.text
    assert "bytes" in caplog.text


def test_generation_cleanup_can_be_disabled(tmp_path):
    rows = _fp8_rows(25)
    first = disk.PLEImageBuilder(tmp_path, "cleanup-disabled", 0, 1, 0, 25)
    first.add_shard("shard", rows, 0, 25)
    first.finalize(0.5)
    changed = rows.clone().view(torch.uint8)
    changed[0, 0] = (changed[0, 0] + 1) % 0x7F
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2
    replacement = disk.PLEImageBuilder(
        tmp_path, "cleanup-disabled", 0, 1, 0, 25, cleanup_generations=False
    )
    replacement.add_shard("shard", changed, 0, 25)
    replacement.finalize(0.5)

    assert len(list(tmp_path.glob("*/rank0.bin"))) == 2


def test_generation_cleanup_skips_a_directory_with_an_open_file(tmp_path):
    rows = _fp8_rows(25)
    first = disk.PLEImageBuilder(tmp_path, "cleanup-open", 0, 1, 0, 25)
    first.add_shard("shard", rows, 0, 25)
    old_image, _, _ = first.finalize(0.5)
    changed = rows.clone().view(torch.uint8)
    changed[0, 0] = (changed[0, 0] + 1) % 0x7F
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2

    with old_image.path.open("rb"):
        replacement = disk.PLEImageBuilder(tmp_path, "cleanup-open", 0, 1, 0, 25)
        replacement.add_shard("shard", changed, 0, 25)
        replacement.finalize(0.5)

    assert old_image.path.exists()


def test_generation_cleanup_skips_a_directory_locked_by_another_process(tmp_path):
    rows = _fp8_rows(25)
    first = disk.PLEImageBuilder(tmp_path, "cleanup-process", 0, 1, 0, 25)
    first.add_shard("shard", rows, 0, 25)
    old_image, _, _ = first.finalize(0.5)
    lock_path = old_image.path.parent / ".reader.lock"
    holder_code = """
import fcntl
import os
import sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_SH)
print("ready", flush=True)
sys.stdin.read()
"""
    holder = subprocess.Popen(
        [sys.executable, "-u", "-c", holder_code, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "ready"
    changed = rows.clone().view(torch.uint8)
    changed[0, 0] = (changed[0, 0] + 1) % 0x7F
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2
    try:
        replacement = disk.PLEImageBuilder(tmp_path, "cleanup-process", 0, 1, 0, 25)
        replacement.add_shard("shard", changed, 0, 25)
        replacement.finalize(0.5)
        assert old_image.path.exists()
    finally:
        holder.stdin.close()
        holder.wait(timeout=5.0)


def test_builder_sweeps_only_old_dot_prefixed_scratch_files(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    stale_raw = tmp_path / ".rank0.abc.rows.tmp"
    stale_image = generation / ".rank0.bin.abc.tmp"
    recent = tmp_path / ".rank1.recent.rows.tmp"
    visible = tmp_path / "rank0.visible.rows.tmp"
    for path in (stale_raw, stale_image, recent, visible):
        path.write_bytes(b"scratch")
    old = time.time() - disk._STALE_SCRATCH_AGE_SECONDS - 1
    os.utime(stale_raw, (old, old))
    os.utime(stale_image, (old, old))

    builder = disk.PLEImageBuilder(
        tmp_path, "scratch-sweep", 0, 1, 0, 25, allow_reuse=False
    )
    builder.close()

    assert not stale_raw.exists()
    assert not stale_image.exists()
    assert recent.exists()
    assert visible.exists()


def test_failed_weight_reload_keeps_the_previous_image_serving(monkeypatch):
    class Builder:
        def __init__(self, **kwargs):
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
        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(disk, "PLEImageBuilder", Builder)
    old_fetcher = Closeable()
    old_image = object()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.embedding_dim = 4
    embedding.tp_size = 1
    embedding._fetcher = old_fetcher
    embedding._image = old_image
    embedding._image_builder = None
    embedding._executor = Executor()
    embedding._future = None
    embedding._prefill_submit_future = None
    embedding._transfer_buffers = {}
    embedding._active_transfer_device = None
    embedding._prefill_host_ids = None
    embedding._builder_args = {}
    embedding._new_executor = Executor
    embedding.allocate_output = lambda shape, device: torch.empty(
        shape, dtype=torch.bfloat16
    )
    embedding._launch_fetch = lambda ids, output, **kwargs: output.fill_(7)
    embedding.wait_for_prefetch = lambda: None
    embedding.reduce = lambda output: output

    def fake_loader():
        yield "shard_0.weight", _fp8_rows(1)
        raise OSError("fake loader failed")

    embedding.prepare_weight_reload()
    with pytest.raises(OSError, match="fake loader failed"):
        for name, loaded_weight in fake_loader():
            embedding.add_checkpoint_shard(name, loaded_weight, 0, 1)

    assert embedding._fetcher is old_fetcher
    assert embedding._image is old_image
    assert not old_fetcher.closed
    embedding.resume_storage()
    assert torch.all(embedding.forward(torch.tensor([0])) == 7)


def test_partial_weight_reload_names_filtered_updates():
    class Builder:
        def finalize(self, weight_scale):
            raise ValueError(
                "PLE checkpoint shards do not exactly cover TP rank 0: [[0, 1]]"
            )

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._weight_reload_pending = False
    embedding._image_builder = Builder()
    embedding._image = object()
    embedding._fetcher = object()
    embedding.weight_scale = torch.tensor(0.5)

    with pytest.raises(ValueError, match="filtered updates"):
        embedding.finalize_image()


def test_successful_weight_reload_swaps_fetchers_after_replacement_opens():
    image = SimpleNamespace(path=Path("replacement.bin"))

    class Builder:
        def finalize(self, weight_scale):
            return (
                image,
                False,
                {
                    "conversion_seconds": 1.0,
                    "conversion_gib_per_s": 2.0,
                },
            )

    class Fetcher:
        def __init__(self, with_stats=False):
            self.closed = False
            if with_stats:
                self.hot = SimpleNamespace(
                    rows=torch.empty(0, dtype=torch.uint8),
                    bitmap=np.empty(0, dtype=np.uint64),
                    rank_prefix=np.empty(0, dtype=np.uint32),
                )
                self.reader = SimpleNamespace(staging=torch.empty(0, dtype=torch.uint8))
                self.dynamic = SimpleNamespace(rows=torch.empty(0, dtype=torch.uint8))
                self._prefill_slots = []

        def close(self):
            self.closed = True

    old_fetcher = Fetcher()
    new_fetcher = Fetcher(with_stats=True)
    old_image = object()
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._weight_reload_pending = False
    embedding._image_builder = Builder()
    embedding._image = old_image
    embedding._fetcher = old_fetcher
    embedding._build_fetcher = lambda candidate: new_fetcher
    embedding.weight_scale = torch.tensor(0.5)
    embedding._rank = 0
    embedding._hot_frequency_file = None
    embedding._hot_cache_gb = 0.0

    embedding.finalize_image()

    assert embedding._image is image
    assert embedding._fetcher is new_fetcher
    assert old_fetcher.closed
    assert not new_fetcher.closed


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
    image = build_test_image(
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
    image = build_test_image(tmp_path, _fp8_rows(25), config_sha256="old")
    with image.path.open("r+b") as handle:
        handle.write(b"PLEDISK2")
    with pytest.raises(ValueError, match=rf"{image.path.parent}.*rebuild"):
        disk.PLEImageBuilder(tmp_path, "old", 0, 1, 0, 25)


def test_weight_scale_change_rebuilds_the_image(tmp_path, caplog):
    rows = _fp8_rows(25)
    build_test_image(tmp_path, rows, config_sha256="scale", weight_scale=0.25)
    builder = disk.PLEImageBuilder(tmp_path, "scale", 0, 1, 0, 25)
    builder.add_shard("test.shard_0.weight", rows, 0, 25)
    with caplog.at_level("WARNING"):
        image, reused, _ = builder.finalize(0.5)
    assert not reused
    assert image.header["weight_scale"] == 0.5
    assert "scale changed" in caplog.text


def test_unsampled_payload_change_rebuilds_same_shape_image(tmp_path, caplog):
    rows = _fp8_rows(20_000)
    build_test_image(tmp_path, rows, config_sha256="payload", weight_scale=0.5)
    changed = rows.view(torch.uint8).clone()
    changed[1, 7] ^= 1
    changed = attach_checkpoint_source(changed.view(torch.float8_e4m3fn))
    changed._sglang_checkpoint_source["mtime_ns"] = 2
    builder = disk.PLEImageBuilder(tmp_path, "payload", 0, 1, 0, 20_000)
    builder.add_shard(
        "test.shard_0.weight",
        changed,
        0,
        20_000,
    )
    with caplog.at_level("WARNING"):
        _, reused, _ = builder.finalize(0.5)
    assert not reused
    assert "identity changed" in caplog.text


def test_unchanged_source_metadata_uses_digest_fast_path(tmp_path, monkeypatch, caplog):
    rows = _fp8_rows(25)
    build_test_image(tmp_path, rows, config_sha256="fast-path", weight_scale=0.5)
    builder = disk.PLEImageBuilder(tmp_path, "fast-path", 0, 1, 0, 25)
    monkeypatch.setattr(
        builder,
        "_stream_shard",
        lambda *args, **kwargs: pytest.fail("fast path streamed checkpoint rows"),
    )

    with caplog.at_level("INFO"):
        builder.add_shard("test.shard_0.weight", rows, 0, 25)
        _, reused, _ = builder.finalize(0.5)

    assert reused
    assert "reuse fast path" in caplog.text


def test_changed_mtime_with_the_same_full_digest_reuses_image(tmp_path, caplog):
    rows = _fp8_rows(25)
    build_test_image(tmp_path, rows, config_sha256="digest-match", weight_scale=0.5)
    changed_metadata = attach_checkpoint_source(rows.clone())
    changed_metadata._sglang_checkpoint_source["mtime_ns"] = 2
    builder = disk.PLEImageBuilder(tmp_path, "digest-match", 0, 1, 0, 25)

    with caplog.at_level("INFO"):
        builder.add_shard("test.shard_0.weight", changed_metadata, 0, 25)
        _, reused, _ = builder.finalize(0.5)

    assert reused
    assert "verified the full content digest" in caplog.text


def test_unregistered_staging_reads_with_the_same_pointer(tmp_path, monkeypatch):
    raw = _fp8_rows(50).view(torch.uint8)
    image = build_test_image(tmp_path, raw.view(torch.float8_e4m3fn))
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
    image = build_test_image(tmp_path, raw.view(torch.float8_e4m3fn))
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


def test_disk_gather_uses_the_driver_capture_predicate(monkeypatch):
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = object()
    embedding._future = None
    embedding.embedding_dim = 4
    embedding.allocate_output = lambda shape, device: torch.empty(
        shape, dtype=torch.bfloat16, device=device
    )
    embedding._launch_fetch = lambda *args, **kwargs: pytest.fail(
        "capture launched a disk fetch"
    )
    monkeypatch.setattr(qwen4_exp_module, "get_is_model_capture_mode", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(qwen4_exp_module, "_is_stream_capturing", lambda stream: True)

    with pytest.raises(RuntimeError, match="requires a preallocated output"):
        embedding.gather(torch.tensor([0], dtype=torch.long))


def test_disk_fetch_coalesces_pages_across_batch_and_drafts(tmp_path, monkeypatch):
    rows = _fp8_rows(256)
    raw = rows.view(torch.uint8)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(image, hot_cache_gb=0, max_pages=32)
    try:
        ids = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
        actual = fetcher.fetch(ids)
        expected = raw.index_select(0, torch.from_numpy(ids.reshape(-1))).reshape(
            2, 3, disk.ROW_BYTES
        )
        assert torch.equal(actual, expected)
        stats = fetcher.last_fetch_stats
        assert stats.rows_requested == 6
        assert stats.cold_pages == 1
        assert stats.coalesced_rows == 5
    finally:
        fetcher.close()


def test_disk_fetch_reuses_caller_owned_output(tmp_path, monkeypatch):
    rows = _fp8_rows(256)
    raw = rows.view(torch.uint8)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(image, hot_cache_gb=0, max_pages=32)
    ids = np.array([[1, 2], [-1, 5]], dtype=np.int64)
    output = torch.full((*ids.shape, disk.ROW_BYTES), 0xFF, dtype=torch.uint8)
    try:
        actual = fetcher.fetch(ids, out=output)
        assert actual is output
        assert torch.equal(actual[0, 0], raw[1])
        assert torch.equal(actual[0, 1], raw[2])
        assert torch.equal(actual[1, 0], torch.zeros(disk.ROW_BYTES, dtype=torch.uint8))
        assert torch.equal(actual[1, 1], raw[5])
        assert fetcher.fetch(ids, out=output) is output
    finally:
        fetcher.close()


def test_row_content_validator_reports_corrupt_dynamic_hit(
    tmp_path, monkeypatch, caplog
):
    rows = _fp8_rows(25)
    raw = rows.view(torch.uint8)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(
        image, hot_cache_gb=0, dynamic_capacity_rows=8, max_pages=1
    )
    diagnostics = disk.PLEFetchDiagnostics(
        image,
        rank=3,
        module_prefix="model.layers.1.ple",
        max_pages=1,
        validate_row_content=True,
    )
    assert diagnostics._validator.reader is not fetcher.reader
    corrupt = raw[7].numpy().copy()
    corrupt[19] ^= np.uint8(1)
    fetcher.dynamic._insert(7, corrupt)
    tiers = np.empty(1, dtype=np.uint8)
    try:
        actual = fetcher.fetch(np.array([7], dtype=np.int64), tier_out=tiers)
        assert tiers.tolist() == [disk.PLE_ROW_TIER_DYNAMIC]
        with caplog.at_level("ERROR", logger=disk.__name__):
            diagnostics.submit(41, np.array([7], dtype=np.int64), actual, tiers)
            assert diagnostics.wait_validation(1.0)

        snapshot = diagnostics.stats_snapshot()
        assert snapshot["row_content_mismatches"] == 1
        assert "rank=3" in caplog.text
        assert "module=model.layers.1.ple" in caplog.text
        assert "step=41" in caplog.text
        assert "id=7" in caplog.text
        assert "tier=dynamic" in caplog.text
        assert "first_differing_byte=19" in caplog.text
    finally:
        diagnostics.close()
        fetcher.close()


def test_row_content_validator_accepts_exact_rows(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    hot_path = tmp_path / "hot.bin"
    disk.write_hot_frequency_file(
        hot_path,
        {0: np.array([1], dtype=np.uint32)},
        fingerprint=image.header["fingerprint"],
        total_rows=25,
        tp_size=1,
        padding_divisor=1,
    )
    fetcher = disk.DiskRowFetcher(
        image,
        hot_frequency_file=str(hot_path),
        hot_cache_gb=(disk.ROW_BYTES + 1) / (1 << 30),
        dynamic_capacity_rows=8,
        prefill_buffer_tokens=1,
        prefill_read_pages=1,
        max_pages=1,
    )
    diagnostics = disk.PLEFetchDiagnostics(
        image,
        rank=0,
        module_prefix="model.layers.1.ple",
        max_pages=1,
        validate_row_content=True,
    )
    fetcher.dynamic._insert(2, rows.view(torch.uint8)[2].numpy())
    assert fetcher.submit_prefill(np.array([3], dtype=np.int64))
    fetcher.wait_prefill()
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    tiers = np.empty(ids.shape, dtype=np.uint8)
    try:
        actual = fetcher.fetch(ids, tier_out=tiers)
        assert tiers.tolist() == [
            disk.PLE_ROW_TIER_STATIC,
            disk.PLE_ROW_TIER_DYNAMIC,
            disk.PLE_ROW_TIER_PREFILL,
            disk.PLE_ROW_TIER_COLD,
        ]
        diagnostics.submit(1, ids, actual, tiers)
        assert diagnostics.wait_validation(1.0)
        snapshot = diagnostics.stats_snapshot()
        assert snapshot["row_content_rows_checked"] == 4
        assert snapshot["row_content_mismatches"] == 0
    finally:
        diagnostics.close()
        fetcher.close()


def test_row_content_validator_never_blocks_the_serving_fetch(tmp_path, monkeypatch):
    rows = _fp8_rows(25)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(image, hot_cache_gb=0, max_pages=1)
    diagnostics = disk.PLEFetchDiagnostics(
        image,
        rank=0,
        module_prefix="model.layers.1.ple",
        max_pages=1,
        validate_row_content=True,
        validation_max_inflight=1,
    )
    validator_entered = threading.Event()
    release_validator = threading.Event()
    fetch_finished = threading.Event()
    result = {}
    original_validate = diagnostics._validator._validate

    def stalled_validate(batch):
        validator_entered.set()
        release_validator.wait()
        original_validate(batch)

    monkeypatch.setattr(diagnostics._validator, "_validate", stalled_validate)

    def fetch_and_submit():
        ids = np.array([4], dtype=np.int64)
        tiers = np.empty(ids.shape, dtype=np.uint8)
        actual = fetcher.fetch(ids, tier_out=tiers)
        result.update(ids=ids, tiers=tiers, actual=actual)
        diagnostics.submit(9, ids, actual, tiers)
        fetch_finished.set()

    worker = threading.Thread(target=fetch_and_submit)
    try:
        worker.start()
        assert validator_entered.wait(1.0)
        assert fetch_finished.wait(1.0)
        assert not diagnostics.wait_validation(0.01)
        diagnostics.submit(10, result["ids"], result["actual"], result["tiers"])
        assert diagnostics.stats_snapshot()["row_content_batches_dropped"] == 1
    finally:
        release_validator.set()
        worker.join(1.0)
        diagnostics.close()
        fetcher.close()


def test_row_trace_writes_step_hashes_and_tier_counts(tmp_path):
    path = tmp_path / "row-trace.jsonl"
    image = SimpleNamespace()
    diagnostics = disk.PLEFetchDiagnostics(
        image,
        rank=2,
        module_prefix="model.layers.1.ple",
        max_pages=1,
        row_trace_path=path,
    )
    ids = np.array([11, -1], dtype=np.int64)
    rows = np.arange(2 * disk.ROW_BYTES, dtype=np.uint8).reshape(2, disk.ROW_BYTES)
    tiers = np.array(
        [disk.PLE_ROW_TIER_COLD, disk.PLE_ROW_TIER_UNOWNED], dtype=np.uint8
    )
    diagnostics.submit(17, ids, rows, tiers)
    diagnostics.close()

    rank_path = tmp_path / "row-trace.rank2.jsonl"
    records = [json.loads(line) for line in rank_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["rank"] == 2
    assert records[0]["module"] == "model.layers.1.ple"
    assert records[0]["step"] == 17
    assert len(records[0]["ids_hash"]) == 32
    assert len(records[0]["payload_hash"]) == 32
    assert records[0]["tier_counts"] == {
        "unowned": 1,
        "static": 0,
        "prefill": 0,
        "dynamic": 0,
        "cold": 1,
    }


def test_row_diagnostic_shutdown_waits_are_bounded():
    validator = disk._PLERowContentValidator.__new__(disk._PLERowContentValidator)
    validator._closed = False
    validator.max_inflight = 1
    validator.reader = SimpleNamespace(_read_lock_timeout_seconds=0.01)
    validator.wait = lambda timeout=None: False

    with pytest.raises(TimeoutError, match="row validation.*shutdown"):
        validator.close()

    class Worker:
        def join(self, timeout=None):
            assert timeout is not None

        def is_alive(self):
            return True

    trace = disk._PLERowTraceWriter.__new__(disk._PLERowTraceWriter)
    trace._closed = False
    trace._condition = threading.Condition()
    trace._worker = Worker()

    with pytest.raises(TimeoutError, match="row trace.*shutdown"):
        trace.close()


def test_row_diagnostics_environment_is_opt_in(tmp_path, monkeypatch):
    image = build_test_image(tmp_path, _fp8_rows(25))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    monkeypatch.delenv("SGLANG_PLE_DISK_VALIDATE_ROW_CONTENT", raising=False)
    monkeypatch.delenv("SGLANG_PLE_DISK_ROW_TRACE", raising=False)
    assert (
        disk.PLEFetchDiagnostics.from_env(
            image, rank=0, module_prefix="ple", max_pages=1
        )
        is None
    )

    monkeypatch.setenv("SGLANG_PLE_DISK_VALIDATE_ROW_CONTENT", "1")
    monkeypatch.setenv("SGLANG_PLE_DISK_ROW_TRACE", str(tmp_path / "trace.jsonl"))
    diagnostics = disk.PLEFetchDiagnostics.from_env(
        image, rank=0, module_prefix="ple", max_pages=1
    )
    try:
        snapshot = diagnostics.stats_snapshot()
        assert snapshot["row_content_validation_enabled"]
        assert snapshot["row_trace_enabled"]
    finally:
        diagnostics.close()


def test_row_diagnostic_environment_uses_the_registry(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.setenv("SGLANG_PLE_DISK_VALIDATE_ROW_CONTENT", "1")
    monkeypatch.setenv("SGLANG_PLE_DISK_ROW_TRACE", "/tmp/trace-{rank}.jsonl")

    assert envs.SGLANG_PLE_DISK_VALIDATE_ROW_CONTENT.get() is True
    assert envs.SGLANG_PLE_DISK_ROW_TRACE.get() == "/tmp/trace-{rank}.jsonl"


def test_registered_image_failures_cover_crc_fingerprint_and_short_file(
    tmp_path, monkeypatch
):
    rows = _fp8_rows(50)
    image = build_test_image(tmp_path, rows)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        disk.open_ple_image(image.path, expected_fingerprint="wrong")

    with image.path.open("r+b") as handle:
        handle.seek(disk.PAGE_BYTES + 7)
        original = handle.read(1)
        handle.seek(disk.PAGE_BYTES + 7)
        handle.write(bytes([original[0] ^ 0xFF]))
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    reader = disk.DirectPageReader(image, max_pages=1)
    try:
        with pytest.raises(IOError, match="checksum mismatch"):
            reader.read(np.array([0], dtype=np.int64))
    finally:
        reader.close()

    short_image = build_test_image(tmp_path / "short", rows)
    with short_image.path.open("r+b") as handle:
        handle.truncate(short_image.path.stat().st_size - 1)
    with pytest.raises(IOError, match="short image|size mismatch"):
        disk.open_ple_image(short_image.path)


def test_disk_transfer_submits_the_staged_payload_to_diagnostics(monkeypatch):
    events = []
    ids = np.array([12], dtype=np.int64)
    raw_host = torch.empty((1, disk.ROW_BYTES), dtype=torch.uint8)

    class Fetcher:
        last_fetch_stats = object()

        def fetch(self, fetched_ids, *, out, admit_dynamic, tier_out):
            assert np.array_equal(fetched_ids, ids)
            out.fill_(7)
            tier_out.fill(disk.PLE_ROW_TIER_COLD)

    class DeviceBuffer:
        def record_stream(self, stream):
            pass

        def copy_(self, source, non_blocking=False):
            assert source is raw_host
            assert non_blocking

        def view(self, dtype):
            return torch.zeros_like(raw_host).view(dtype)

    class Completion:
        def record(self, stream):
            events.append("completion")

    class Diagnostics:
        def submit(self, step_index, fetched_ids, rows, tiers):
            events.append(
                (
                    "diagnostics",
                    step_index,
                    fetched_ids.copy(),
                    rows.numpy().copy(),
                    tiers.copy(),
                )
            )

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = Fetcher()
    embedding._row_diagnostics = Diagnostics()
    monkeypatch.setattr(
        qwen4_exp_module.torch.cuda, "stream", lambda value: nullcontext()
    )
    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "Event", Completion)

    embedding._fetch_to_device(
        SimpleNamespace(shape=(1,), numpy=lambda: ids),
        SimpleNamespace(synchronize=lambda: None),
        raw_host,
        DeviceBuffer(),
        torch.empty((1, disk.ROW_BYTES), dtype=torch.bfloat16),
        object(),
        True,
        23,
    )

    assert events[0] == "completion"
    _, step_index, submitted_ids, submitted_rows, submitted_tiers = events[1]
    assert step_index == 23
    assert submitted_ids.tolist() == [12]
    assert np.all(submitted_rows == 7)
    assert submitted_tiers.tolist() == [disk.PLE_ROW_TIER_COLD]


def test_dynamic_wtinylfu_admission_eviction_and_exactness(tmp_path, monkeypatch):
    rows = _fp8_rows(1024)
    raw = rows.view(torch.uint8)
    image = build_test_image(tmp_path, rows)
    _patch_fetcher_library(monkeypatch, image)
    monkeypatch.setattr(
        disk, "pageable_memory_access_uses_host_page_tables", lambda: True
    )
    fetcher = disk.DiskRowFetcher(
        image, hot_cache_gb=0, dynamic_capacity_rows=16, max_pages=32
    )
    try:
        same_set = []
        for row_id in range(rows.shape[0]):
            if int(fetcher.dynamic._set_indices(np.array([row_id]))[0]) == 0:
                same_set.append(row_id)
            if len(same_set) == 12:
                break
        protected = same_set[0]
        assert torch.equal(fetcher.fetch(np.array([protected]))[0], raw[protected])
        fetcher.dynamic.flush()
        for _ in range(12):
            assert torch.equal(fetcher.fetch(np.array([protected]))[0], raw[protected])
        fetcher.dynamic.flush()
        for row_id in same_set[1:]:
            assert torch.equal(fetcher.fetch(np.array([row_id]))[0], raw[row_id])
            fetcher.dynamic.flush()

        cached = set(int(row_id) for row_id in fetcher.dynamic.tags[0] if row_id >= 0)
        assert protected in cached
        assert same_set[-1] in cached
        assert same_set[1] not in cached
        assert len(cached) == fetcher.dynamic._WAYS
        actual = fetcher.fetch(np.array([protected]))
        assert torch.equal(actual[0], raw[protected])
        assert fetcher.last_fetch_stats.dynamic_hits == 1
        assert fetcher.last_fetch_stats.cold_pages == 0
    finally:
        fetcher.close()


def test_future_contexts_preserve_request_and_token_order():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._prefill_buffer_tokens = 4
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._future_lookup_contexts = None
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([layer])
    requests = [
        SimpleNamespace(
            origin_input_ids=[10, 11, 12, 13, 14],
            extend_range=SimpleNamespace(end=2),
        ),
        SimpleNamespace(
            origin_input_ids=[20, 21, 22, 23],
            extend_range=SimpleNamespace(end=1),
        ),
    ]
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.arange(3),
    )

    model.prepare_model_batch(SimpleNamespace(reqs=requests), forward_batch)

    torch.testing.assert_close(
        layer._future_lookup_contexts,
        torch.tensor([[10, 11, 12], [11, 12, 13], [12, 13, 14], [2, 20, 21]]),
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
