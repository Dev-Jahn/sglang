"""Exact disk backing for Qwen4 PLE embeddings with 160-byte FP8 rows.

Format 3 stores one 4 KiB metadata block followed by records containing 25
160-byte FP8 rows and 96 bytes of zero padding. The manifest records each
checkpoint shard's global row range. The CRC sidecar contains one CRC32 per
data record.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import queue
import shutil
import struct
import tempfile
import threading
import time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

PAGE_BYTES = 4096
ROW_BYTES = 160
ROWS_PER_PAGE = 25
IMAGE_MAGIC = b"PLEDISK3"
CRC_MAGIC = b"PLCRC001"
HOT_MAGIC = b"PLHOT001"
FORMAT_VERSION = 3
HOT_FORMAT_VERSION = 1
REQUIRED_SGL_KERNEL_VERSION = "0.4.6.post2"
CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
FETCHER_FAILURE_SETUP = 1
FETCHER_FAILURE_REGISTER_BUFFER = 2
FETCHER_ERR_POISONED = getattr(errno, "EUCLEAN", 117)
IORING_MAX_ENTRIES = 32768


def resolve_hot_frequency_file(
    path: Optional[str], ple_layer_index: int, ple_layer_count: int
) -> Optional[str]:
    if not path:
        return None
    ple_layer_index = int(ple_layer_index)
    ple_layer_count = max(1, int(ple_layer_count))
    if not 0 <= ple_layer_index < ple_layer_count:
        raise ValueError(
            f"PLE layer index {ple_layer_index} is outside [0, {ple_layer_count})"
        )
    if "{layer}" in path:
        return path.replace("{layer}", str(ple_layer_index))
    if ple_layer_count > 1:
        raise ValueError(
            "--ple-disk-hot-frequency-file must contain {layer} when the "
            "checkpoint has more than one PLE layer"
        )
    return path


def _allocate_host_tensor(*size, pin_memory: bool = True, **kwargs) -> torch.Tensor:
    return torch.empty(*size, pin_memory=pin_memory, **kwargs)


def _current_cuda_device() -> int:
    return int(torch.cuda.current_device())


def _set_cuda_device(device: int) -> None:
    torch.cuda.set_device(device)


def pageable_memory_access_uses_host_page_tables(
    device: Optional[int] = None,
) -> bool:
    if device is None:
        device = _current_cuda_device()
    try:
        properties = torch.cuda.get_device_properties(device)
        for name in (
            "pageableMemoryAccessUsesHostPageTables",
            "pageable_memory_access_uses_host_page_tables",
        ):
            value = getattr(properties, name, None)
            if value is not None:
                return bool(value)
    except (AttributeError, RuntimeError):
        pass

    cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else None
    library_names = [
        name
        for name in dict.fromkeys(
            [
                f"libcudart.so.{cuda_major}" if cuda_major else None,
                "libcudart.so.13",
                "libcudart.so.12",
                "libcudart.so",
            ]
        )
        if name is not None
    ]
    errors = []
    for library_name in library_names:
        try:
            cudart = ctypes.CDLL(library_name, use_errno=True)
            get_attribute = cudart.cudaDeviceGetAttribute
            get_attribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            get_attribute.restype = ctypes.c_int
            value = ctypes.c_int()
            rc = get_attribute(
                ctypes.byref(value),
                CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
                int(device),
            )
            if rc == 0:
                return bool(value.value)
            errors.append(f"{library_name}: cuda error {rc}")
        except (AttributeError, OSError) as exc:
            errors.append(f"{library_name}: {exc}")
    raise RuntimeError(
        "Could not query CUDA pageable-memory host page tables on device "
        f"{device} ({'; '.join(errors)})"
    )


def resolve_max_read_pages(value: Optional[int]) -> int:
    if value is not None:
        return int(value)
    return 1024 if pageable_memory_access_uses_host_page_tables() else 4096


def _logical_block_size(path: str | Path) -> int:
    try:
        device = os.stat(path).st_dev
        sysfs_path = Path("/sys/dev/block") / f"{os.major(device)}:{os.minor(device)}"
        candidates = (
            sysfs_path / "queue/logical_block_size",
            sysfs_path.resolve().parent / "queue/logical_block_size",
        )
        value = None
        last_error = None
        for candidate in candidates:
            try:
                value = int(candidate.read_text().strip())
                break
            except (OSError, ValueError) as exc:
                last_error = exc
        if value is None:
            raise OSError(f"logical block size is unavailable: {last_error}")
        if value <= 0 or value & (value - 1):
            raise ValueError(f"invalid logical block size {value}")
        return value
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not determine the logical block size for {path}; expose "
            "/sys/dev/block to the server and use block-backed storage"
        ) from exc


def _write_metadata_page(magic: bytes, metadata: Mapping) -> bytes:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    prefix = struct.pack("<8sI", magic, len(encoded))
    if len(prefix) + len(encoded) > PAGE_BYTES:
        raise ValueError("PLE image metadata does not fit in one 4 KiB block")
    return prefix + encoded + bytes(PAGE_BYTES - len(prefix) - len(encoded))


def _read_metadata_page(path: Path, expected_magic: bytes) -> dict:
    with path.open("rb", buffering=0) as handle:
        block = handle.read(PAGE_BYTES)
    if len(block) != PAGE_BYTES:
        raise IOError(f"short PLE metadata read from {path}")
    magic, length = struct.unpack_from("<8sI", block)
    if magic != expected_magic or length > PAGE_BYTES - 12:
        raise ValueError(f"invalid PLE metadata header in {path}")
    try:
        return json.loads(block[12 : 12 + length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PLE metadata JSON in {path}") from exc


def _validate_manifest_ranges(manifest: list[dict], vocab_end: int, path: Path) -> None:
    ranges = []
    for item in manifest:
        if "row_start" not in item or "row_end" not in item:
            raise ValueError(
                f"PLE image manifest {path} lacks shard row ranges; rebuild it"
            )
        start = int(item["row_start"])
        end = int(item["row_end"])
        if end <= start:
            raise ValueError(
                f"PLE image manifest {path} has an invalid shard range; rebuild it"
            )
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(
                f"PLE image manifest {path} ranges are not contiguous; rebuild it"
            )
        cursor = end
    if cursor < int(vocab_end):
        raise ValueError(
            f"PLE image manifest {path} does not cover [0, {vocab_end}); rebuild it"
        )


def _sanitize_config(value):
    ignored = {
        "ple_storage",
        "ple_disk_dir",
        "ple_disk_hot_cache_gb",
        "ple_disk_hot_frequency_file",
        "ple_disk_stats_log_interval",
        "ple_disk_dynamic_cache_gb",
        "ple_disk_prefill_buffer_tokens",
        "ple_disk_prefill_read_pages",
        "ple_disk_max_read_pages",
    }
    if isinstance(value, dict):
        return {
            key: _sanitize_config(item)
            for key, item in sorted(value.items())
            if key not in ignored
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    return value


def config_digest(config) -> str:
    raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    try:
        payload = json.dumps(
            _sanitize_config(raw),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Qwen4 PLE model configuration is not JSON-serializable"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def checkpoint_fingerprint(config_sha256: str, manifest: Iterable[Mapping]) -> str:
    payload = {
        "config_sha256": config_sha256,
        "shards": sorted(
            (dict(item) for item in manifest), key=lambda item: item["name"]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sample_row_payload(loaded_weight: torch.Tensor) -> str:
    row_count = int(loaded_weight.shape[0])
    sample_count = min(64, row_count)
    if sample_count:
        indices = (
            torch.linspace(
                0,
                row_count - 1,
                sample_count,
                dtype=torch.float64,
                device=loaded_weight.device,
            )
            .round()
            .to(torch.long)
        )
        indices = torch.unique_consecutive(indices)
        sampled = (
            loaded_weight.detach()
            .index_select(0, indices)
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
        )
        index_bytes = indices.to(device="cpu").numpy().astype("<i8").tobytes()
        payload = index_bytes + sampled.tobytes()
    else:
        payload = b""
    return hashlib.sha256(struct.pack("<Q", row_count) + payload).hexdigest()


@dataclass(frozen=True)
class PLEImage:
    path: Path
    crc_path: Path
    header: dict
    checksums: np.ndarray

    @property
    def vocab_start(self) -> int:
        return int(self.header["vocab_start"])

    @property
    def vocab_end(self) -> int:
        return int(self.header["vocab_end"])

    @property
    def num_rows(self) -> int:
        return self.vocab_end - self.vocab_start

    @property
    def num_pages(self) -> int:
        return int(self.header["num_pages"])


@dataclass(frozen=True)
class PLEFetchStats:
    rows_requested: int = 0
    static_hits: int = 0
    dynamic_hits: int = 0
    prefill_hits: int = 0
    cold_pages: int = 0
    coalesced_rows: int = 0


def open_ple_image(
    path: str | Path,
    *,
    expected_fingerprint: Optional[str] = None,
    expected_rank: Optional[int] = None,
) -> PLEImage:
    path = Path(path)
    header = _read_metadata_page(path, IMAGE_MAGIC)
    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"PLE image {path} uses format version "
            f"{header.get('format_version')!r}; rebuild it for format "
            f"version {FORMAT_VERSION}"
        )
    required = {
        "page_bytes": PAGE_BYTES,
        "rows_per_page": ROWS_PER_PAGE,
        "row_bytes": ROW_BYTES,
        "dtype": "float8_e4m3fn",
    }
    for key, expected in required.items():
        if header.get(key) != expected:
            raise ValueError(
                f"PLE image {path} has incompatible {key}: "
                f"{header.get(key)!r} != {expected!r}"
            )
    if (
        expected_fingerprint is not None
        and header.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError(
            f"PLE image fingerprint mismatch: {header.get('fingerprint')} != "
            f"{expected_fingerprint}"
        )
    if expected_rank is not None and int(header.get("rank", -1)) != expected_rank:
        raise ValueError(f"PLE image rank mismatch for {path}")
    num_pages = int(header["num_pages"])
    expected_size = PAGE_BYTES * (num_pages + 1)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise IOError(
            f"PLE image size mismatch/short image: {actual_size} != {expected_size}"
        )
    crc_path = path.with_suffix(".crc32")
    raw = crc_path.read_bytes()
    if len(raw) != 16 + 4 * num_pages:
        raise IOError(f"PLE checksum sidecar has the wrong length: {crc_path}")
    magic, count = struct.unpack_from("<8sQ", raw)
    if magic != CRC_MAGIC or count != num_pages:
        raise ValueError(f"invalid PLE checksum sidecar: {crc_path}")
    checksums = np.frombuffer(raw, dtype="<u4", offset=16, count=num_pages).copy()
    manifest_path = path.parent / "manifest.json"
    try:
        manifest_doc = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PLE image manifest: {manifest_path}") from exc
    manifest = manifest_doc.get("shards")
    if not isinstance(manifest, list):
        raise ValueError(f"PLE image manifest lacks shards: {manifest_path}")
    if manifest_doc.get("fingerprint") != header.get("fingerprint") or manifest_doc.get(
        "config_sha256"
    ) != header.get("config_sha256"):
        raise ValueError(
            f"PLE image sidecars do not match {path}; delete {path.parent} "
            "before rebuilding"
        )
    if manifest_doc.get("module_prefix") != header.get("module_prefix"):
        raise ValueError(
            f"PLE image module identity does not match {path}; delete "
            f"{path.parent} before rebuilding"
        )
    _validate_manifest_ranges(manifest, int(header["vocab_end"]), manifest_path)
    return PLEImage(path, crc_path, header, checksums)


class PLEImageBuilder:
    """Streams checkpoint shards into a TP-local packed image."""

    def __init__(
        self,
        root: str | Path,
        config_sha256: str,
        rank: int,
        tp_size: int,
        vocab_start: int,
        vocab_end: int,
        *,
        module_prefix: str = "ple",
        image_count: int = 1,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = config_sha256
        self.rank = int(rank)
        self.tp_size = int(tp_size)
        self.vocab_start = int(vocab_start)
        self.vocab_end = int(vocab_end)
        self.module_prefix = str(module_prefix)
        self.image_count = int(image_count)
        if self.image_count <= 0:
            raise ValueError("image_count must be positive")
        self.num_rows = self.vocab_end - self.vocab_start
        self.manifest: list[dict] = []
        self.intervals: list[tuple[int, int]] = []
        self._raw_fd: Optional[int] = None
        self._raw_path: Optional[Path] = None
        self._reuse = self._find_reuse_candidate()
        self._expected_manifest = None
        if self._reuse is None:
            num_pages = (self.num_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
            per_rank_bytes = self.num_rows * ROW_BYTES + (num_pages + 1) * PAGE_BYTES
            required_bytes = per_rank_bytes * self.tp_size * self.image_count
            free_bytes = shutil.disk_usage(self.root).free
            if free_bytes < required_bytes:
                raise OSError(
                    errno.ENOSPC,
                    f"PLE image build needs {required_bytes} free bytes in "
                    f"{self.root} for all {self.tp_size} tensor-parallel ranks "
                    f"and {self.image_count} PLE layers, "
                    f"found {free_bytes}",
                )
        if self._reuse is not None:
            manifest_path = self._reuse.path.parent / "manifest.json"
            try:
                manifest_doc = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid PLE image manifest: {manifest_path}"
                ) from exc
            self._expected_manifest = manifest_doc.get("shards")
            if not isinstance(self._expected_manifest, list):
                raise ValueError(f"PLE image manifest lacks shards: {manifest_path}")

    def _find_reuse_candidate(self) -> Optional[PLEImage]:
        matches = []
        for path in self.root.glob(f"*/rank{self.rank}.bin"):
            try:
                header = _read_metadata_page(path, IMAGE_MAGIC)
            except (OSError, ValueError):
                continue
            if (
                header.get("config_sha256") == self.config_sha256
                and int(header.get("tp_size", -1)) == self.tp_size
                and int(header.get("vocab_start", -1)) == self.vocab_start
                and int(header.get("vocab_end", -1)) == self.vocab_end
                and header.get("module_prefix") == self.module_prefix
            ):
                try:
                    image = open_ple_image(path, expected_rank=self.rank)
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"incomplete PLE image state in {path.parent}; delete that "
                        "directory before rebuilding"
                    ) from exc
                matches.append(image)
        if len(matches) > 1:
            raise ValueError(
                "multiple PLE disk images match this module/config/rank; remove "
                f"stale images under {self.root}"
            )
        return matches[0] if matches else None

    def _ensure_raw(self) -> int:
        if self._raw_fd is None:
            fd, name = tempfile.mkstemp(
                prefix=f".rank{self.rank}.", suffix=".rows.tmp", dir=self.root
            )
            path = Path(name)
            try:
                os.ftruncate(fd, self.num_rows * ROW_BYTES)
            except BaseException:
                os.close(fd)
                path.unlink(missing_ok=True)
                raise
            self._raw_fd = fd
            self._raw_path = path
        return self._raw_fd

    def add_shard(
        self, name: str, loaded_weight: torch.Tensor, row_start: int, row_end: int
    ) -> None:
        try:
            self._add_shard(name, loaded_weight, row_start, row_end)
        except BaseException:
            self.close()
            raise

    def _add_shard(
        self, name: str, loaded_weight: torch.Tensor, row_start: int, row_end: int
    ) -> None:
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != ROW_BYTES:
            raise ValueError(
                f"PLE disk storage requires [rows,{ROW_BYTES}] weights, got "
                f"{tuple(loaded_weight.shape)} for {name}"
            )
        if loaded_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"PLE disk storage requires float8_e4m3fn checkpoint rows, got "
                f"{loaded_weight.dtype} for {name}"
            )
        item = {
            "name": name,
            "rows": int(loaded_weight.shape[0]),
            "row_bytes": ROW_BYTES,
            "bytes": int(loaded_weight.numel() * loaded_weight.element_size()),
            "row_start": int(row_start),
            "row_end": int(row_end),
            "sample_sha256": _sample_row_payload(loaded_weight),
        }
        self.manifest.append(item)
        ov_start = max(int(row_start), self.vocab_start)
        ov_end = min(int(row_end), self.vocab_end)
        if ov_start >= ov_end:
            return
        self.intervals.append((ov_start - self.vocab_start, ov_end - self.vocab_start))
        if self._reuse is not None:
            return
        src_start = ov_start - int(row_start)
        src_end = src_start + ov_end - ov_start
        raw = (
            loaded_weight[src_start:src_end]
            .detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
        )
        view = memoryview(raw).cast("B")
        offset = (ov_start - self.vocab_start) * ROW_BYTES
        written = 0
        try:
            fd = self._ensure_raw()
            while written < len(view):
                count = os.pwrite(fd, view[written:], offset + written)
                if count <= 0:
                    raise IOError("short write while materializing PLE row image")
                written += count
        except BaseException:
            self.close()
            raise

    def _validate_coverage(self) -> None:
        merged = []
        for start, end in sorted(self.intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        if merged != [[0, self.num_rows]]:
            raise ValueError(
                f"PLE checkpoint shards do not exactly cover TP rank {self.rank}: {merged}"
            )

    def finalize(self, weight_scale: float) -> tuple[PLEImage, bool, dict]:
        try:
            return self._finalize(weight_scale)
        finally:
            self.close()

    def _finalize(self, weight_scale: float) -> tuple[PLEImage, bool, dict]:
        self._validate_coverage()
        manifest = sorted(self.manifest, key=lambda item: item["name"])
        fingerprint = checkpoint_fingerprint(self.config_sha256, manifest)
        if self._reuse is not None:
            expected = sorted(self._expected_manifest, key=lambda item: item["name"])
            if expected != manifest:
                raise ValueError(
                    "PLE disk image shard manifest mismatch; delete "
                    f"{self._reuse.path.parent} before rebuilding"
                )
            if self._reuse.header.get("fingerprint") != fingerprint:
                raise ValueError(
                    "PLE disk image fingerprint mismatch; delete "
                    f"{self._reuse.path.parent} before rebuilding"
                )
            if float(self._reuse.header.get("weight_scale")) != float(weight_scale):
                raise ValueError(
                    "PLE disk image weight_scale mismatch; delete "
                    f"{self._reuse.path.parent} before rebuilding"
                )
            return (
                self._reuse,
                True,
                {
                    "conversion_seconds": 0.0,
                    "conversion_gib_per_s": 0.0,
                },
            )

        if self._raw_fd is None or self._raw_path is None:
            raise RuntimeError("PLE image builder has no row data")
        os.fsync(self._raw_fd)
        started = time.perf_counter()
        num_pages = (self.num_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
        final_dir = self.root / fingerprint
        final_dir.mkdir(parents=True, exist_ok=True)
        image_path = final_dir / f"rank{self.rank}.bin"
        crc_path = final_dir / f"rank{self.rank}.crc32"
        manifest_path = final_dir / "manifest.json"
        header = {
            "format_version": FORMAT_VERSION,
            "fingerprint": fingerprint,
            "config_sha256": self.config_sha256,
            "module_prefix": self.module_prefix,
            "rank": self.rank,
            "tp_size": self.tp_size,
            "vocab_start": self.vocab_start,
            "vocab_end": self.vocab_end,
            "row_bytes": ROW_BYTES,
            "dtype": "float8_e4m3fn",
            "weight_scale": float(weight_scale),
            "page_bytes": PAGE_BYTES,
            "rows_per_page": ROWS_PER_PAGE,
            "num_pages": num_pages,
            "layout": "page0 metadata; 25 contiguous rows/page; 96-byte zero pad",
        }
        checksums = np.empty(num_pages, dtype="<u4")
        out_fd, tmp_image_name = tempfile.mkstemp(
            prefix=f".rank{self.rank}.bin.", suffix=".tmp", dir=final_dir
        )
        tmp_image = Path(tmp_image_name)
        tmp_crc: Optional[Path] = None
        manifest_tmp: Optional[Path] = None
        try:
            header_page = _write_metadata_page(IMAGE_MAGIC, header)
            if os.write(out_fd, header_page) != len(header_page):
                raise IOError("short metadata write while packing PLE image")
            pages_per_chunk = 1024
            for first_page in range(0, num_pages, pages_per_chunk):
                count = min(pages_per_chunk, num_pages - first_page)
                first_row = first_page * ROWS_PER_PAGE
                rows = min(count * ROWS_PER_PAGE, self.num_rows - first_row)
                raw = os.pread(self._raw_fd, rows * ROW_BYTES, first_row * ROW_BYTES)
                if len(raw) != rows * ROW_BYTES:
                    raise IOError("short read while packing PLE image")
                packed = np.zeros((count, PAGE_BYTES), dtype=np.uint8)
                row_data = np.zeros((count, ROWS_PER_PAGE, ROW_BYTES), dtype=np.uint8)
                row_data.reshape(-1, ROW_BYTES)[:rows] = np.frombuffer(
                    raw, dtype=np.uint8
                ).reshape(rows, ROW_BYTES)
                packed[:, : ROWS_PER_PAGE * ROW_BYTES] = row_data.reshape(
                    count, ROWS_PER_PAGE * ROW_BYTES
                )
                for index in range(count):
                    checksums[first_page + index] = zlib.crc32(packed[index])
                data = memoryview(packed).cast("B")
                written = 0
                while written < len(data):
                    nbytes = os.write(out_fd, data[written:])
                    if nbytes <= 0:
                        raise IOError("short sequential write while packing PLE image")
                    written += nbytes
            os.fsync(out_fd)
        except BaseException:
            tmp_image.unlink(missing_ok=True)
            raise
        finally:
            os.close(out_fd)
        try:
            crc_payload = (
                struct.pack("<8sQ", CRC_MAGIC, num_pages) + checksums.tobytes()
            )
            crc_fd, tmp_crc_name = tempfile.mkstemp(
                prefix=f".rank{self.rank}.crc32.", suffix=".tmp", dir=final_dir
            )
            tmp_crc = Path(tmp_crc_name)
            try:
                if os.write(crc_fd, crc_payload) != len(crc_payload):
                    raise IOError("short write while storing PLE checksums")
                os.fsync(crc_fd)
            finally:
                os.close(crc_fd)

            # Every rank derives this document from the complete shard manifest.
            # Concurrent rank writes therefore install the same payload.
            manifest_doc = {
                "fingerprint": fingerprint,
                "config_sha256": self.config_sha256,
                "module_prefix": self.module_prefix,
                "shards": manifest,
            }
            manifest_payload = (
                json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n"
            ).encode()
            manifest_fd, manifest_tmp_name = tempfile.mkstemp(
                prefix=".manifest.", suffix=".tmp", dir=final_dir
            )
            manifest_tmp = Path(manifest_tmp_name)
            try:
                if os.write(manifest_fd, manifest_payload) != len(manifest_payload):
                    raise IOError("short write while storing the PLE manifest")
                os.fsync(manifest_fd)
            finally:
                os.close(manifest_fd)

            os.replace(tmp_crc, crc_path)
            os.replace(manifest_tmp, manifest_path)
            directory_fd = os.open(final_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(tmp_image, image_path)
            directory_fd = os.open(final_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            elapsed = max(time.perf_counter() - started, 1e-12)
            stats = {
                "conversion_seconds": elapsed,
                "conversion_gib_per_s": (num_pages * PAGE_BYTES) / (1 << 30) / elapsed,
            }
            return (
                open_ple_image(
                    image_path,
                    expected_fingerprint=fingerprint,
                    expected_rank=self.rank,
                ),
                False,
                stats,
            )
        finally:
            tmp_image.unlink(missing_ok=True)
            if tmp_crc is not None:
                tmp_crc.unlink(missing_ok=True)
            if manifest_tmp is not None:
                manifest_tmp.unlink(missing_ok=True)

    def close(self) -> None:
        if self._raw_fd is not None:
            try:
                os.close(self._raw_fd)
            finally:
                self._raw_fd = None
        if self._raw_path is not None:
            try:
                self._raw_path.unlink(missing_ok=True)
            finally:
                self._raw_path = None


def build_test_image(
    root: str | Path,
    rows: torch.Tensor,
    *,
    rank: int = 0,
    tp_size: int = 1,
    vocab_start: int = 0,
    config_sha256: str = "test-config",
    weight_scale: float = 1.0,
    module_prefix: str = "ple",
) -> PLEImage:
    builder = PLEImageBuilder(
        root,
        config_sha256,
        rank,
        tp_size,
        vocab_start,
        vocab_start + rows.shape[0],
        module_prefix=module_prefix,
    )
    builder.add_shard(
        "test.shard_0.weight", rows, vocab_start, vocab_start + rows.shape[0]
    )
    image, _, _ = builder.finalize(weight_scale)
    return image


def write_hot_frequency_file(
    path: str | Path,
    ids_by_rank: Mapping[int, np.ndarray],
    *,
    fingerprint: str,
    total_rows: int,
    tp_size: int,
    padding_divisor: int,
) -> None:
    if not fingerprint:
        raise ValueError("PLE hot-frequency files require an image fingerprint")
    total_rows = int(total_rows)
    tp_size = int(tp_size)
    padding_divisor = int(padding_divisor)
    if total_rows <= 0 or tp_size <= 0 or padding_divisor <= 0:
        raise ValueError("PLE hot-frequency geometry values must be positive")
    padded_rows = (
        (total_rows + padding_divisor - 1) // padding_divisor * padding_divisor
    )
    if padded_rows % tp_size:
        raise ValueError(
            f"padded PLE row count {padded_rows} is not divisible by TP={tp_size}"
        )
    if set(ids_by_rank) != set(range(tp_size)):
        raise ValueError(
            f"PLE hot-frequency file requires ranks [0, {tp_size}); "
            f"got {sorted(ids_by_rank)}"
        )
    path = Path(path)
    ranks = []
    arrays = []
    offset = 0
    rows_per_rank = padded_rows // tp_size
    for rank, ids in sorted(ids_by_rank.items()):
        array = np.asarray(ids, dtype="<u4")
        if np.unique(array).size != array.size:
            raise ValueError(f"duplicate PLE hot row id in rank {rank}")
        vocab_start = int(rank) * rows_per_rank
        vocab_end = min(total_rows, vocab_start + rows_per_rank)
        if np.any((array < vocab_start) | (array >= vocab_end)):
            raise ValueError(
                f"PLE hot row id is outside rank {rank} range "
                f"[{vocab_start}, {vocab_end})"
            )
        ranks.append(
            {
                "rank": int(rank),
                "offset": offset,
                "count": int(array.size),
                "vocab_start": vocab_start,
                "vocab_end": vocab_end,
            }
        )
        arrays.append(array)
        offset += int(array.size)
    header = {
        "format_version": HOT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "dtype": "uint32-global-row-id",
        "total_rows": total_rows,
        "tp_size": tp_size,
        "padding_divisor": padding_divisor,
        "ranks": ranks,
    }
    with path.open("wb") as handle:
        handle.write(_write_metadata_page(HOT_MAGIC, header))
        for array in arrays:
            handle.write(array.tobytes())


def read_hot_frequency_file(
    path: str | Path,
    rank: int,
    *,
    expected_fingerprint: str = "",
    expected_tp_size: Optional[int] = None,
    expected_vocab_start: Optional[int] = None,
    expected_vocab_end: Optional[int] = None,
) -> np.ndarray:
    path = Path(path)
    header = _read_metadata_page(path, HOT_MAGIC)
    if header.get("format_version") != HOT_FORMAT_VERSION:
        raise ValueError("PLE hot-frequency file format version mismatch")
    if header.get("dtype") != "uint32-global-row-id":
        raise ValueError("PLE hot-frequency file dtype mismatch")
    fingerprint = header.get("fingerprint", "")
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError("PLE hot-frequency file fingerprint mismatch")
    if expected_tp_size is not None and int(header.get("tp_size", -1)) != int(
        expected_tp_size
    ):
        raise ValueError("PLE hot-frequency file tensor-parallel size mismatch")
    entry = next(
        (row for row in header.get("ranks", []) if int(row["rank"]) == rank), None
    )
    if entry is None:
        return np.empty(0, dtype=np.uint32)
    if expected_vocab_start is not None and int(entry.get("vocab_start", -1)) != int(
        expected_vocab_start
    ):
        raise ValueError("PLE hot-frequency file rank start mismatch")
    if expected_vocab_end is not None and int(entry.get("vocab_end", -1)) != int(
        expected_vocab_end
    ):
        raise ValueError("PLE hot-frequency file rank end mismatch")
    offset = PAGE_BYTES + int(entry["offset"]) * 4
    count = int(entry["count"])
    payload_bytes = path.stat().st_size - PAGE_BYTES
    if offset < PAGE_BYTES or count < 0 or offset - PAGE_BYTES > payload_bytes:
        raise ValueError("invalid PLE hot-frequency rank range")
    byte_count = count * 4
    if byte_count > payload_bytes - (offset - PAGE_BYTES):
        raise ValueError("PLE hot-frequency rank range exceeds the file size")
    with path.open("rb", buffering=0) as handle:
        handle.seek(offset)
        raw = handle.read(byte_count)
    if len(raw) != byte_count:
        raise IOError("short PLE hot-frequency file read")
    return np.frombuffer(raw, dtype="<u4").copy()


_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


class RankSelectHotCache:
    """Exact bitmap membership plus O(1) rank for packed pinned hot rows."""

    def __init__(
        self,
        image: PLEImage,
        global_ids: np.ndarray,
        budget_gb: float,
        reader=None,
    ):
        capacity = int(float(budget_gb) * (1 << 30) // ROW_BYTES)
        ids = np.asarray(global_ids, dtype=np.int64)
        ids = ids[(ids >= image.vocab_start) & (ids < image.vocab_end)]
        if ids.size:
            _, first_positions = np.unique(ids, return_index=True)
            ids = ids[np.sort(first_positions)]
        if ids.size > capacity:
            ids = ids[:capacity]
        local = np.sort((ids - image.vocab_start).astype(np.int64))
        logger.info(
            "PLE disk static cache rank=%d budget_rows=%d realized_rows=%d",
            int(image.header["rank"]),
            capacity,
            local.size,
        )
        expected_rows = min(capacity, image.num_rows)
        if global_ids.size and expected_rows and local.size * 10 < expected_rows:
            logger.warning(
                "PLE disk static cache rank=%d filled less than 10%% of its row "
                "budget (%d of %d); check the hot-file TP geometry and corpus",
                int(image.header["rank"]),
                local.size,
                expected_rows,
            )
        words = (image.num_rows + 63) // 64
        self.bitmap = np.zeros(words, dtype=np.uint64)
        if local.size:
            word_indices = local // 64
            starts = np.r_[0, np.flatnonzero(np.diff(word_indices)) + 1]
            masks = np.left_shift(np.uint64(1), (local % 64).astype(np.uint64))
            self.bitmap[word_indices[starts]] = np.bitwise_or.reduceat(masks, starts)
        # SGLang targets little-endian Linux hosts. The byte view maps bit zero
        # to the first byte of each uint64 word for both popcount paths.
        counts = _POPCOUNT[self.bitmap.view(np.uint8).reshape(-1, 8)].sum(
            1, dtype=np.uint32
        )
        self.rank_prefix = np.empty(words + 1, dtype=np.uint32)
        self.rank_prefix[0] = 0
        np.cumsum(counts, dtype=np.uint32, out=self.rank_prefix[1:])
        self.rows = _allocate_host_tensor((local.size, ROW_BYTES), dtype=torch.uint8)
        if local.size:
            self._load_rows(image, local, reader)

    def _load_rows(self, image: PLEImage, local: np.ndarray, reader) -> None:
        target = self.rows.numpy()
        page_ids = np.unique(local // ROWS_PER_PAGE)
        if reader is not None:
            for begin in range(0, page_ids.size, reader.max_pages):
                selected_pages = page_ids[begin : begin + reader.max_pages]
                with reader.locked_pages(selected_pages) as pages:
                    first_row = selected_pages[0] * ROWS_PER_PAGE
                    last_row = min(
                        image.num_rows,
                        (selected_pages[-1] + 1) * ROWS_PER_PAGE,
                    )
                    first = int(np.searchsorted(local, first_row, side="left"))
                    last = int(np.searchsorted(local, last_row, side="left"))
                    selected = local[first:last]
                    page_index = np.searchsorted(
                        selected_pages, selected // ROWS_PER_PAGE
                    )
                    within = selected % ROWS_PER_PAGE
                    byte_index = (
                        within[:, None] * ROW_BYTES
                        + np.arange(ROW_BYTES, dtype=np.int64)[None, :]
                    )
                    target[first:last] = pages[page_index[:, None], byte_index]
            return

        fd = os.open(image.path, os.O_RDONLY)
        try:
            cursor = 0
            for page_id in page_ids:
                end_row = min(image.num_rows, (int(page_id) + 1) * ROWS_PER_PAGE)
                end = int(np.searchsorted(local, end_row, side="left"))
                raw = os.pread(
                    fd,
                    PAGE_BYTES,
                    (int(page_id) + 1) * PAGE_BYTES,
                )
                if len(raw) != PAGE_BYTES:
                    raise IOError("short read while loading PLE hot cache")
                page = np.frombuffer(raw, dtype=np.uint8)
                selected = local[cursor:end]
                if zlib.crc32(page) != int(image.checksums[int(page_id)]):
                    raise IOError(
                        f"PLE checksum mismatch on hot-cache page {int(page_id)}"
                    )
                within = selected % ROWS_PER_PAGE
                byte_index = (
                    within[:, None] * ROW_BYTES
                    + np.arange(ROW_BYTES, dtype=np.int64)[None, :]
                )
                target[cursor:end] = page[byte_index]
                cursor = end
            if cursor != local.size:
                raise IOError("PLE hot-cache image scan did not cover every row")
        finally:
            os.close(fd)

    def lookup(self, local_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_ids = np.asarray(local_ids, dtype=np.int64)
        valid = (local_ids >= 0) & (local_ids < (self.bitmap.size * 64))
        safe = np.where(valid, local_ids, 0)
        word_index = safe // 64
        bit = (safe % 64).astype(np.uint64)
        words = self.bitmap[word_index]
        hit = valid & (((words >> bit) & np.uint64(1)) != 0)
        lower = (np.left_shift(np.uint64(1), bit) - np.uint64(1)) & words
        within = _POPCOUNT[lower.view(np.uint8).reshape(-1, 8)].sum(1, dtype=np.uint32)
        slots = self.rank_prefix[word_index].astype(np.int64) + within.astype(np.int64)
        slots[~hit] = -1
        return hit, slots


class WTinyLFURowCache:
    """Bounded exact row cache with asynchronous W-TinyLFU admission."""

    _WAYS = 8
    _SKETCH_DEPTH = 4
    _LOOKUP_MAX_RETRIES = 8
    _HASH_MIX = np.array(
        [
            0x9E3779B185EBCA87,
            0xC2B2AE3D27D4EB4F,
            0x165667B19E3779F9,
            0x85EBCA77C2B2AE63,
        ],
        dtype=np.uint64,
    )

    def __init__(
        self,
        budget_gb: float = 0.0,
        *,
        capacity_rows: Optional[int] = None,
        queue_batches: int = 64,
    ) -> None:
        requested = (
            int(capacity_rows)
            if capacity_rows is not None
            else int(float(budget_gb) * (1 << 30) // ROW_BYTES)
        )
        self.num_sets = max(0, requested) // self._WAYS
        self.num_slots = self.num_sets * self._WAYS
        self.capacity = self.num_slots
        self.tags = np.full((self.num_sets, self._WAYS), -1, dtype=np.int64)
        self.recency = np.zeros((self.num_sets, self._WAYS), dtype=np.uint64)
        self._versions = np.zeros(self.num_sets, dtype=np.uint64)
        self.rows = _allocate_host_tensor(
            (self.num_slots, ROW_BYTES), dtype=torch.uint8
        )
        sketch_target = max(1024, min(1 << 20, max(1, self.capacity // 8)))
        self.sketch_width = 1 << (sketch_target - 1).bit_length()
        self.sketch = np.zeros((self._SKETCH_DEPTH, self.sketch_width), dtype=np.uint8)
        self._clock = 0
        self._sample_count = 0
        self._reset_interval = max(1, self.capacity * 10)
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_batches)))
        self._pending_condition = threading.Condition()
        self._pending_batches = 0
        self._closed = False
        self._dropped_batches = 0
        self._worker_error: Optional[BaseException] = None
        self._worker = None
        if self.capacity:
            self._worker = threading.Thread(
                target=self._admission_loop,
                name="ple-wtinylfu",
                daemon=True,
            )
            self._worker.start()

    def _set_indices(self, ids: np.ndarray) -> np.ndarray:
        values = np.asarray(ids, dtype=np.uint64)
        mixed = values * self._HASH_MIX[0]
        mixed ^= mixed >> np.uint64(33)
        return (mixed % np.uint64(self.num_sets)).astype(np.int64)

    def _sketch_indices(self, row_id: int) -> np.ndarray:
        value = np.uint64(row_id)
        mixed = value * self._HASH_MIX
        mixed ^= mixed >> np.uint64(29)
        return (mixed & np.uint64(self.sketch_width - 1)).astype(np.int64)

    def _frequency(self, row_id: int) -> int:
        columns = self._sketch_indices(row_id)
        return int(
            min(
                self.sketch[depth, columns[depth]]
                for depth in range(self._SKETCH_DEPTH)
            )
        )

    def _increment(self, row_id: int) -> None:
        columns = self._sketch_indices(row_id)
        for depth, column in enumerate(columns):
            value = int(self.sketch[depth, column])
            if value < 255:
                self.sketch[depth, column] = value + 1
        self._sample_count += 1
        if self._sample_count >= self._reset_interval:
            self.sketch >>= np.uint8(1)
            self._sample_count = 0

    def lookup_into(
        self,
        local_ids: np.ndarray,
        output: np.ndarray,
        *,
        record_hits: bool = True,
    ) -> np.ndarray:
        ids = np.asarray(local_ids, dtype=np.int64).reshape(-1)
        hit = np.zeros(ids.size, dtype=np.bool_)
        if not self.capacity or not ids.size:
            return hit
        pending = np.arange(ids.size)
        cached_rows = self.rows.numpy()
        for attempt in range(self._LOOKUP_MAX_RETRIES):
            sets = self._set_indices(ids[pending])
            # These numpy version and tag operations retain the GIL. An odd
            # version marks a set while its cached row is being updated. A
            # free-threaded Python build needs acquire and release atomics for
            # these version and tag accesses.
            before = self._versions[sets].copy()
            ready = (before & np.uint64(1)) == 0
            candidates = self.tags[sets].copy()
            matches = candidates == ids[pending, None]
            matched = matches.any(axis=1)
            if np.any(ready & matched):
                selected = np.flatnonzero(ready & matched)
                ways = matches[selected].argmax(axis=1)
                slots = sets[selected] * self._WAYS + ways
                output[pending[selected]] = cached_rows[slots]
            after = self._versions[sets]
            stable = ready & (before == after)
            hit[pending[stable]] = matched[stable]
            pending = pending[~stable]
            if not pending.size:
                break
            if attempt + 1 < self._LOOKUP_MAX_RETRIES:
                time.sleep(0)
        if record_hits and np.any(hit):
            self.record(ids[hit], None)
        return hit

    def record(self, local_ids: np.ndarray, exact_rows: Optional[np.ndarray]) -> None:
        if not self.capacity or self._closed:
            return
        ids = np.asarray(local_ids, dtype=np.int64).reshape(-1).copy()
        rows = (
            None
            if exact_rows is None
            else np.asarray(exact_rows, dtype=np.uint8).reshape(-1, ROW_BYTES).copy()
        )
        with self._pending_condition:
            self._pending_batches += 1
        try:
            self._queue.put_nowait((ids, rows))
        except queue.Full:
            with self._pending_condition:
                self._pending_batches -= 1
                if self._pending_batches == 0:
                    self._pending_condition.notify_all()
            with self._lock:
                self._dropped_batches += 1
                dropped = self._dropped_batches
            if dropped == 1 or dropped & (dropped - 1) == 0:
                logger.warning(
                    "PLE disk dynamic cache dropped %d admission batches because "
                    "its queue was full",
                    dropped,
                )

    def _insert(self, row_id: int, exact_row: np.ndarray) -> None:
        set_index = int(self._set_indices(np.array([row_id]))[0])
        self._versions[set_index] = np.uint64(int(self._versions[set_index]) + 1)
        try:
            tags = self.tags[set_index]
            existing = np.flatnonzero(tags == row_id)
            self._clock += 1
            if existing.size:
                self.recency[set_index, int(existing[0])] = self._clock
                return

            # One of eight ways is the admission window. A smaller canonical
            # window cannot be represented by this fixed eight-way layout.
            window_way = 0
            candidate_id = int(tags[window_way])
            candidate_row = None
            if candidate_id >= 0:
                slot = set_index * self._WAYS + window_way
                candidate_row = self.rows.numpy()[slot].copy()
            tags[window_way] = row_id
            self.rows[set_index * self._WAYS + window_way].copy_(
                torch.from_numpy(exact_row)
            )
            self.recency[set_index, window_way] = self._clock
            if candidate_id < 0:
                return

            main_tags = tags[1:]
            empty = np.flatnonzero(main_tags < 0)
            if empty.size:
                victim_way = int(empty[0]) + 1
            else:
                frequencies = np.array(
                    [self._frequency(int(tag)) for tag in main_tags], dtype=np.int32
                )
                minimum = frequencies.min()
                tied = np.flatnonzero(frequencies == minimum) + 1
                victim_way = int(tied[np.argmin(self.recency[set_index, tied])])
                victim_id = int(tags[victim_way])
                if self._frequency(candidate_id) < self._frequency(victim_id):
                    return
            tags[victim_way] = candidate_id
            self.rows[set_index * self._WAYS + victim_way].copy_(
                torch.from_numpy(candidate_row)
            )
            self.recency[set_index, victim_way] = self._clock
        finally:
            self._versions[set_index] = np.uint64(int(self._versions[set_index]) + 1)

    def _admission_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                ids, rows = item
                with self._lock:
                    for index, row_id in enumerate(ids):
                        value = int(row_id)
                        self._increment(value)
                        if rows is not None:
                            self._insert(value, rows[index])
            except BaseException as exc:
                with self._lock:
                    if self._worker_error is None:
                        self._worker_error = exc
                logger.exception("PLE disk dynamic cache admission failed")
            finally:
                self._queue.task_done()
                if item is not None:
                    with self._pending_condition:
                        self._pending_batches -= 1
                        if self._pending_batches == 0:
                            self._pending_condition.notify_all()

    def flush(self, timeout: float = 5.0) -> None:
        if self.capacity and not self._closed:
            deadline = time.monotonic() + timeout
            with self._pending_condition:
                while self._pending_batches:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "PLE disk dynamic cache admission did not drain within "
                            f"{timeout:.1f}s"
                        )
                    self._pending_condition.wait(remaining)
            with self._lock:
                worker_error = self._worker_error
            if worker_error is not None:
                raise RuntimeError(
                    "PLE disk dynamic cache admission failed"
                ) from worker_error

    def close(self) -> None:
        if not self.capacity or self._closed:
            return
        error = None
        try:
            self.flush()
        except BaseException as exc:
            error = exc
        self._closed = True
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full as exc:
            error = error or RuntimeError(
                "PLE disk dynamic cache could not enqueue its shutdown marker"
            )
            if error is not exc:
                error.__cause__ = exc
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            error = error or RuntimeError(
                "PLE disk dynamic cache worker did not stop within 5 seconds"
            )
        if error is not None:
            raise error


def _installed_sgl_kernel_version() -> Optional[str]:
    try:
        return importlib.metadata.version("sglang-kernel")
    except importlib.metadata.PackageNotFoundError:
        return None


def _find_helper_library() -> Path:
    spec = importlib.util.find_spec("sgl_kernel")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    installed = _installed_sgl_kernel_version()
    try:
        required = Version(REQUIRED_SGL_KERNEL_VERSION)
        installed_version = Version(installed) if installed is not None else None
        old_wheel = installed_version is None or installed_version < required
    except InvalidVersion:
        old_wheel = True
    if old_wheel:
        found_text = installed or "no installed distribution"
        raise RuntimeError(
            "PLE disk storage requires sglang-kernel >= "
            f"{REQUIRED_SGL_KERNEL_VERSION}; found {found_text}. Install the matching "
            "sglang-kernel wheel"
        )
    for location in locations:
        package_dir = Path(location)
        candidates = sorted(package_dir.glob("qwen4_ple_disk_fetcher*.so"))
        if candidates:
            return candidates[0]
    markers = [
        package_dir / "qwen4_ple_disk_fetcher.build"
        for package_dir in map(Path, locations)
    ]
    enabled_marker = any(
        marker.is_file() and marker.read_text().strip() == "enabled"
        for marker in markers
    )
    if enabled_marker:
        raise RuntimeError(
            f"sglang-kernel {installed} reports Qwen4 PLE io_uring support, but "
            "qwen4_ple_disk_fetcher.so is missing; reinstall the wheel"
        )
    raise RuntimeError(
        f"sglang-kernel {installed} was built without Qwen4 PLE io_uring "
        "support; install a Linux wheel with the disk fetcher enabled"
    )


class DirectPageReader:
    """Direct page reader with serialized access to its staging area."""

    def __init__(self, image: PLEImage, max_pages: int = 4096):
        self.image = image
        self.max_pages = int(max_pages)
        if not 1 <= self.max_pages <= IORING_MAX_ENTRIES:
            raise ValueError(
                f"max_pages must be between 1 and {IORING_MAX_ENTRIES}, got "
                f"{self.max_pages}"
            )
        self.register_buffer = not pageable_memory_access_uses_host_page_tables()
        staging_bytes = self.max_pages * PAGE_BYTES
        logical_block_size = _logical_block_size(image.path)
        if logical_block_size > PAGE_BYTES:
            raise RuntimeError(
                "PLE disk images require storage with 4096-byte logical blocks; "
                f"{image.path} reports {logical_block_size} bytes"
            )
        self.alignment = PAGE_BYTES
        self._staging_allocation = _allocate_host_tensor(
            staging_bytes + self.alignment - 1,
            dtype=torch.uint8,
            pin_memory=self.register_buffer,
        )
        alignment_offset = (-self._staging_allocation.data_ptr()) & (self.alignment - 1)
        self.staging = self._staging_allocation[
            alignment_offset : alignment_offset + staging_bytes
        ]
        self.result = np.empty((self.max_pages, PAGE_BYTES), dtype=np.uint8)
        self.offsets = np.empty(self.max_pages, dtype=np.uint64)
        self._read_lock = threading.Lock()
        if self.staging.data_ptr() & (self.alignment - 1):
            raise RuntimeError("PLE registered staging buffer is not O_DIRECT aligned")
        self.lib = ctypes.CDLL(str(_find_helper_library()), use_errno=True)
        self.lib.ple_fetcher_create.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ple_fetcher_create.restype = ctypes.c_void_p
        self.lib.ple_fetcher_read.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ple_fetcher_read.restype = ctypes.c_int
        self.lib.ple_fetcher_last_error.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.ple_fetcher_last_error.restype = ctypes.c_int
        self.lib.ple_fetcher_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ple_fetcher_destroy.restype = ctypes.c_int
        self.fd = None
        self.handle = None
        self.fd = os.open(image.path, os.O_RDONLY | os.O_DIRECT)
        failure_stage = ctypes.c_int()
        self.handle = self.lib.ple_fetcher_create(
            self.fd,
            self.staging.data_ptr(),
            self.staging.numel(),
            self.max_pages,
            self.register_buffer,
            ctypes.byref(failure_stage),
        )
        if not self.handle:
            error = ctypes.get_errno()
            os.close(self.fd)
            self.fd = None
            if error == errno.ENOMEM and (
                failure_stage.value == FETCHER_FAILURE_REGISTER_BUFFER
            ):
                raise OSError(
                    error,
                    "PLE io_uring buffer registration exceeded RLIMIT_MEMLOCK "
                    f"while registering {self.staging.numel()} bytes; raise the "
                    "memlock ulimit or lower --ple-disk-max-read-pages",
                )
            if error in (errno.EPERM, errno.EACCES, errno.ENOSYS) and (
                failure_stage.value == FETCHER_FAILURE_SETUP
            ):
                raise OSError(
                    error,
                    "PLE io_uring is blocked; common causes are the default "
                    "container seccomp policy and kernel.io_uring_disabled. "
                    "Allow io_uring or use --ple-storage pinned",
                )
            raise OSError(error, os.strerror(error))

    def read(self, page_ids: np.ndarray) -> np.ndarray:
        with self.locked_pages(page_ids) as pages:
            return pages.copy()

    @contextmanager
    def locked_pages(self, page_ids: np.ndarray):
        if not self._read_lock.acquire(blocking=False):
            raise RuntimeError("PLE direct page reader supports one caller at a time")
        try:
            yield self._read_locked(page_ids)
        finally:
            self._read_lock.release()

    def _read_locked(self, page_ids: np.ndarray) -> np.ndarray:
        page_ids = np.asarray(page_ids, dtype=np.int64)
        if np.any((page_ids < 0) | (page_ids >= self.image.num_pages)):
            raise IndexError("PLE page id outside image")
        result = (
            self.result[: page_ids.size]
            if page_ids.size <= self.max_pages
            else np.empty((page_ids.size, PAGE_BYTES), dtype=np.uint8)
        )
        for begin in range(0, page_ids.size, self.max_pages):
            chunk = page_ids[begin : begin + self.max_pages]
            offsets = self.offsets[: chunk.size]
            np.add(chunk, 1, out=offsets, casting="unsafe")
            offsets *= PAGE_BYTES
            rc = self.lib.ple_fetcher_read(
                self.handle,
                offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                offsets.size,
                self.staging.data_ptr(),
                self.staging.numel(),
            )
            if rc:
                if -rc == FETCHER_ERR_POISONED:
                    raise RuntimeError(
                        "PLE disk fetcher is poisoned after an I/O drain failure; "
                        "restart the fetcher by restarting the server"
                    )
                failed_index = ctypes.c_uint()
                io_result = ctypes.c_int()
                has_detail = self.lib.ple_fetcher_last_error(
                    self.handle,
                    ctypes.byref(failed_index),
                    ctypes.byref(io_result),
                )
                if has_detail < 0:
                    raise OSError(
                        -has_detail,
                        "PLE disk fetcher error details are unavailable: "
                        f"{os.strerror(-has_detail)}",
                    )
                if has_detail and failed_index.value < chunk.size:
                    page_id = int(chunk[failed_index.value])
                    if io_result.value >= 0:
                        detail = f"short read of {io_result.value} bytes on PLE page {page_id}"
                    else:
                        detail = f"PLE page {page_id}: {os.strerror(-io_result.value)}"
                    raise OSError(-rc, detail)
                raise OSError(-rc, os.strerror(-rc))
            pages = (
                self.staging[: offsets.size * PAGE_BYTES]
                .numpy()
                .reshape(-1, PAGE_BYTES)
            )
            crc = np.fromiter(
                (zlib.crc32(page) for page in pages),
                dtype=np.uint32,
                count=pages.shape[0],
            )
            expected = self.image.checksums[chunk]
            mismatch = np.flatnonzero(crc != expected)
            if mismatch.size:
                raise IOError(
                    f"PLE checksum mismatch on page {int(chunk[mismatch[0]])}"
                )
            result[begin : begin + offsets.size] = pages
        return result

    def close(self) -> None:
        with self._read_lock:
            if getattr(self, "handle", None):
                rc = self.lib.ple_fetcher_destroy(self.handle)
                self.handle = None
                try:
                    if rc:
                        raise OSError(
                            -rc,
                            "PLE disk fetcher shutdown failed: " f"{os.strerror(-rc)}",
                        )
                finally:
                    if self.fd is not None:
                        os.close(self.fd)
                        self.fd = None
            elif getattr(self, "fd", None) is not None:
                os.close(self.fd)
                self.fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DiskRowFetcher:
    def __init__(
        self,
        image: PLEImage,
        *,
        hot_frequency_file: Optional[str] = None,
        hot_cache_gb: float = 8.0,
        dynamic_cache_gb: float = 0.0,
        dynamic_capacity_rows: Optional[int] = None,
        prefill_buffer_tokens: int = 0,
        prefill_read_pages: int = 128,
        max_pages: int = 4096,
        ngram_heads: int = 16,
    ) -> None:
        self.image = image
        ids = (
            read_hot_frequency_file(
                hot_frequency_file,
                int(image.header["rank"]),
                expected_fingerprint=str(image.header["fingerprint"]),
                expected_tp_size=int(image.header["tp_size"]),
                expected_vocab_start=image.vocab_start,
                expected_vocab_end=image.vocab_end,
            )
            if hot_frequency_file
            else np.empty(0, dtype=np.uint32)
        )
        self.reader = DirectPageReader(image, max_pages=max_pages)
        self.prefill_reader = None
        self.hot = None
        self.dynamic = None
        self._prefill_executor = None
        self._closed = False
        try:
            self.hot = RankSelectHotCache(image, ids, hot_cache_gb, reader=self.reader)
            self.prefill_reader = (
                DirectPageReader(image, max_pages=max(1, int(prefill_read_pages)))
                if prefill_buffer_tokens > 0
                else None
            )
            self.dynamic = WTinyLFURowCache(
                dynamic_cache_gb, capacity_rows=dynamic_capacity_rows
            )
            self._thread_stats = threading.local()
            self._prefill_lock = threading.Lock()
            self._prefill_sequence = 0
            self._prefill_max_rows = max(0, int(prefill_buffer_tokens)) * int(
                ngram_heads
            )
            self._prefill_slots = []
            if self._prefill_max_rows:
                for _ in range(2):
                    self._prefill_slots.append(
                        {
                            "state": "empty",
                            "sequence": 0,
                            "count": 0,
                            "ids": np.empty(self._prefill_max_rows, dtype=np.int64),
                            "rows": _allocate_host_tensor(
                                (self._prefill_max_rows, ROW_BYTES),
                                dtype=torch.uint8,
                            ),
                        }
                    )
            executor_kwargs = {
                "max_workers": 1,
                "thread_name_prefix": "ple-prefill",
            }
            if torch.cuda.is_available():
                executor_kwargs.update(
                    initializer=_set_cuda_device,
                    initargs=(_current_cuda_device(),),
                )
            self._prefill_executor = (
                ThreadPoolExecutor(**executor_kwargs) if self._prefill_slots else None
            )
            self._prefill_futures: set[Future] = set()
            self._prefill_disabled = False
            self._prefill_disable_done = threading.Event()
            self._prefill_truncated_submissions = 0
        except BaseException:
            if self._prefill_executor is not None:
                self._prefill_executor.shutdown(wait=True)
            if self.dynamic is not None:
                self.dynamic.close()
            if self.prefill_reader is not None:
                self.prefill_reader.close()
            self.reader.close()
            raise

    @property
    def last_fetch_stats(self) -> PLEFetchStats:
        return getattr(self._thread_stats, "last", PLEFetchStats())

    @last_fetch_stats.setter
    def last_fetch_stats(self, value: PLEFetchStats) -> None:
        self._thread_stats.last = value

    def _lookup_prefill(self, global_ids: np.ndarray, output: np.ndarray) -> np.ndarray:
        ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        hit = np.zeros(ids.size, dtype=np.bool_)
        if not self._prefill_slots or not ids.size:
            return hit
        with self._prefill_lock:
            for slot in self._prefill_slots:
                if slot["state"] != "ready" or slot["count"] == 0:
                    continue
                pending = np.flatnonzero(~hit)
                if not pending.size:
                    break
                slot_ids = slot["ids"][: slot["count"]]
                positions = np.searchsorted(slot_ids, ids[pending])
                valid = positions < slot["count"]
                matches = np.zeros(pending.size, dtype=np.bool_)
                matches[valid] = slot_ids[positions[valid]] == ids[pending[valid]]
                if np.any(matches):
                    selected = pending[matches]
                    output[selected] = slot["rows"].numpy()[positions[matches]]
                    hit[selected] = True
        return hit

    def submit_prefill(self, global_ids: np.ndarray) -> bool:
        if not self._prefill_slots or self._prefill_disabled:
            return False
        ids = np.asarray(global_ids, dtype=np.int64).reshape(-1)
        ids = ids[(ids >= self.image.vocab_start) & (ids < self.image.vocab_end)]
        _, first_positions = np.unique(ids, return_index=True)
        ids = ids[np.sort(first_positions)]
        if not ids.size:
            return False
        if ids.size > self._prefill_max_rows:
            dropped = ids.size - self._prefill_max_rows
            self._prefill_truncated_submissions += 1
            count = self._prefill_truncated_submissions
            if count == 1 or count & (count - 1) == 0:
                logger.warning(
                    "PLE prefill look-ahead omitted %d rows because its token "
                    "buffer is too small; occurrence=%d",
                    dropped,
                    count,
                )
            ids = ids[: self._prefill_max_rows]
        ids = np.sort(ids)
        with self._prefill_lock:
            executor = self._prefill_executor
            if not self._prefill_slots or self._prefill_disabled or executor is None:
                return False
            available = [
                (index, slot)
                for index, slot in enumerate(self._prefill_slots)
                if slot["state"] != "filling"
            ]
            if not available:
                return False
            empty = [item for item in available if item[1]["state"] == "empty"]
            index, slot = (
                empty[0]
                if empty
                else min(available, key=lambda item: item[1]["sequence"])
            )
            self._prefill_sequence += 1
            sequence = self._prefill_sequence
            slot["state"] = "filling"
            slot["sequence"] = sequence
            slot["count"] = 0
            try:
                future = executor.submit(
                    self._fill_prefill_slot, slot, sequence, ids.copy()
                )
            except BaseException:
                slot["state"] = "empty"
                raise
            self._prefill_futures.add(future)
        future.add_done_callback(self._prefill_done)
        return True

    def _prefill_done(self, future: Future) -> None:
        error = future.exception()
        if error is not None:
            self._disable_prefill(error)
        with self._prefill_lock:
            self._prefill_futures.discard(future)

    def _disable_prefill(self, error: BaseException) -> None:
        with self._prefill_lock:
            if self._prefill_disabled:
                owns_disable = False
                disable_done = self._prefill_disable_done
                reader = None
                executor = None
            else:
                owns_disable = True
                self._prefill_disabled = True
                disable_done = self._prefill_disable_done
                for slot in self._prefill_slots:
                    slot["state"] = "empty"
                    slot["count"] = 0
                self._prefill_slots = []
                reader = self.prefill_reader
                self.prefill_reader = None
                executor = self._prefill_executor
                self._prefill_executor = None
        if not owns_disable:
            disable_done.wait()
            return
        try:
            logger.error(
                "PLE prefill look-ahead failed and has been disabled; decode reads "
                "will continue through the CRC-checked decode reader",
                exc_info=(type(error), error, error.__traceback__),
            )
            if reader is not None:
                reader.close()
            if executor is not None:
                executor.shutdown(wait=False)
        finally:
            disable_done.set()

    def _fill_prefill_slot(self, slot: dict, sequence: int, ids: np.ndarray) -> None:
        try:
            count = ids.size
            rows = self.fetch(
                ids,
                out=slot["rows"][:count],
                priority="prefill",
                use_prefill=False,
                admit_dynamic=False,
            ).reshape(-1, ROW_BYTES)
            slot["ids"][:count] = ids
            if rows.data_ptr() != slot["rows"].data_ptr():
                raise RuntimeError("PLE prefill fetch did not reuse its slot buffer")
            with self._prefill_lock:
                if slot["sequence"] != sequence or slot["state"] != "filling":
                    return
                slot["count"] = count
                slot["state"] = "ready"
        except BaseException:
            with self._prefill_lock:
                if slot["sequence"] == sequence:
                    slot["state"] = "empty"
                    slot["count"] = 0
            raise

    def wait_prefill(self) -> None:
        while True:
            with self._prefill_lock:
                futures = list(self._prefill_futures)
            if not futures:
                break
            for future in futures:
                try:
                    future.result()
                except BaseException as exc:
                    self._disable_prefill(exc)
            with self._prefill_lock:
                self._prefill_futures.difference_update(futures)

    def fetch(
        self,
        global_ids: np.ndarray,
        out: Optional[torch.Tensor] = None,
        *,
        priority: str = "decode",
        use_prefill: bool = True,
        admit_dynamic: bool = True,
    ) -> torch.Tensor:
        if priority not in ("decode", "prefill"):
            raise ValueError(f"invalid PLE fetch priority: {priority}")
        if priority == "prefill" and self.prefill_reader is None:
            raise RuntimeError(
                "PLE prefill priority requires a separate prefill reader"
            )
        ids = np.asarray(global_ids, dtype=np.int64)
        expected_shape = (*ids.shape, ROW_BYTES)
        if out is None:
            output = _allocate_host_tensor(expected_shape, dtype=torch.uint8)
        else:
            output = out
            if (
                tuple(output.shape) != expected_shape
                or output.dtype != torch.uint8
                or output.device.type != "cpu"
                or not output.is_contiguous()
            ):
                raise ValueError(
                    "PLE fetch output must be contiguous CPU uint8 with shape "
                    f"{expected_shape}"
                )
        output.zero_()
        flat_ids = ids.reshape(-1)
        owned = (flat_ids >= self.image.vocab_start) & (flat_ids < self.image.vocab_end)
        positions = np.flatnonzero(owned)
        if not positions.size:
            self.last_fetch_stats = PLEFetchStats()
            return output
        local = flat_ids[positions] - self.image.vocab_start
        hit, slots = self.hot.lookup(local)
        out_np = output.numpy().reshape(-1, ROW_BYTES)
        if np.any(hit):
            out_np[positions[hit]] = self.hot.rows.numpy()[slots[hit]]
        pending_positions = positions[~hit]
        pending_local = local[~hit]
        prefill_hits = np.zeros(pending_positions.size, dtype=np.bool_)
        if use_prefill and pending_positions.size:
            prefill_output = np.empty(
                (pending_positions.size, ROW_BYTES), dtype=np.uint8
            )
            prefill_hits = self._lookup_prefill(
                flat_ids[pending_positions], prefill_output
            )
            if np.any(prefill_hits):
                out_np[pending_positions[prefill_hits]] = prefill_output[prefill_hits]
        dynamic_positions = pending_positions[~prefill_hits]
        dynamic_local = pending_local[~prefill_hits]
        dynamic_hits = np.zeros(dynamic_positions.size, dtype=np.bool_)
        if dynamic_positions.size:
            dynamic_output = np.empty(
                (dynamic_positions.size, ROW_BYTES), dtype=np.uint8
            )
            dynamic_hits = self.dynamic.lookup_into(
                dynamic_local, dynamic_output, record_hits=admit_dynamic
            )
            if np.any(dynamic_hits):
                out_np[dynamic_positions[dynamic_hits]] = dynamic_output[dynamic_hits]
        cold_positions = dynamic_positions[~dynamic_hits]
        cold_local = dynamic_local[~dynamic_hits]
        cold_pages = 0
        coalesced_rows = 0
        if cold_positions.size:
            page_ids = cold_local // ROWS_PER_PAGE
            within = cold_local % ROWS_PER_PAGE
            unique, inverse = np.unique(page_ids, return_inverse=True)
            cold_pages = int(unique.size)
            coalesced_rows = int(cold_positions.size - unique.size)
            reader = self.prefill_reader if priority == "prefill" else self.reader
            row_bytes = np.arange(ROW_BYTES)[None, :]
            inverse_order = np.argsort(inverse, kind="stable")
            sorted_inverse = inverse[inverse_order]
            for begin in range(0, unique.size, reader.max_pages):
                end = min(begin + reader.max_pages, unique.size)
                with reader.locked_pages(unique[begin:end]) as pages:
                    first = np.searchsorted(sorted_inverse, begin, side="left")
                    last = np.searchsorted(sorted_inverse, end, side="left")
                    selected = inverse_order[first:last]
                    page_index = inverse[selected] - begin
                    byte_index = within[selected, None] * ROW_BYTES + row_bytes
                    out_np[cold_positions[selected]] = pages[
                        page_index[:, None], byte_index
                    ]
            if admit_dynamic:
                self.dynamic.record(cold_local, out_np[cold_positions])
        self.last_fetch_stats = PLEFetchStats(
            rows_requested=int(positions.size),
            static_hits=int(np.count_nonzero(hit)),
            dynamic_hits=int(np.count_nonzero(dynamic_hits)),
            prefill_hits=int(np.count_nonzero(prefill_hits)),
            cold_pages=cold_pages,
            coalesced_rows=coalesced_rows,
        )
        return output

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        executor = getattr(self, "_prefill_executor", None)
        if executor is not None:
            try:
                self.wait_prefill()
            except BaseException:
                logger.warning(
                    "PLE prefill look-ahead failed during shutdown", exc_info=True
                )
            executor.shutdown(wait=True)
            self._prefill_executor = None
        dynamic = getattr(self, "dynamic", None)
        if dynamic is not None:
            dynamic.close()
            self.dynamic = None
        prefill_reader = getattr(self, "prefill_reader", None)
        if prefill_reader is not None:
            prefill_reader.close()
            self.prefill_reader = None
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()
            self.reader = None
