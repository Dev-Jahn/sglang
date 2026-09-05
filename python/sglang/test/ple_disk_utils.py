"""Test helpers shared by the Qwen4 PLE disk suites."""

from __future__ import annotations

import errno
import re
from pathlib import Path

import torch


def attach_checkpoint_source(
    tensor: torch.Tensor, *, source_name: str = "test.safetensors"
) -> torch.Tensor:
    tensor._sglang_checkpoint_source = {
        "file": source_name,
        "size": tensor.nbytes,
        "mtime_ns": 1,
    }
    return tensor


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
):
    from sglang.srt.models.qwen4_ple_disk import PLEImageBuilder

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
        "test.shard_0.weight",
        attach_checkpoint_source(rows),
        vocab_start,
        vocab_start + rows.shape[0],
    )
    image, _, _ = builder.finalize(weight_scale)
    return image


def native_reader_unavailable_reason(exc: BaseException) -> str | None:
    from sglang.srt.models.qwen4_ple_disk import DirectIOUnavailableError

    if isinstance(exc, RuntimeError):
        messages = (
            "sgl_kernel is not importable",
            "sglang-kernel",
            "qwen4_ple_disk_fetcher",
            "io_uring support",
            "required ABI version symbol",
            "Could not determine the logical block size",
        )
        if any(message in str(exc) for message in messages):
            return f"PLE disk helper or direct-I/O storage is unavailable: {exc}"
    if isinstance(exc, DirectIOUnavailableError):
        return f"PLE io_uring or O_DIRECT is unavailable: {exc}"
    if isinstance(exc, OSError) and exc.errno in {
        errno.EPERM,
        errno.EACCES,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
    }:
        return f"PLE io_uring or O_DIRECT is unavailable: {exc}"
    return None


def fetcher_header_constants() -> dict[str, int]:
    header = (
        Path(__file__).resolve().parents[1]
        / "kernels/aot/csrc/host/qwen4_ple_disk_fetcher.h"
    ).read_text()
    return {
        name: int(value)
        for name, value in re.findall(
            r"^#define\s+(PLE_FETCHER_[A-Z0-9_]+)\s+([0-9]+)U?\s*$",
            header,
            flags=re.MULTILINE,
        )
    }
