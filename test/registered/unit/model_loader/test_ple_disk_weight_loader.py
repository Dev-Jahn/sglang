# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""PLE checkpoint identity behavior in shared weight iterators."""

import sys
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from sglang.srt.model_loader import weight_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_fastsafetensors_iterator_does_not_preopen_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint.safetensors"
    tensor = torch.arange(4)
    safetensors.torch.save_file({"checkpoint.weight": tensor}, source)

    class Group:
        rank = lambda self: 0
        size = lambda self: 1

    class Batch:
        key_to_rank_lidx = {"streamed.weight": None}

        def get_tensor(self, name):
            return tensor

    class Loader:
        def __init__(self, *args, **kwargs):
            pass

        def add_filenames(self, rank_file_map):
            self.rank_file_map = rank_file_map

        def copy_files_to_device(self):
            return Batch()

        def close(self):
            pass

    safe_open_calls = []
    monkeypatch.setattr(weight_utils, "SingleGroup", Group)
    monkeypatch.setattr(weight_utils, "SafeTensorsFileLoader", Loader)
    monkeypatch.setattr(
        weight_utils.safetensors,
        "safe_open",
        lambda *args, **kwargs: safe_open_calls.append(args),
    )

    loaded = list(weight_utils.fastsafetensors_weights_iterator([str(source)]))

    assert [name for name, _ in loaded] == ["streamed.weight"]
    assert safe_open_calls == []
    assert not hasattr(loaded[0][1], weight_utils.CHECKPOINT_SOURCE_TENSOR_ATTR)


def test_runai_iterator_does_not_preopen_checkpoint_or_require_known_key(
    tmp_path, monkeypatch
):
    source = tmp_path / "checkpoint.safetensors"
    tensor = torch.arange(4)
    safetensors.torch.save_file({"checkpoint.weight": tensor}, source)

    class Streamer:
        files_to_tensors_metadata = {"fixture": [object()]}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stream_files(self, *args, **kwargs):
            pass

        def get_tensors(self):
            yield "streamed.weight", tensor

    safe_open_calls = []
    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer",
        SimpleNamespace(SafetensorsStreamer=Streamer),
    )
    monkeypatch.setattr(
        weight_utils.safetensors,
        "safe_open",
        lambda *args, **kwargs: safe_open_calls.append(args),
    )

    loaded = list(weight_utils.runai_safetensors_weights_iterator([str(source)]))

    assert [name for name, _ in loaded] == ["streamed.weight"]
    assert safe_open_calls == []
    assert not hasattr(loaded[0][1], weight_utils.CHECKPOINT_SOURCE_TENSOR_ATTR)


@pytest.mark.parametrize("disable_mmap", [False, True])
def test_safetensors_loader_attaches_source_identity(tmp_path, disable_mmap):
    import safetensors.torch

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    source = tmp_path / "checkpoint.safetensors"
    safetensors.torch.save_file({"weight": torch.arange(4)}, source)
    name, tensor = next(
        safetensors_weights_iterator([str(source)], disable_mmap=disable_mmap)
    )
    stat = source.stat()
    assert name == "weight"
    assert tensor._sglang_checkpoint_source == {
        "file": source.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
