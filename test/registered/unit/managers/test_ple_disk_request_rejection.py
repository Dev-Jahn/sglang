"""PLE disk request-boundary coverage."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components import weight_updater
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_DISK_ERROR = (
    "--ple-storage disk does not support online weight updates or memory release "
    "and resume requests"
)
_UPDATE_METHODS = (
    "update_weights_from_disk",
    "init_weights_update_group",
    "destroy_weights_update_group",
    "update_weights_from_distributed",
    "update_weights_from_tensor",
    "update_weights_from_ipc",
)


def _manager(storage):
    model = torch.nn.Linear(1, 1)
    worker = MagicMock()
    worker.model_runner = SimpleNamespace(
        model=model,
        server_args=SimpleNamespace(
            ple_storage=storage,
            weight_cache_mode="off",
        ),
    )
    for method_name in _UPDATE_METHODS:
        getattr(worker, method_name).return_value = (False, "base behavior")

    manager = SchedulerWeightUpdaterManager(
        tp_worker=worker,
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=MagicMock(),
        flush_cache=MagicMock(return_value=True),
        is_fully_idle=MagicMock(return_value=True),
    )
    return manager, worker


@pytest.mark.parametrize("method_name", _UPDATE_METHODS)
def test_disk_storage_rejects_online_weight_update_before_worker_call(
    monkeypatch, method_name
):
    manager, worker = _manager("disk")
    barrier = MagicMock()
    monkeypatch.setattr(weight_updater.torch.distributed, "barrier", barrier)
    request = MagicMock(disable_draft_model=False)

    with pytest.raises(RuntimeError, match=_DISK_ERROR):
        getattr(manager, method_name)(request)

    getattr(worker, method_name).assert_not_called()
    barrier.assert_not_called()


@pytest.mark.parametrize(
    ("method_name", "request_input"),
    (
        (
            "release_memory_occupation",
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS]),
        ),
        (
            "resume_memory_occupation",
            ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS]),
        ),
    ),
)
def test_disk_storage_rejects_memory_request_before_state_or_collective(
    monkeypatch, method_name, request_input
):
    manager, _ = _manager("disk")
    manager.offload_tags = {GPU_MEMORY_TYPE_WEIGHTS}
    manager.stashed_model_static_state = {"buffers": []}
    barrier = MagicMock()
    monkeypatch.setattr(weight_updater.torch.distributed, "barrier", barrier)

    with pytest.raises(RuntimeError, match=_DISK_ERROR):
        getattr(manager, method_name)(request_input)

    manager.is_fully_idle.assert_not_called()
    manager.memory_saver_adapter.pause.assert_not_called()
    manager.memory_saver_adapter.resume.assert_not_called()
    barrier.assert_not_called()


@pytest.mark.parametrize("storage", ("gpu", "pinned"))
def test_other_storage_keeps_weight_update_release_and_resume_behavior(
    monkeypatch, storage
):
    manager, worker = _manager(storage)
    barrier = MagicMock()
    monkeypatch.setattr(weight_updater.torch.distributed, "barrier", barrier)
    monkeypatch.setattr(
        weight_updater.torch,
        "get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )

    result = manager.update_weights_from_disk(MagicMock())
    assert not result.success
    worker.update_weights_from_disk.assert_called_once()

    manager.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )
    manager.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )

    manager.memory_saver_adapter.pause.assert_called_once_with(GPU_MEMORY_TYPE_WEIGHTS)
    manager.memory_saver_adapter.resume.assert_called_once_with(GPU_MEMORY_TYPE_WEIGHTS)
    assert barrier.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
