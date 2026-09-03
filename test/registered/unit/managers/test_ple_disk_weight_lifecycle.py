"""PLE disk weight lifecycle synchronization tests."""

import sys
from types import SimpleNamespace

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


class _FailingStorageModel(torch.nn.Module):
    supports_storage_lifecycle_hook = True

    def __init__(self, events):
        super().__init__()
        self.events = events

    def close(self):
        self.events.append("close")
        raise OSError("rank-local close failure")

    def resume_storage(self):
        self.events.append("resume_storage")


class _FailingResumeModel(torch.nn.Module):
    supports_storage_lifecycle_hook = True

    def __init__(self, events):
        super().__init__()
        self.events = events

    def resume_storage(self):
        self.events.append("resume_storage")
        raise OSError("rank-local resume failure")


def test_close_failure_reaches_collectives_before_raising(monkeypatch):
    events = []
    group = object()
    model = _FailingStorageModel(events)
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
                server_args=SimpleNamespace(weight_cache_mode="off"),
            )
        ),
        draft_worker=None,
        tp_cpu_group=group,
        memory_saver_adapter=SimpleNamespace(pause=lambda tag: None),
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
    )

    def all_reduce(failed, *, op, group):
        events.append("all_reduce")

    def barrier(*, group):
        events.append("barrier")

    monkeypatch.setattr(weight_updater.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(weight_updater.torch.distributed, "barrier", barrier)

    with pytest.raises(OSError, match="rank-local close failure"):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    assert events == ["close", "all_reduce", "barrier"]


def test_resume_failure_reaches_collectives_before_raising(monkeypatch):
    events = []
    group = object()
    model = _FailingResumeModel(events)
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
                server_args=SimpleNamespace(weight_cache_mode="off"),
            )
        ),
        draft_worker=None,
        tp_cpu_group=group,
        memory_saver_adapter=SimpleNamespace(resume=lambda tag: None),
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
        offload_tags={GPU_MEMORY_TYPE_WEIGHTS},
        stashed_model_static_state={},
    )
    monkeypatch.setattr(weight_updater, "_import_static_state", lambda *args: None)
    monkeypatch.setattr(
        weight_updater.torch.distributed,
        "all_reduce",
        lambda failed, *, op, group: events.append("all_reduce"),
    )
    monkeypatch.setattr(
        weight_updater.torch.distributed,
        "barrier",
        lambda *, group: events.append("barrier"),
    )

    with pytest.raises(OSError, match="rank-local resume failure"):
        manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    assert events == ["barrier", "resume_storage", "all_reduce", "barrier"]
    assert manager.stashed_model_static_state == {}


def test_peer_storage_failure_is_raised_after_barrier(monkeypatch):
    events = []
    manager = SchedulerWeightUpdaterManager(
        tp_worker=None,
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=None,
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
    )

    def all_reduce(failed, *, op, group):
        events.append("all_reduce")
        failed.fill_(1)

    monkeypatch.setattr(weight_updater.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        weight_updater.torch.distributed,
        "barrier",
        lambda *, group: events.append("barrier"),
    )

    with pytest.raises(RuntimeError, match="another tensor-parallel rank"):
        manager._run_tp_storage_operation(
            lambda: events.append("operation"), "storage operation"
        )

    assert events == ["operation", "all_reduce", "barrier"]


def test_close_failure_preserves_state_for_a_later_resume(monkeypatch):
    events = []
    model = _FailingStorageModel(events)
    adapter = SimpleNamespace(
        pause=lambda tag: events.append("pause"),
        resume=lambda tag: events.append("resume"),
    )
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
                server_args=SimpleNamespace(weight_cache_mode="off"),
            )
        ),
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=adapter,
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
    )
    monkeypatch.setattr(
        weight_updater,
        "_export_static_state",
        lambda candidate: events.append("export") or {"saved": True},
    )
    monkeypatch.setattr(
        weight_updater,
        "_import_static_state",
        lambda candidate, state: events.append(("import", state)),
    )
    monkeypatch.setattr(
        weight_updater.torch.distributed, "all_reduce", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        weight_updater.torch.distributed,
        "barrier",
        lambda *args, **kwargs: events.append("barrier"),
    )

    with pytest.raises(OSError, match="rank-local close failure"):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    assert manager.stashed_model_static_state == {"saved": True}
    assert events == ["export", "close", "barrier"]
    manager.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )
    assert ("import", {"saved": True}) in events
    assert "resume_storage" in events


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
