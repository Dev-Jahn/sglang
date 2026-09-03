"""PLE storage hook coverage for online weight updates."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_tensor_update_arms_the_storage_reload_hook(monkeypatch):
    events = []

    class Model:
        supports_storage_lifecycle_hook = True

        def prepare_weight_reload(self):
            events.append("prepare")

        def load_weights(self, tensors):
            events.append(("load", list(tensors)))

    model = Model()
    runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
    )
    monkeypatch.setattr(
        weight_updater, "_unsupported_derived_weight_cache_error", lambda: None
    )
    monkeypatch.setattr(weight_updater, "monkey_patch_torch_reductions", lambda: None)
    monkeypatch.setattr(
        weight_updater.torch,
        "get_device_module",
        lambda device: SimpleNamespace(current_device=lambda: "cpu"),
    )

    success, _ = updater.update_weights_from_tensor([("ple.weight", torch.ones(1))])

    assert success
    assert events[0] == "prepare"
    assert events[1][0] == "load"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
