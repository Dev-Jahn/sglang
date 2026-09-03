"""PLE storage hook coverage for online weight updates."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Prepared(BaseException):
    pass


@pytest.mark.parametrize(
    ("path", "invoke"),
    [
        (
            "disk",
            lambda updater: updater.update_weights_from_disk("/tmp/model", "auto"),
        ),
        (
            "distributed",
            lambda updater: updater.update_weights_from_distributed(
                ["weight"], [torch.float32], [(1,)], "update"
            ),
        ),
        (
            "tensor",
            lambda updater: updater.update_weights_from_tensor(
                [("weight", torch.ones(1))]
            ),
        ),
        (
            "flattened tensor",
            lambda updater: updater.update_weights_from_tensor(
                {}, load_format="flattened_bucket"
            ),
        ),
        (
            "ipc",
            lambda updater: updater.update_weights_from_ipc(
                SimpleNamespace(zmq_handles=[])
            ),
        ),
    ],
)
def test_every_weight_update_path_prepares_storage(monkeypatch, path, invoke):
    model = SimpleNamespace(load_weights=MagicMock())
    runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
    updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(
            model_path="/tmp/model", revision=None, dtype=torch.float32
        ),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: runner,
        _model_update_group={"update": object()},
    )
    prepare = MagicMock(side_effect=_Prepared(path))
    monkeypatch.setattr(WeightUpdater, "_prepare_model_for_weight_update", prepare)
    monkeypatch.setattr(
        weight_updater, "_unsupported_derived_weight_cache_error", lambda: None
    )
    monkeypatch.setattr(weight_updater, "monkey_patch_torch_reductions", lambda: None)
    loader = MagicMock(spec=DefaultModelLoader)
    loader._get_weights_iterator.return_value = iter(())
    monkeypatch.setattr(weight_updater, "get_model_loader", lambda *args: loader)
    monkeypatch.setattr(
        weight_updater,
        "get_available_gpu_memory",
        lambda *args, **kwargs: 1.0,
    )

    with pytest.raises(_Prepared, match=path):
        invoke(updater)

    prepare.assert_called_once_with()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
