# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for model batch hooks at tensor-parallel worker entry points."""

from types import SimpleNamespace

import pytest

from sglang.srt.managers import tp_worker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def routed_batch(monkeypatch):
    events = []
    forward_batch = SimpleNamespace(
        apply_deprecated_skip_attn_backend_init=lambda value: events.append(
            ("deprecated", value)
        )
    )
    monkeypatch.setattr(
        tp_worker.ForwardBatch,
        "init_new",
        lambda *args, **kwargs: forward_batch,
    )

    runner = SimpleNamespace(
        prepare_model_batch=lambda batch, prepared: events.append(
            ("prepare", batch, prepared)
        ),
        forward=lambda prepared, **kwargs: events.append(("forward", prepared))
        or SimpleNamespace(
            logits_output=None,
            can_run_graph=True,
            expert_distribution_metrics=None,
        ),
    )
    return events, forward_batch, runner


def test_embedding_entry_prepares_the_model_batch(routed_batch):
    events, forward_batch, runner = routed_batch
    batch = object()
    worker = SimpleNamespace(model_runner=runner)

    tp_worker.BaseTpWorker.forward_batch_embedding(worker, batch)

    assert events[:2] == [
        ("prepare", batch, forward_batch),
        ("forward", forward_batch),
    ]


def test_generation_entry_prepares_the_model_batch(routed_batch):
    events, forward_batch, runner = routed_batch
    batch = SimpleNamespace(hicache_consumer_index=7)
    worker = SimpleNamespace(
        model_runner=runner,
        set_hicache_consumer=lambda index: events.append(("consumer", index)),
        is_dllm=lambda: True,
        _forward_batch_generation_dllm=lambda prepared, schedule: (
            prepared,
            schedule,
        ),
    )

    result = tp_worker.TpModelWorker.forward_batch_generation(worker, batch)

    assert ("prepare", batch, forward_batch) in events
    assert result == (forward_batch, batch)


def test_split_prefill_entry_prepares_the_first_chunk(routed_batch):
    events, forward_batch, runner = routed_batch
    batch = SimpleNamespace(split_index=0, split_forward_count=2)
    worker = SimpleNamespace(model_runner=runner)

    result = tp_worker.TpModelWorker.forward_batch_split_prefill(worker, batch)

    assert events[:2] == [
        ("prepare", batch, forward_batch),
        ("forward", forward_batch),
    ]
    assert batch.split_forward_batch is forward_batch
    assert result.can_run_cuda_graph is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
