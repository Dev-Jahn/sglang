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
import torch

from sglang.srt.managers import tp_worker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpDiskEmbedding,
    Qwen4ExpModel,
    Qwen4ExpPLELayer,
)
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


def test_prebuilt_generation_entry_prepares_the_model_batch(routed_batch):
    events, forward_batch, runner = routed_batch
    worker = SimpleNamespace(
        model_runner=runner,
        is_dllm=lambda: True,
        _forward_batch_generation_dllm=lambda prepared, schedule: (
            prepared,
            schedule,
        ),
    )

    result = tp_worker.TpModelWorker.forward_batch_generation(
        worker, None, forward_batch=forward_batch
    )

    assert ("prepare", None, forward_batch) in events
    assert result == (forward_batch, None)


def test_prebuilt_extend_entry_skips_disk_lookahead_without_schedule_batch():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._prefill_buffer_tokens = 8
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._future_lookup_contexts = object()
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([layer])
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.arange(3),
    )

    Qwen4ExpModel.prepare_model_batch(model, None, forward_batch)

    assert layer._future_lookup_contexts is None


def test_dllm_helper_relies_on_the_generation_entry_preparation(routed_batch):
    events, forward_batch, runner = routed_batch
    algorithm = SimpleNamespace(
        fdfo=False,
        run=lambda model_runner, prepared, states: (
            None,
            None,
            None,
            None,
            False,
        ),
    )
    worker = SimpleNamespace(model_runner=runner, dllm_algorithm=algorithm)

    tp_worker.TpModelWorker._forward_batch_generation_dllm(worker, forward_batch, None)

    assert ("prepare", None, forward_batch) not in events


def test_split_prefill_entry_prepares_later_chunks(routed_batch):
    events, forward_batch, runner = routed_batch
    batch = SimpleNamespace(
        split_index=1,
        split_forward_count=2,
        split_forward_batch=forward_batch,
    )
    worker = SimpleNamespace(model_runner=runner)

    tp_worker.TpModelWorker.forward_batch_split_prefill(worker, batch)

    assert events[:2] == [
        ("prepare", batch, forward_batch),
        ("forward", forward_batch),
    ]


def test_model_batch_hook_runs_once_for_a_prepared_forward_batch():
    events = []
    model = SimpleNamespace(
        supports_model_batch_hook=True,
        prepare_model_batch=lambda batch, prepared: events.append((batch, prepared)),
    )
    runner = SimpleNamespace(model=model)
    forward_batch = SimpleNamespace()
    batch = object()

    ModelRunner.prepare_model_batch(runner, batch, forward_batch)
    ModelRunner.prepare_model_batch(runner, batch, forward_batch)

    assert events == [(batch, forward_batch)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
