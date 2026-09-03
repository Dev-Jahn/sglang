"""Qwen4 PLE disk CUDA graph hook tests."""

import sys
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpDiskEmbedding,
    Qwen4ExpModel,
    Qwen4ExpPLELayer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_transfer_records_completion_on_the_prefetch_stream(monkeypatch):
    events = []

    class DeviceBuffer:
        def record_stream(self, stream):
            events.append(("record_stream", stream))

        def copy_(self, source, non_blocking=False):
            events.append(("copy", source, non_blocking))

        def view(self, dtype):
            return torch.zeros((1, 4), dtype=torch.uint8).view(dtype)

    class Completion:
        def record(self, stream):
            events.append(("completion", stream))

    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._fetcher = SimpleNamespace(
        fetch=lambda ids, out, admit_dynamic: events.append(("fetch", out)),
        last_fetch_stats=object(),
    )
    stream = object()
    raw_host = object()
    monkeypatch.setattr(
        qwen4_exp_module.torch.cuda, "stream", lambda value: nullcontext()
    )
    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "Event", Completion)

    embedding._fetch_to_device(
        SimpleNamespace(numpy=lambda: [1]),
        SimpleNamespace(synchronize=lambda: None),
        raw_host,
        DeviceBuffer(),
        torch.empty((1, 4), dtype=torch.bfloat16),
        stream,
        True,
    )

    assert events[1:] == [
        ("record_stream", stream),
        ("copy", raw_host, True),
        ("completion", stream),
    ]


def test_disk_graph_replay_wait_is_a_noop_without_staged_work():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    waits = []
    embedding.wait_for_graph_step = lambda generation: waits.append(generation)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._graph_replay_generation = None
    layer._graph_replay_stage_expected = False

    layer.wait_cuda_graph_replay()

    assert waits == []


def test_disk_graph_replay_rejects_captured_buffer_without_same_step_staging():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._graph_replay_generation = None
    layer._graph_replay_stage_expected = False
    layer._graph_replay_capture_expected_key = None
    layer._pending_graph_embedding_validation = None
    layer._pending_graph_lookup_validation = None
    layer._completed_graph_embedding_validation = deque()
    layer._completed_graph_lookup_validation = deque()
    key = (ForwardMode.IDLE, 4)
    layer._graph_lookup_id_buffers = {key: torch.zeros((4, 1), dtype=torch.long)}

    layer.prepare_cuda_graph_replay(None, None, key)

    with pytest.raises(RuntimeError, match="did not stage its captured buffer"):
        layer.wait_cuda_graph_replay()


def test_disk_capture_retains_the_graph_updated_lookup_buffer(monkeypatch):
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.gather = lambda input_ids, out: out

    captured_ids = torch.arange(64, dtype=torch.long).view(4, 16)
    ngram_embedding = SimpleNamespace(
        ngram_embedding=offloaded,
        ngram_heads=16,
        gather_dp_tokens=False,
        compute_ngram_ids=lambda batch: captured_ids,
        _prepare_embedding_lookup=lambda ids, forward_batch, physical_tokens: (
            ids,
            physical_tokens,
        ),
        _hash_contexts=lambda contexts: (_ for _ in ()).throw(
            AssertionError("capture must not hash eager look-ahead contexts")
        ),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = ngram_embedding
    layer._prefetch_stream = object()
    layer._prefetch_state = None
    layer._future_lookup_contexts = torch.ones((3, 3), dtype=torch.long)
    layer._graph_lookup_id_buffers = {}
    layer._graph_lookup_validation_due = set()
    layer._is_capturing = lambda: True
    layer._get_prefetch_buffer = lambda tokens, ids, graph_key: torch.empty(
        (tokens, 16, 10), dtype=torch.bfloat16
    )

    batch = SimpleNamespace(
        mode=ForwardMode.DECODE, physical_tokens=4, processed_tokens=4
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(4),
        global_dp_buffer_len=None,
        forward_mode=ForwardMode.DECODE,
        _original_forward_mode=None,
    )
    layer.start_prefetch(batch, forward_batch)
    assert layer._prefetch_state[0].shape == (4, 16, 10)
    assert layer._future_lookup_contexts is None
    captured_ids[0, 0] = -1
    key = (ForwardMode.DECODE, 4)
    assert layer._graph_lookup_id_buffers[key].data_ptr() == captured_ids.data_ptr()
    assert layer._graph_lookup_id_buffers[key][0, 0].item() == -1


def test_disk_capture_keys_equal_token_counts_by_forward_mode(monkeypatch):
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.gather = lambda input_ids, out: out

    ngram_embedding = SimpleNamespace(
        ngram_embedding=offloaded,
        ngram_heads=1,
        gather_dp_tokens=False,
        compute_ngram_ids=lambda batch: torch.full(
            (4, 1), int(batch.mode), dtype=torch.long
        ),
        _prepare_embedding_lookup=lambda ids, forward_batch, physical_tokens: (
            ids,
            physical_tokens,
        ),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = ngram_embedding
    layer._prefetch_stream = object()
    layer._prefetch_state = None
    layer._future_lookup_contexts = None
    layer._graph_lookup_id_buffers = {}
    layer._graph_lookup_validation_due = set()
    layer._is_capturing = lambda: True
    layer._get_prefetch_buffer = lambda tokens, ids, graph_key: torch.empty(
        (tokens, 1, 4), dtype=torch.bfloat16
    )

    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        batch = SimpleNamespace(mode=mode, physical_tokens=4, processed_tokens=4)
        forward_batch = SimpleNamespace(
            input_ids=torch.arange(4),
            global_dp_buffer_len=None,
            forward_mode=mode,
            _original_forward_mode=None,
        )
        layer.start_prefetch(batch, forward_batch)
        layer._prefetch_state = None

    assert set(layer._graph_lookup_id_buffers) == {
        (ForwardMode.DECODE, 4),
        (ForwardMode.TARGET_VERIFY, 4),
    }


def test_disk_capture_uses_one_key_when_batch_and_runtime_modes_differ(monkeypatch):
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.gather = lambda input_ids, out: out
    offloaded.reduce = lambda embeddings: embeddings
    offloaded.weight_scale = 1

    ngram_embedding = SimpleNamespace(
        ngram_embedding=offloaded,
        ngram_heads=1,
        gather_dp_tokens=False,
        compute_ngram_ids=lambda batch: torch.arange(4).view(4, 1),
        _prepare_embedding_lookup=lambda ids, forward_batch, physical_tokens: (
            ids,
            physical_tokens,
        ),
        _finish_embedding_lookup=lambda embeddings, *args: embeddings,
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = ngram_embedding
    layer._prefetch_stream = object()
    layer._prefetch_state = None
    layer._future_lookup_contexts = None
    layer._graph_lookup_id_buffers = {}
    layer._graph_embedding_snapshot_buffers = {}
    layer._graph_lookup_validation_due = set()
    layer._validate_graph_staging = True
    layer._is_capturing = lambda: True
    layer._get_prefetch_buffer = lambda tokens, ids, graph_key: torch.empty(
        (tokens, 1, 4), dtype=torch.bfloat16
    )

    batch = SimpleNamespace(
        mode=ForwardMode.TARGET_VERIFY,
        physical_tokens=4,
        processed_tokens=4,
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(4),
        global_dp_buffer_len=None,
        forward_mode=ForwardMode.DECODE,
        _original_forward_mode=None,
    )

    layer.start_prefetch(batch, forward_batch)
    layer._consume_prefetched_embeddings(forward_batch)

    expected = {(ForwardMode.TARGET_VERIFY, 4)}
    assert set(layer._graph_lookup_id_buffers) == expected
    assert set(layer._graph_embedding_snapshot_buffers) == expected


def test_graph_replay_prepares_shared_batch_once(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import CudaGraphReplayInput

    prepared_batch = SimpleNamespace(physical_tokens=2)
    forward_batch = SimpleNamespace(input_ids=torch.arange(1))
    prepare_calls = []
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)) or prepared_batch,
    )

    disk_embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(disk_embedding)
    received = []

    def make_layer(value):
        ngram = SimpleNamespace(
            ngram_embedding=disk_embedding,
            ngram_size=3,
            eos_token_id=2,
            compute_ngram_ids=lambda batch: torch.full((2, 1), value),
            _prepare_embedding_lookup=lambda ids, batch, tokens: (ids + 10, tokens),
        )
        return SimpleNamespace(
            ple_embedding=ngram,
            prepare_cuda_graph_replay=lambda batch, lookup_ids: received.append(
                (batch, lookup_ids.clone())
            ),
        )

    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([make_layer(1), make_layer(2)])
    replay = CudaGraphReplayInput(
        padded_num_tokens=2,
        input_ids=torch.arange(2),
        req_pool_indices=torch.arange(2),
        out_cache_loc=torch.ones(2),
        forward_mode=ForwardMode.DECODE,
        runtime_forward_batch=forward_batch,
    )

    model.prepare_cuda_graph_replay(replay)

    assert len(prepare_calls) == 1
    assert [item[0] for item in received] == [prepared_batch, prepared_batch]
    assert [item[1].tolist() for item in received] == [
        [[11], [11]],
        [[12], [12]],
    ]


def test_graph_replay_prepare_rolls_back_every_disk_layer(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import CudaGraphReplayInput

    disk_embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(disk_embedding)
    events = []

    def make_layer(index, fail=False):
        ngram = SimpleNamespace(
            ngram_embedding=disk_embedding,
            compute_ngram_ids=lambda batch: torch.zeros((1, 1), dtype=torch.long),
            _prepare_embedding_lookup=lambda ids, batch, tokens: (ids, tokens),
        )

        def prepare(batch, lookup_ids):
            events.append(("prepare", index))
            if fail:
                raise OSError("injected layer failure")

        return SimpleNamespace(
            ple_embedding=ngram,
            prepare_cuda_graph_replay=prepare,
            release_cuda_graph_replay=lambda: events.append(("reset", index)),
        )

    layers = [make_layer(0), make_layer(1, fail=True), make_layer(2)]
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter(layers)
    monkeypatch.setattr(
        qwen4_exp_module,
        "_prepare_ple_batch",
        lambda *args, **kwargs: SimpleNamespace(physical_tokens=1),
    )

    replay = CudaGraphReplayInput(
        padded_num_tokens=1,
        input_ids=torch.zeros(1, dtype=torch.long),
        req_pool_indices=torch.zeros(1, dtype=torch.long),
        out_cache_loc=torch.ones(1, dtype=torch.long),
        forward_mode=ForwardMode.DECODE,
        runtime_forward_batch=SimpleNamespace(),
    )

    with pytest.raises(OSError, match="injected layer failure"):
        model.prepare_cuda_graph_replay(replay)

    assert events == [
        ("prepare", 0),
        ("prepare", 1),
        ("reset", 0),
        ("reset", 1),
        ("reset", 2),
    ]


def test_failed_embedding_future_does_not_block_the_next_graph_step(monkeypatch):
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.embedding_dim = 4
    embedding._active_transfer_device = "cuda:0"
    embedding._graph_generation = 0
    embedding._active_graph_generation = None
    failed = Future()
    failed.set_exception(OSError("injected fetch failure"))
    embedding._future = failed

    with pytest.raises(OSError, match="injected fetch failure"):
        embedding.wait_for_prefetch()

    monkeypatch.setattr(embedding, "_launch_fetch", lambda *args, **kwargs: None)
    generation = embedding.stage_graph_step(
        torch.zeros((1,), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.bfloat16),
    )
    assert generation == 1


def test_prefetch_completion_joins_the_transfer_device_stream(monkeypatch):
    class ComputeStream:
        def __init__(self):
            self.events = []

        def wait_event(self, event):
            self.events.append(event)

    completion = object()
    transfer_device = torch.device("cuda:5")
    compute_stream = ComputeStream()
    current_stream_devices = []

    def current_stream(device=None):
        current_stream_devices.append(device)
        return compute_stream

    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "current_stream", current_stream)
    future = Future()
    future.set_result(
        (
            completion,
            SimpleNamespace(
                rows_requested=1,
                static_hits=0,
                dynamic_hits=0,
                prefill_hits=0,
                cold_pages=1,
                coalesced_rows=0,
            ),
            0.0,
            0.0,
        )
    )
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._future = future
    embedding._active_transfer_device = transfer_device
    embedding._wait_histogram_bin = lambda wait_us: 0
    embedding._stats_log_interval = 0
    embedding._transfer_buffers = {}
    embedding._transfer_buffer_retain_rows = 1
    embedding._stats = defaultdict(float, miss_wait_hist=[0])

    embedding.wait_for_prefetch()

    assert current_stream_devices == [transfer_device]
    assert compute_stream.events == [completion]


def test_eager_forward_exception_resets_prefetch_before_the_next_forward():
    completed = Future()
    completed.set_result((SimpleNamespace(synchronize=lambda: None),))
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._future = completed
    embedding._active_graph_generation = 3
    layer = SimpleNamespace(reset_eager_prefetch=embedding.reset_eager_step)
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model._ple_layers = lambda: iter([layer])

    def fail(*args, **kwargs):
        raise OSError("injected mid-forward failure")

    model._forward_impl = fail
    with pytest.raises(OSError, match="mid-forward"):
        model.forward(None, None, None)
    assert embedding._future is None
    assert embedding._active_graph_generation is None

    expected = object()
    model._forward_impl = lambda *args, **kwargs: (
        expected if embedding._future is None else None
    )
    assert model.forward(None, None, None) is expected


def test_layer_wait_resets_graph_state_after_fetch_failure():
    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(offloaded)
    offloaded.wait_for_graph_step = lambda generation: (_ for _ in ()).throw(
        OSError("injected layer wait")
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=offloaded)
    layer._graph_replay_generation = 4
    layer._graph_replay_stage_expected = True
    layer._graph_replay_lookup_tokens = 1
    layer._graph_replay_prefetch_buffer = torch.zeros(1)
    layer._prefetch_stream = object()
    layer._validate_graph_staging = False

    with pytest.raises(OSError, match="injected layer wait"):
        layer.wait_cuda_graph_replay()

    assert layer._graph_replay_generation is None
    assert not layer._graph_replay_stage_expected
    assert layer._graph_replay_lookup_tokens is None
    assert layer._graph_replay_prefetch_buffer is None


def test_model_wait_keeps_replay_validation_pending_until_finish():
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    events = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(ngram_embedding=embedding),
        wait_cuda_graph_replay=lambda: events.append("wait"),
        release_cuda_graph_replay=lambda: events.append("reset"),
    )
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model._ple_layers = lambda: iter([layer])

    model.wait_cuda_graph_replay()

    assert events == ["wait"]


def test_graph_lookup_validation_zero_checks_only_the_first_replay(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv(
        "SGLANG_PLE_DISK_GRAPH_LOOKUP_VALIDATION_INTERVAL", raising=False
    )
    assert envs.SGLANG_PLE_DISK_GRAPH_LOOKUP_VALIDATION_INTERVAL.get() == 0

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_replay_steps = 0
    key = (ForwardMode.DECODE, 4)
    layer._graph_lookup_validation_due = {key}
    layer._graph_lookup_validation_interval = 0

    assert layer._graph_lookup_validation_required(key, 1)
    layer._graph_lookup_validation_due.clear()
    assert all(
        not layer._graph_lookup_validation_required(key, step) for step in range(2, 10)
    )
    assert layer._graph_replay_steps == 0
    assert layer._graph_lookup_validation_due == set()

    layer._graph_lookup_validation_interval = 256
    layer._graph_lookup_validation_due = {key}
    assert layer._graph_lookup_validation_required(key, 1)
    layer._graph_lookup_validation_due.clear()
    assert all(
        not layer._graph_lookup_validation_required(key, step) for step in range(2, 256)
    )
    assert layer._graph_lookup_validation_required(key, 256)

    layer._graph_lookup_validation_interval = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        layer._graph_lookup_validation_required(key, 257)


def test_graph_replay_shared_buffer_requires_a_captured_size(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_prefetch_buffer = torch.empty((8, 16))
    key = (ForwardMode.DECODE, 4)
    layer._graph_lookup_id_buffers = {key: torch.empty((4, 1), dtype=torch.long)}
    monkeypatch.setattr(qwen4_exp_module, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp_module, "is_sm121", lambda: False)

    assert layer._select_graph_prefetch_buffer(key).shape == (4, 16)
    with pytest.raises(RuntimeError, match="no captured staging buffer"):
        layer._select_graph_prefetch_buffer((ForwardMode.DECODE, 2))


def test_capture_start_drops_references_from_the_previous_graph():
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_lookup_id_buffers = {1: object()}
    layer._graph_embedding_snapshot_buffers = {1: object()}
    layer._graph_lookup_validation_due = {1}
    layer._pending_graph_lookup_validation = object()
    layer._pending_graph_embedding_validation = object()

    layer.reset_cuda_graph_capture_buffers()

    assert layer._graph_lookup_id_buffers == {}
    assert layer._graph_embedding_snapshot_buffers == {}
    assert layer._graph_lookup_validation_due == set()
    assert layer._pending_graph_lookup_validation is None
    assert layer._pending_graph_embedding_validation is None
    layer.finish_cuda_graph_replay()


def test_graph_replay_release_discards_pending_validation(monkeypatch):
    embedding = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    torch.nn.Module.__init__(embedding)
    monkeypatch.setattr(embedding, "reset_graph_step", lambda: None)
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.ple_embedding = SimpleNamespace(ngram_embedding=embedding)
    layer._graph_replay_generation = 1
    layer._graph_replay_stage_expected = True
    layer._graph_replay_lookup_tokens = 4
    layer._graph_replay_prefetch_buffer = object()
    layer._pending_graph_lookup_validation = object()
    layer._pending_graph_embedding_validation = object()

    layer.release_cuda_graph_replay()

    assert layer._pending_graph_lookup_validation is None
    assert layer._pending_graph_embedding_validation is None


def test_graph_lookup_validation_checks_the_current_replay(monkeypatch):
    class ReadyEvent:
        def record(self, stream):
            self.stream = stream

        def query(self):
            return True

    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "Event", ReadyEvent)
    monkeypatch.setattr(qwen4_exp_module.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(
        qwen4_exp_module,
        "_allocate_host_tensor",
        lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._pending_graph_embedding_validation = None
    layer._completed_graph_embedding_validation = deque()
    layer._pending_graph_lookup_validation = (
        (ForwardMode.DECODE, 2),
        17,
        torch.tensor([[3], [5]], dtype=torch.long),
    )
    layer._completed_graph_lookup_validation = deque()
    layer._graph_validation_free_slots = deque()
    layer._graph_lookup_id_buffers = {
        (ForwardMode.DECODE, 2): torch.tensor([[3], [7]], dtype=torch.long)
    }

    layer.finish_cuda_graph_replay()
    assert len(layer._completed_graph_lookup_validation) == 1
    with pytest.raises(
        RuntimeError,
        match=r"one step behind.*step 17.*already emitted.*lookup.tokens=2",
    ):
        layer.validate_cuda_graph_replay()


def test_graph_validation_rejects_a_missing_capture_buffer():
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._pending_graph_embedding_validation = None
    layer._completed_graph_embedding_validation = deque()
    layer._pending_graph_lookup_validation = (
        (ForwardMode.DECODE, 2),
        17,
        torch.tensor([[3], [5]], dtype=torch.long),
    )
    layer._completed_graph_lookup_validation = deque()
    layer._graph_validation_free_slots = deque()
    layer._graph_lookup_id_buffers = {}

    with pytest.raises(RuntimeError, match="capture buffer is missing"):
        layer.finish_cuda_graph_replay()

    assert layer._pending_graph_lookup_validation is None
    assert not layer._completed_graph_lookup_validation


def test_graph_validation_recycles_every_consumed_ready_entry():
    class ReadyEvent:
        def query(self):
            return True

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer._graph_validation_free_slots = deque()
    layer._completed_graph_lookup_validation = deque(
        [
            (2, 17, torch.tensor([True]), ReadyEvent()),
            (4, 18, torch.tensor([False]), ReadyEvent()),
        ]
    )

    with pytest.raises(RuntimeError, match="step 17"):
        layer._consume_graph_validation(
            "_completed_graph_lookup_validation", "lookup mismatch"
        )

    assert not layer._completed_graph_lookup_validation
    assert len(layer._graph_validation_free_slots) == 2


def test_forward_batch_declares_model_batch_hook_state():
    assert "_model_batch_hook_prepared" in ForwardBatch.__dataclass_fields__


def test_graph_replay_uses_the_explicit_padded_token_extent(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import (
        CudaGraphReplayInput,
        ForwardMode,
    )

    runtime = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.ones(2, dtype=torch.int32),
        seq_lens_sum=2,
        spec_info=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        global_num_tokens_cpu=None,
        global_num_tokens_gpu=None,
        dp_padding_mode=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        _original_forward_mode=None,
        num_token_non_padded_cpu=1,
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
        extend_seq_lens_cpu=[1],
        extend_prefix_lens_cpu=[0],
        extend_num_tokens=1,
    )
    replay = CudaGraphReplayInput(
        padded_num_tokens=2,
        input_ids=torch.arange(2),
        req_pool_indices=torch.arange(2, dtype=torch.int32),
        out_cache_loc=torch.ones(2, dtype=torch.int64),
        forward_mode=ForwardMode.DECODE,
        runtime_forward_batch=runtime,
    )
    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices,
        get_ngram_context=lambda indices: torch.zeros(
            (indices.numel(), 2), dtype=torch.long
        ),
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    batch = qwen4_exp_module._prepare_ple_batch(
        replay.input_ids,
        runtime,
        ngram_size=3,
        ngram_eos_token_id=2,
        replay=replay,
    )
    assert batch.physical_tokens == 2
    assert batch.processed_tokens == 2
    assert batch.lengths.tolist() == [1, 1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
