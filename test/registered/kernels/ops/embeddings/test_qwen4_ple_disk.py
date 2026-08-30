import errno
import os
import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpDiskEmbedding,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
)
from sglang.srt.models.qwen4_ple_disk import (
    PAGE_BYTES,
    ROW_BYTES,
    DirectPageReader,
    DiskRowFetcher,
    PLEImageBuilder,
    build_test_image,
    open_ple_image,
    write_hot_frequency_file,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA pinned memory is required"
)


@pytest.fixture(scope="module", autouse=True)
def require_ple_direct_io(tmp_path_factory):
    root = tmp_path_factory.mktemp("ple-direct-io-probe")
    _, rows = _fp8_rows(25)
    image = build_test_image(root, rows)
    reader = None
    try:
        reader = DirectPageReader(image, max_pages=1)
        reader.read(np.array([0], dtype=np.int64))
    except OSError as exc:
        if exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.ENOMEM,
        }:
            reason = f"PLE io_uring or O_DIRECT is unavailable: {exc}"
            # Self-hosted runner operators may enable this after confirming the
            # runner permits io_uring and its scratch filesystem supports O_DIRECT.
            if os.environ.get("SGLANG_CI_PLE_DISK_IO_URING") == "1":
                pytest.fail(reason)
            pytest.skip(reason)
        raise
    except RuntimeError as exc:
        message = str(exc)
        if (
            "requires sglang-kernel" in message
            or "qwen4_ple_disk_fetcher" in message
            or "io_uring support" in message
        ):
            pytest.skip(f"PLE disk helper is unavailable: {exc}")
        raise
    finally:
        if reader is not None:
            reader.close()


def _fp8_rows(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260827)
    raw = torch.randint(
        0, 256, (count, ROW_BYTES), dtype=torch.uint8, generator=generator
    )
    # Avoid NaN FP8 encodings so BF16 equality has ordinary value semantics.
    raw[(raw & 0x7F) == 0x7F] = 0
    return raw, raw.view(torch.float8_e4m3fn)


def _context_ids(contexts: np.ndarray, eos: int, rows_per_head: int) -> np.ndarray:
    """Small deterministic 16-head hash with the same EOS segment contract."""
    contexts = contexts.copy()
    result = np.empty((contexts.shape[0], 16), dtype=np.int64)
    for row, context in enumerate(contexts):
        previous = eos
        history = []
        for token in context:
            if token == eos:
                history = []
            history.append(int(token))
            history = history[-3:]
            previous = history[-2] if len(history) >= 2 else eos
        for head in range(16):
            ngram = 2 if head < 8 else 3
            values = ([eos] * ngram + history)[-ngram:]
            mixed = sum((index + 3) * value for index, value in enumerate(values))
            result[row, head] = (
                head * rows_per_head + (mixed + 97 * head + previous) % rows_per_head
            )
    return result


def _source_embedding(rows: int):
    weight = nn.Parameter(
        torch.empty((rows, ROW_BYTES), dtype=torch.float8_e4m3fn, device="cuda"),
        requires_grad=False,
    )
    indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=0,
        padded_org_vocab_end_index=rows,
        padded_added_vocab_start_index=rows,
        padded_added_vocab_end_index=rows,
        org_vocab_start_index=0,
        org_vocab_end_index=rows,
        added_vocab_start_index=rows,
        added_vocab_end_index=rows,
    )
    return SimpleNamespace(
        weight=weight,
        quant_method=UnquantizedEmbeddingMethod(),
        quant_config=None,
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=1,
        num_embeddings=rows,
        org_vocab_size=rows,
        padding_size=1,
        num_added_embeddings=0,
        use_presharded_weights=False,
        org_vocab_size_padded=rows,
        num_embeddings_padded=rows,
        shard_indices=indices,
        embedding_dim=ROW_BYTES,
        num_embeddings_per_partition=rows,
        num_org_embeddings_per_partition=rows,
        num_added_embeddings_per_partition=0,
        weight_scale=torch.tensor([0.25], dtype=torch.bfloat16, device="cuda"),
    )


@pytest.mark.parametrize("padded_tokens", [4, 8])
def test_gpu_storage_gather_keeps_padded_slot_ids_in_range(padded_tokens, monkeypatch):
    from sglang.srt.layers import vocab_parallel_embedding as vocab_module
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    total_rows = 512
    source = _source_embedding(total_rows)
    source.weight = nn.Parameter(
        torch.arange(total_rows * ROW_BYTES, dtype=torch.bfloat16, device="cuda").view(
            total_rows, ROW_BYTES
        ),
        requires_grad=False,
    )
    source.output_dtype = torch.bfloat16
    embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    nn.Module.__init__(embedding)
    for name, value in vars(source).items():
        setattr(embedding, name, value)
    monkeypatch.setattr(
        vocab_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "1")

    expected_ids = torch.arange(
        padded_tokens * 16, dtype=torch.long, device="cuda"
    ).view(padded_tokens, 16)
    batch = SimpleNamespace(
        ngram_context=torch.arange(
            padded_tokens * 3, dtype=torch.long, device="cuda"
        ).view(padded_tokens, 3),
        use_decode_fast_path=True,
        valid_tokens=torch.tensor(
            [True] + [False] * (padded_tokens - 1), device="cuda"
        ),
        mode=ForwardMode.DECODE,
    )
    pool = SimpleNamespace(ple_window_cache=None)
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(module)
    module._mask_invalid_ngram_ids = False
    module._hash_contexts = lambda contexts, decode_sized=False: expected_ids.clone()

    lookup_ids = module.compute_ngram_ids(batch)
    assert torch.equal(lookup_ids, expected_ids)
    actual = embedding(lookup_ids)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, embedding.weight[lookup_ids], rtol=0, atol=0)


def test_disk_image_matches_exact_fp8_gather_across_hot_and_cold_rows(tmp_path):
    rows_per_head = 4096
    raw, rows = _fp8_rows(16 * rows_per_head)
    root = tmp_path
    scale = 0.00019931793212890625
    image = build_test_image(root, rows, weight_scale=scale)
    hot_ids = np.array([0, 17, 4096 + 3, 15 * 4096 + 4095], dtype=np.uint32)
    hot_file = root / "hot.bin"
    write_hot_frequency_file(
        hot_file,
        {0: hot_ids},
        fingerprint=image.header["fingerprint"],
        total_rows=image.vocab_end,
        tp_size=1,
        padding_divisor=1,
    )
    fetcher = DiskRowFetcher(
        image, hot_frequency_file=str(hot_file), hot_cache_gb=0.01, max_pages=256
    )
    try:
        rng = np.random.default_rng(20260827)
        contexts = rng.integers(0, 2048, size=(64, 3), dtype=np.int64)
        contexts[0] = [11, 2, 19]
        contexts[1] = [2, 2, 2]
        contexts[2] = [7, 11, 2]
        ids = _context_ids(contexts, eos=2, rows_per_head=rows_per_head)
        ids[0, :4] = hot_ids
        actual_raw = fetcher.fetch(ids)
        expected_raw = raw.index_select(0, torch.from_numpy(ids.reshape(-1))).reshape(
            *ids.shape, ROW_BYTES
        )
        assert torch.equal(actual_raw, expected_raw)

        actual = (
            actual_raw.to("cuda", non_blocking=True)
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
            * scale
        )
        expected = (
            rows.index_select(0, torch.from_numpy(ids.reshape(-1)))
            .reshape(*ids.shape, ROW_BYTES)
            .to("cuda")
            .to(torch.bfloat16)
            * scale
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finally:
        fetcher.close()


def test_disk_embedding_async_d2h_fetch_h2d(tmp_path, monkeypatch):
    raw, rows = _fp8_rows(256)
    root = tmp_path
    config = SimpleNamespace(
        ple_disk_dir=str(root),
        ple_disk_hot_cache_gb=0.0,
        ple_disk_hot_frequency_file=None,
        ple_disk_prefill_buffer_tokens=4,
        ple_disk_prefill_read_pages=2,
        to_dict=lambda: {"model_type": "synthetic-qwen4", "rows": 256},
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, attn_tp_rank=0),
    )
    embedding = Qwen4ExpDiskEmbedding(_source_embedding(256), config, 256)
    embedding.add_checkpoint_shard("synthetic.shard_0.weight", rows, 0, 256)
    embedding.finalize_image()
    try:
        ids = torch.tensor([[0, 17, 255], [128, 3, 99]], device="cuda")
        output = embedding.gather(ids)
        embedding.wait_for_prefetch()
        expected = (
            rows.index_select(0, ids.cpu().flatten())
            .reshape(*ids.shape, ROW_BYTES)
            .to("cuda")
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
        scaled = output * embedding.weight_scale
        torch.testing.assert_close(
            scaled,
            expected * embedding.weight_scale,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        stats = embedding.stats_snapshot()
        assert stats["steps"] == 1
        assert stats["rows_requested"] == ids.numel()
        assert stats["cold_pages"] > 0
        assert stats["miss_steps"] == 1
        assert sum(stats["miss_wait_hist"]) == 1
        assert stats["miss_host_wait_us"] == stats["host_wait_us"]
        assert stats["miss_ids_ready_wait_us"] == stats["ids_ready_wait_us"]
        assert stats["miss_storage_fetch_us"] == stats["storage_fetch_us"]

        future_ids = torch.tensor([[7, 11, 23], [80, 120, 220]], device="cuda")
        embedding.queue_prefill(future_ids)
        embedding._prefill_submit_future.result()
        embedding._fetcher.wait_prefill()
        prefetched = embedding.gather(future_ids)
        embedding.wait_for_prefetch()
        expected_prefetched = (
            rows.index_select(0, future_ids.cpu().flatten())
            .reshape(*future_ids.shape, ROW_BYTES)
            .to("cuda")
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(
            prefetched, expected_prefetched, rtol=0, atol=0, equal_nan=True
        )
        assert embedding.stats_snapshot()["prefill_hits"] == future_ids.numel()
    finally:
        embedding.close()


def test_disk_graph_step_uses_static_output_and_generation_tags(tmp_path, monkeypatch):
    raw, rows = _fp8_rows(512)
    root = tmp_path
    config = SimpleNamespace(
        ple_disk_dir=str(root),
        ple_disk_hot_cache_gb=0.0,
        ple_disk_hot_frequency_file=None,
        ple_disk_stats_log_interval=0,
        to_dict=lambda: {"model_type": "synthetic-qwen4", "rows": 512},
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, attn_tp_rank=0),
    )
    embedding = Qwen4ExpDiskEmbedding(_source_embedding(512), config, 512)
    embedding.add_checkpoint_shard("synthetic.shard_0.weight", rows, 0, 512)
    embedding.finalize_image()
    try:
        static_output = embedding.allocate_output(
            (4, 16, ROW_BYTES), torch.device("cuda")
        )
        static_output.zero_()
        capture_ids = torch.zeros((4, 16), dtype=torch.long, device="cuda")
        monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: True)
        assert embedding.gather(capture_ids, out=static_output) is static_output
        assert embedding._future is None

        graph_output = torch.empty_like(static_output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output.copy_(static_output * embedding.weight_scale)

        monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: False)
        step1 = torch.arange(64, device="cuda").reshape(4, 16)
        generation = embedding.stage_graph_step(step1, static_output)
        with pytest.raises(RuntimeError, match="generation"):
            embedding.wait_for_graph_step(generation + 1)
        embedding.wait_for_graph_step(generation)
        transfer_pointers = tuple(
            tensor.data_ptr()
            for tensor in next(iter(embedding._transfer_buffers.values()))[1:]
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            raw[:64]
            .reshape(4, 16, ROW_BYTES)
            .view(torch.float8_e4m3fn)
            .to("cuda")
            .to(torch.bfloat16)
            * embedding.weight_scale
        )
        torch.testing.assert_close(
            graph_output, expected, rtol=0, atol=0, equal_nan=True
        )

        # One request ends and the batch is compacted before the next replay.
        step2 = torch.cat(
            [
                torch.arange(128, 144, device="cuda"),
                torch.arange(256, 272, device="cuda"),
                torch.zeros(32, dtype=torch.long, device="cuda"),
            ]
        ).reshape(4, 16)
        generation = embedding.stage_graph_step(step2, static_output)
        embedding.wait_for_graph_step(generation)
        assert len(embedding._transfer_buffers) == 1
        assert transfer_pointers == tuple(
            tensor.data_ptr()
            for tensor in next(iter(embedding._transfer_buffers.values()))[1:]
        )
        graph.replay()
        torch.cuda.synchronize()
        expected_rows = raw.index_select(0, step2.cpu().flatten()).reshape(
            4, 16, ROW_BYTES
        )
        expected = (
            expected_rows.view(torch.float8_e4m3fn).to("cuda").to(torch.bfloat16)
            * embedding.weight_scale
        )
        torch.testing.assert_close(
            graph_output, expected, rtol=0, atol=0, equal_nan=True
        )
    finally:
        embedding.close()


def test_disk_graph_greedy_trace_matches_pinned_for_20_prompts(tmp_path, monkeypatch):
    rows_per_head = 256
    total_rows = 16 * rows_per_head
    _, rows = _fp8_rows(total_rows)
    root = tmp_path
    config = SimpleNamespace(
        ple_disk_dir=str(root),
        ple_disk_hot_cache_gb=0.01,
        ple_disk_hot_frequency_file=None,
        ple_disk_stats_log_interval=0,
        to_dict=lambda: {"model_type": "synthetic-qwen4", "rows": total_rows},
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, attn_tp_rank=0),
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_req_to_token_pool",
        lambda: SimpleNamespace(ple_window_cache=None),
    )
    disk = Qwen4ExpDiskEmbedding(_source_embedding(total_rows), config, total_rows)
    disk.add_checkpoint_shard("synthetic.shard_0.weight", rows, 0, total_rows)
    disk.finalize_image()
    pinned = Qwen4ExpPinnedHostEmbedding(_source_embedding(total_rows))
    pinned.weight.data.copy_(rows)

    hash_module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(hash_module)
    hash_module.enable_ple_fusion = False
    hash_module.ngram_size = 3
    hash_module.heads_per_ngram = 8
    hash_module.ngram_heads = 16
    hash_module.eos_token_id = 2
    hash_module.layer_multipliers = torch.tensor(
        [1000003, 1000033, 1000037], dtype=torch.long, device="cuda"
    )
    hash_module.ngram_heads_vocab_sizes = torch.full(
        (16,), rows_per_head, dtype=torch.long, device="cuda"
    )
    hash_module.ngram_heads_offsets = torch.arange(
        0, total_rows, rows_per_head, dtype=torch.long, device="cuda"
    )
    static_output = disk.allocate_output((20, 16, ROW_BYTES), torch.device("cuda"))
    graph_output = torch.empty_like(static_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output.copy_(static_output * disk.weight_scale)
    history = torch.arange(40, device="cuda", dtype=torch.long).reshape(20, 2) + 3
    try:
        for step in range(256):
            sampled = (
                torch.arange(20, device="cuda", dtype=torch.long) * 17 + step * 13 + 3
            ) % 8192
            contexts = torch.cat([history, sampled.unsqueeze(1)], dim=1)
            ids = hash_module._hash_contexts(contexts)
            generation = disk.stage_graph_step(ids, static_output)
            disk.wait_for_graph_step(generation)
            graph.replay()
            expected = pinned.gather(ids) * pinned.weight_scale
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_output, expected, rtol=0, atol=0, equal_nan=True
            )
            history = contexts[:, 1:]
    finally:
        disk.close()


def test_real_ngram_hash_with_eos_matches_pinned_gather(tmp_path, monkeypatch):
    rows_per_head = 4096
    total_rows = 16 * rows_per_head
    _, rows = _fp8_rows(total_rows)
    root = tmp_path
    config = SimpleNamespace(
        ple_disk_dir=str(root),
        ple_disk_hot_cache_gb=0.0,
        ple_disk_hot_frequency_file=None,
        to_dict=lambda: {"model_type": "synthetic-qwen4", "rows": total_rows},
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, attn_tp_rank=0),
    )
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_req_to_token_pool",
        lambda: SimpleNamespace(ple_window_cache=None),
    )

    hash_module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(hash_module)
    hash_module.enable_ple_fusion = False
    hash_module.ngram_size = 3
    hash_module.heads_per_ngram = 8
    hash_module.ngram_heads = 16
    hash_module.eos_token_id = 2
    hash_module.layer_multipliers = torch.tensor(
        [1000003, 1000033, 1000037], dtype=torch.long, device="cuda"
    )
    hash_module.ngram_heads_vocab_sizes = torch.full(
        (16,), rows_per_head, dtype=torch.long, device="cuda"
    )
    hash_module.ngram_heads_offsets = torch.arange(
        0, total_rows, rows_per_head, dtype=torch.long, device="cuda"
    )
    generator = torch.Generator(device="cuda").manual_seed(20260827)
    contexts = torch.randint(3, 4096, (128, 3), generator=generator, device="cuda")
    contexts[0] = torch.tensor([11, 2, 19], device="cuda")
    contexts[1] = torch.tensor([2, 2, 2], device="cuda")
    contexts[2] = torch.tensor([7, 11, 2], device="cuda")
    ids = hash_module._hash_contexts(contexts)

    pinned = Qwen4ExpPinnedHostEmbedding(_source_embedding(total_rows))
    pinned.weight.data.copy_(rows)
    disk = Qwen4ExpDiskEmbedding(_source_embedding(total_rows), config, total_rows)
    disk.add_checkpoint_shard("synthetic.shard_0.weight", rows, 0, total_rows)
    disk.finalize_image()
    try:
        expected = pinned.gather(ids)
        actual = disk.gather(ids)
        disk.wait_for_prefetch()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            actual * disk.weight_scale,
            expected * pinned.weight_scale,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    finally:
        disk.close()


def test_disk_tp_shards_sum_to_unsharded_bytes(tmp_path):
    raw, rows = _fp8_rows(8192)
    root = tmp_path
    images = []
    for rank, (start, end) in enumerate(((0, 4096), (4096, 8192))):
        builder = PLEImageBuilder(root, "tp-config", rank, 2, start, end)
        builder.add_shard("test.shard_0.weight", rows, 0, rows.shape[0])
        image, _, _ = builder.finalize(1.0)
        images.append(image)
    ids = np.array([[0, 4095, 4096, 8191, -1, 9000]], dtype=np.int64)
    outputs = []
    for image in images:
        fetcher = DiskRowFetcher(image, hot_cache_gb=0, max_pages=32)
        try:
            outputs.append(fetcher.fetch(ids))
        finally:
            fetcher.close()
    assert torch.count_nonzero(outputs[0][0, 2:]).item() == 0
    assert torch.count_nonzero(outputs[1][0, :2]).item() == 0
    combined = outputs[0] + outputs[1]
    expected = torch.zeros_like(combined)
    expected[0, :4] = raw[torch.from_numpy(ids[0, :4])]
    assert torch.equal(combined, expected)


def test_disk_fetch_coalesces_pages_across_batch_and_drafts(tmp_path):
    raw, rows = _fp8_rows(256)
    root = tmp_path
    image = build_test_image(root, rows)
    fetcher = DiskRowFetcher(image, hot_cache_gb=0, max_pages=32)
    try:
        # Six rows from one page appear across two requests and three draft slots.
        ids = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
        actual = fetcher.fetch(ids)
        expected = raw.index_select(0, torch.from_numpy(ids.reshape(-1))).reshape(
            2, 3, ROW_BYTES
        )
        assert torch.equal(actual, expected)
        stats = fetcher.last_fetch_stats
        assert stats.rows_requested == 6
        assert stats.static_hits == 0
        assert stats.cold_pages == 1
        assert stats.coalesced_rows == 5
    finally:
        fetcher.close()


def test_disk_fetch_reuses_caller_owned_output(tmp_path):
    raw, rows = _fp8_rows(256)
    root = tmp_path
    image = build_test_image(root, rows)
    fetcher = DiskRowFetcher(image, hot_cache_gb=0, max_pages=32)
    ids = np.array([[1, 2], [-1, 5]], dtype=np.int64)
    output = torch.full((*ids.shape, ROW_BYTES), 0xFF, dtype=torch.uint8).pin_memory()
    try:
        actual = fetcher.fetch(ids, out=output)
        result_pointer = fetcher.reader.result.ctypes.data
        assert actual is output
        assert torch.equal(actual[0, 0], raw[1])
        assert torch.equal(actual[0, 1], raw[2])
        assert torch.equal(actual[1, 0], torch.zeros(ROW_BYTES, dtype=torch.uint8))
        assert torch.equal(actual[1, 1], raw[5])
        fetcher.fetch(ids, out=output)
        assert fetcher.reader.result.ctypes.data == result_pointer
    finally:
        fetcher.close()


def test_dynamic_wtinylfu_admission_eviction_and_exactness(tmp_path):
    raw, rows = _fp8_rows(1024)
    root = tmp_path
    image = build_test_image(root, rows)
    fetcher = DiskRowFetcher(
        image, hot_cache_gb=0, dynamic_capacity_rows=16, max_pages=32
    )
    try:
        same_set = []
        for row_id in range(rows.shape[0]):
            if int(fetcher.dynamic._set_indices(np.array([row_id]))[0]) == 0:
                same_set.append(row_id)
            if len(same_set) == 12:
                break
        protected = same_set[0]
        assert torch.equal(fetcher.fetch(np.array([protected]))[0], raw[protected])
        fetcher.dynamic.flush()
        for _ in range(12):
            assert torch.equal(fetcher.fetch(np.array([protected]))[0], raw[protected])
        fetcher.dynamic.flush()
        for row_id in same_set[1:]:
            assert torch.equal(fetcher.fetch(np.array([row_id]))[0], raw[row_id])
            fetcher.dynamic.flush()

        actual = fetcher.fetch(np.array([protected]))
        assert torch.equal(actual[0], raw[protected])
        assert fetcher.last_fetch_stats.dynamic_hits == 1
        assert fetcher.last_fetch_stats.cold_pages == 0
        assert fetcher.dynamic.rows.shape[0] == 16
    finally:
        fetcher.close()


def test_prefill_pipeline_orders_rows_and_stays_double_buffered(tmp_path):
    raw, rows = _fp8_rows(1024)
    root = tmp_path
    image = build_test_image(root, rows)
    hot_file = root / "hot.bin"
    write_hot_frequency_file(
        hot_file,
        {0: np.array([301], dtype=np.uint32)},
        fingerprint=image.header["fingerprint"],
        total_rows=image.vocab_end,
        tp_size=1,
        padding_divisor=1,
    )
    fetcher = DiskRowFetcher(
        image,
        hot_frequency_file=str(hot_file),
        hot_cache_gb=0.001,
        prefill_buffer_tokens=4,
        prefill_read_pages=2,
        max_pages=32,
    )
    prefill_calls = []
    decode_calls = []
    prefill_locked_pages = fetcher.prefill_reader.locked_pages
    decode_locked_pages = fetcher.reader.locked_pages

    @contextmanager
    def record_prefill(page_ids):
        prefill_calls.append(np.asarray(page_ids).copy())
        with prefill_locked_pages(page_ids) as pages:
            yield pages

    @contextmanager
    def record_decode(page_ids):
        decode_calls.append(np.asarray(page_ids).copy())
        with decode_locked_pages(page_ids) as pages:
            yield pages

    fetcher.prefill_reader.locked_pages = record_prefill
    fetcher.reader.locked_pages = record_decode
    try:
        static_rows = fetcher.hot.rows.clone()
        chunks = [
            np.array([100, 1, 75, 26, 51, 125], dtype=np.int64),
            np.array([201, 176, 151], dtype=np.int64),
            np.array([301, 276, 251], dtype=np.int64),
        ]
        for ids in chunks:
            assert fetcher.submit_prefill(ids)
            fetcher.wait_prefill()
        assert len(fetcher._prefill_slots) == 2
        assert sum(slot["rows"].numel() for slot in fetcher._prefill_slots) == (
            2 * 4 * 16 * ROW_BYTES
        )
        assert all(slot["state"] == "ready" for slot in fetcher._prefill_slots)
        assert prefill_calls
        assert all(call.size <= 2 for call in prefill_calls)
        assert all(np.all(np.diff(call) >= 0) for call in prefill_calls)

        current = chunks[-1][::-1].copy()
        actual = fetcher.fetch(current)
        assert torch.equal(actual, raw.index_select(0, torch.from_numpy(current)))
        assert fetcher.last_fetch_stats.static_hits == 1
        assert fetcher.last_fetch_stats.prefill_hits == current.size - 1
        assert fetcher.last_fetch_stats.cold_pages == 0
        assert torch.equal(fetcher.hot.rows, static_rows)

        cold = np.array([900, 925], dtype=np.int64)
        fetcher.fetch(cold)
        assert decode_calls
        assert fetcher.reader is not fetcher.prefill_reader
        assert fetcher.prefill_reader.max_pages == 2
    finally:
        fetcher.close()


def test_decode_ring_progresses_while_prefill_ring_is_blocked(tmp_path):
    raw, rows = _fp8_rows(512)
    root = tmp_path
    image = build_test_image(root, rows)
    fetcher = DiskRowFetcher(
        image,
        hot_cache_gb=0,
        prefill_buffer_tokens=4,
        prefill_read_pages=2,
        max_pages=8,
    )
    entered = threading.Event()
    release = threading.Event()
    prefill_locked_pages = fetcher.prefill_reader.locked_pages

    @contextmanager
    def block_prefill(page_ids):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the prefill reader")
        with prefill_locked_pages(page_ids) as pages:
            yield pages

    fetcher.prefill_reader.locked_pages = block_prefill
    try:
        assert fetcher.submit_prefill(np.array([1, 26, 51], dtype=np.int64))
        assert entered.wait(timeout=5)
        decode_ids = np.array([300, 325], dtype=np.int64)
        actual = fetcher.fetch(decode_ids)
        assert torch.equal(actual, raw.index_select(0, torch.from_numpy(decode_ids)))
        assert fetcher.last_fetch_stats.cold_pages == 2
    finally:
        release.set()
        fetcher.close()


def test_disk_fails_closed_on_corruption_fingerprint_and_short_image(tmp_path):
    _, rows = _fp8_rows(128)
    root = tmp_path
    image = build_test_image(root, rows)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        open_ple_image(image.path, expected_fingerprint="not-the-checkpoint")

    fetcher = DiskRowFetcher(image, hot_cache_gb=0, max_pages=8)
    try:
        fd = os.open(image.path, os.O_RDWR)
        try:
            original = os.pread(fd, 1, PAGE_BYTES + 7)
            os.pwrite(fd, bytes([original[0] ^ 0xFF]), PAGE_BYTES + 7)
            os.fsync(fd)
        finally:
            os.close(fd)
        with pytest.raises(IOError, match="checksum mismatch"):
            fetcher.fetch(np.array([[0]], dtype=np.int64))
    finally:
        fetcher.close()

    with image.path.open("r+b") as handle:
        handle.truncate(image.path.stat().st_size - 1)
    with pytest.raises(IOError, match="short image|size mismatch"):
        open_ple_image(image.path)


def test_graph_replay_smaller_than_capture_uses_padded_lookup_extent(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import (
        CudaGraphReplayInput,
        ForwardMode,
    )
    from sglang.srt.models.qwen4_exp import Qwen4ExpModel, Qwen4ExpPLELayer

    device = torch.device("cuda")
    lookup_tokens = 4
    heads = 16
    row_width = 10
    staged = []

    offloaded = Qwen4ExpDiskEmbedding.__new__(Qwen4ExpDiskEmbedding)
    nn.Module.__init__(offloaded)
    offloaded.stage_graph_step = lambda ids, out: staged.append(ids.clone()) or 1
    offloaded.wait_for_graph_step = lambda generation: None
    offloaded.reset_graph_step = lambda: None

    ngram = SimpleNamespace(
        ngram_embedding=offloaded,
        ngram_heads=heads,
        compute_ngram_ids=lambda batch: torch.arange(
            batch.processed_tokens * heads, device=device, dtype=torch.long
        ).view(batch.processed_tokens, heads),
        _prepare_embedding_lookup=lambda ids, forward_batch, physical_tokens: (
            ids,
            physical_tokens,
        ),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = ngram
    layer.ple_embed_dim = heads * row_width
    layer._prefetch_stream = torch.cuda.Stream()
    layer._graph_prefetch_buffer = None
    layer._graph_prefetch_buffers = {
        lookup_tokens: torch.zeros(
            (lookup_tokens, heads * row_width),
            dtype=torch.bfloat16,
            device=device,
        )
    }
    layer._graph_replay_generation = None
    layer._graph_replay_stage_expected = False
    layer._graph_replay_lookup_tokens = None
    layer._graph_replay_prefetch_buffer = None
    layer._graph_replay_steps = 0
    layer._graph_lookup_validation_interval = 1
    layer._graph_lookup_validation_due = set()
    layer._validate_graph_staging = False
    layer._completed_graph_lookup_validation = deque()
    layer._completed_graph_embedding_validation = deque()
    layer._graph_validation_free_slots = deque()
    captured_ids = torch.zeros((lookup_tokens, heads), dtype=torch.long, device=device)
    layer._graph_lookup_id_buffers = {lookup_tokens: captured_ids}
    captured_rows = torch.zeros(
        (lookup_tokens, heads * row_width), dtype=torch.bfloat16, device=device
    )
    layer._graph_embedding_snapshot_buffers = {lookup_tokens: captured_rows}
    replay_ids = torch.arange(
        lookup_tokens * heads, device=device, dtype=torch.long
    ).view(lookup_tokens, heads)
    replay_rows = torch.full_like(captured_rows, 3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_ids.copy_(replay_ids)
        captured_rows.copy_(replay_rows)
    layer._pending_graph_lookup_validation = None
    layer._pending_graph_embedding_validation = None

    pool = SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda indices: indices.long(),
        get_ngram_context=lambda indices: torch.zeros(
            (indices.numel(), 2), dtype=torch.long, device=device
        ),
    )
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)

    runtime = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        num_token_non_padded_cpu=1,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        global_dp_buffer_len=None,
        out_cache_loc=torch.tensor([20], dtype=torch.long, device=device),
        req_pool_indices=torch.tensor([7], dtype=torch.long, device=device),
    )
    replay = CudaGraphReplayInput(
        padded_num_tokens=lookup_tokens,
        input_ids=torch.tensor([17, 0, 0, 0], dtype=torch.long, device=device),
        req_pool_indices=torch.tensor([7, 0, 0, 0], dtype=torch.long, device=device),
        seq_lens=torch.tensor([9, 1, 1, 1], dtype=torch.long, device=device),
        seq_lens_sum=12,
        out_cache_loc=torch.tensor([20, 0, 0, 0], dtype=torch.long, device=device),
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=None,
        runtime_forward_batch=runtime,
    )
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    nn.Module.__init__(model)
    model.ple_ngram_size = 3
    model.ple_ngram_eos_token_id = 2
    model._ple_layers = lambda: iter([layer])

    legacy_batch = qwen4_exp_module._prepare_ple_batch(
        replay.input_ids,
        runtime,
        ngram_size=model.ple_ngram_size,
        ngram_eos_token_id=model.ple_ngram_eos_token_id,
    )
    legacy_ids = ngram.compute_ngram_ids(legacy_batch)
    assert legacy_batch.processed_tokens == 1
    assert legacy_ids.shape[0] == 1
    assert 1 not in layer._graph_prefetch_buffers
    assert 1 not in layer._graph_lookup_id_buffers
    assert 1 not in layer._graph_embedding_snapshot_buffers

    model.prepare_cuda_graph_replay(replay)

    assert staged[0].shape == (lookup_tokens, heads)
    assert layer._graph_replay_lookup_tokens == lookup_tokens
    assert layer._pending_graph_lookup_validation[0] == lookup_tokens
    layer._pending_graph_embedding_validation = (
        lookup_tokens,
        replay_rows.clone(),
    )

    model.wait_cuda_graph_replay()
    graph.replay()
    model.finish_cuda_graph_replay()
    model.reset_cuda_graph_replay()

    assert layer._pending_graph_lookup_validation is None
    assert layer._pending_graph_embedding_validation is None
    assert len(layer._completed_graph_lookup_validation) == 1
    assert len(layer._completed_graph_embedding_validation) == 1

    torch.cuda.current_stream().synchronize()

    model.prepare_cuda_graph_replay(replay)

    assert not layer._completed_graph_lookup_validation
    assert not layer._completed_graph_embedding_validation
    model.reset_cuda_graph_replay()


def test_graph_lookup_validation_records_an_async_host_result():
    layer = qwen4_exp_module.Qwen4ExpPLELayer.__new__(qwen4_exp_module.Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    lookup_ids = torch.arange(32, dtype=torch.long, device="cuda").view(2, 16)
    layer._pending_graph_lookup_validation = (2, lookup_ids.clone())
    layer._completed_graph_lookup_validation = deque()
    layer._pending_graph_embedding_validation = None
    layer._completed_graph_embedding_validation = deque()
    layer._graph_validation_free_slots = deque()
    layer._graph_lookup_id_buffers = {2: lookup_ids}

    with profile(activities=[ProfilerActivity.CPU]) as finish_profile:
        layer.finish_cuda_graph_replay()
    assert len(layer._completed_graph_lookup_validation) == 1

    layer._completed_graph_lookup_validation[0][2].synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as consume_profile:
        layer.validate_cuda_graph_replay()

    operation_names = {event.key for event in finish_profile.key_averages()} | {
        event.key for event in consume_profile.key_averages()
    }
    forbidden = {
        "aten::item",
        "aten::_local_scalar_dense",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    }
    assert operation_names.isdisjoint(forbidden)
    assert not layer._completed_graph_lookup_validation


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
