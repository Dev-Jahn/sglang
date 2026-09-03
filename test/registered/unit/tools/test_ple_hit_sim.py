"""PLE hot-row simulator tests."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models import qwen4_ple_disk as disk
from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding
from sglang.srt.models.qwen4_ple_hash import (
    PLEMetadata,
    hash_contexts_numpy,
    hash_token_stream_numpy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fp8_rows(count: int) -> torch.Tensor:
    raw = torch.arange(count * disk.ROW_BYTES, dtype=torch.uint8).reshape(
        count, disk.ROW_BYTES
    )
    raw[(raw & 0x7F) == 0x7F] = 0
    return raw.view(torch.float8_e4m3fn)


def test_image_metadata_round_trips_into_hit_sim(tmp_path):
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_metadata", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)
    metadata = {
        "multipliers": [3, 5, 7],
        "vocab_sizes": [13, 17],
        "offsets": [0, 13],
        "eos_token_id": 2,
        "ngram_size": 3,
    }
    builder = disk.PLEImageBuilder(
        tmp_path,
        "metadata",
        0,
        1,
        0,
        25,
        ple_metadata=metadata,
    )
    builder.add_shard("shard", _fp8_rows(25), 0, 25)
    image, _, _ = builder.finalize(0.5)

    path = image.path.parent / "ple-metadata.json"
    document = json.loads(path.read_text())
    loaded = hit_sim.load_metadata(path)

    assert document["fingerprint"] == image.header["fingerprint"]
    assert loaded.fingerprint == image.header["fingerprint"]
    assert loaded.multipliers.tolist() == metadata["multipliers"]
    assert loaded.vocab_sizes.tolist() == metadata["vocab_sizes"]
    assert loaded.offsets.tolist() == metadata["offsets"]


def test_numpy_hash_matches_production_on_seeded_token_stream(monkeypatch):
    metadata = PLEMetadata(
        multipliers=np.array([1000003, 1000033, 1000037], dtype=np.int64),
        vocab_sizes=np.array([101 + 2 * index for index in range(16)], dtype=np.int64),
        offsets=np.cumsum(
            np.r_[np.int64(0), np.array([101 + 2 * index for index in range(15)])]
        ),
        eos_token_id=2,
    )
    rng = np.random.default_rng(20260830)
    tokens = rng.integers(3, 32000, size=4096, dtype=np.int64)
    tokens[::127] = 2
    tokens[1::509] = 2
    module = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.enable_ple_fusion = False
    module.ngram_size = 3
    module.heads_per_ngram = 8
    module.ngram_heads = 16
    module.eos_token_id = metadata.eos_token_id
    module.layer_multipliers = torch.from_numpy(metadata.multipliers)
    module.ngram_heads_vocab_sizes = torch.from_numpy(metadata.vocab_sizes)
    module.ngram_heads_offsets = torch.from_numpy(metadata.offsets)
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_req_to_token_pool",
        lambda: SimpleNamespace(ple_window_cache=None),
    )

    contexts = np.full((tokens.size, 3), metadata.eos_token_id, dtype=np.int64)
    contexts[:, 2] = tokens
    contexts[1:, 1] = tokens[:-1]
    contexts[2:, 0] = tokens[:-2]
    actual = module._hash_contexts(torch.from_numpy(contexts)).numpy()
    expected = hash_contexts_numpy(contexts, metadata)
    assert np.array_equal(actual, expected)
    assert np.array_equal(hash_token_stream_numpy(tokens, metadata), expected)


def test_hit_sim_selects_accessed_rows_and_splits_tp_ranks(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_test", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)

    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([4, 3], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        eos_token_id=2,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "multipliers": metadata.multipliers.tolist(),
                "vocab_sizes": metadata.vocab_sizes.tolist(),
                "offsets": metadata.offsets.tolist(),
                "eos_token_id": metadata.eos_token_id,
                "ngram_size": metadata.ngram_size,
                "fingerprint": "test-image",
            }
        )
    )
    tokens = np.array([1, 2, 3, 1, 0, 2], dtype="<i4")
    token_path = tmp_path / "tokens.i32"
    tokens.tofile(token_path)
    count_dir = tmp_path / "counts"
    output_path = tmp_path / "hot.bin"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--tokens",
            str(token_path),
            "--metadata",
            str(metadata_path),
            "--work-dir",
            str(count_dir),
            "--output",
            str(output_path),
            "--tp-size",
            "2",
            "--budget-gib",
            "0.00001",
            "--padding-divisor",
            "4",
            "--chunk-tokens",
            "2",
        ],
    )

    hit_sim.main()

    count_files = [
        np.memmap(
            count_dir / f"head{head:02d}.u64",
            mode="r",
            dtype=np.uint64,
            shape=(int(size),),
        )
        for head, size in enumerate(metadata.vocab_sizes)
    ]
    assert all(array.dtype == np.uint64 for array in count_files)
    ids, frequencies = hit_sim.select_rows(count_files, metadata, capacity=10)
    ranks = hit_sim.split_ranks(ids, frequencies, total_rows=7, tp_size=2, divisor=4)
    for rank, expected in ranks.items():
        assert np.array_equal(
            disk.read_hot_frequency_file(
                output_path, rank, expected_fingerprint="test-image"
            ),
            expected,
        )


def test_hit_sim_counting_records_each_head(tmp_path):
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_count", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)
    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([4, 3], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        eos_token_id=2,
    )
    token_path = tmp_path / "tokens.i32"
    np.array([1, 2, 3, 1], dtype="<i4").tofile(token_path)
    counts = hit_sim.open_counts(tmp_path / "counts", metadata)
    hit_sim.count_rows(token_path, metadata, counts, chunk_tokens=2)
    assert counts[0].tolist() == [1, 1, 0, 2]
    assert counts[1].tolist() == [0, 3, 1]


def test_hit_sim_fills_each_rank_from_its_share_of_a_skewed_corpus():
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_skew", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)
    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([4, 4], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        eos_token_id=2,
    )
    counts = [
        np.array([100, 90, 80, 70], dtype=np.uint64),
        np.array([4, 3, 2, 1], dtype=np.uint64),
    ]

    ranks = hit_sim.select_rows_by_rank(
        counts,
        metadata,
        capacity=4,
        total_rows=8,
        tp_size=2,
        divisor=4,
    )

    assert ranks[0].tolist() == [0, 1]
    assert ranks[1].tolist() == [4, 5]


def test_hit_sim_selection_handles_a_large_peak_frequency():
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_peak", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)
    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([4, 3], dtype=np.int64),
        offsets=np.array([0, 4], dtype=np.int64),
        eos_token_id=2,
    )
    counts = [
        np.array([10**12, 7, 7, 1], dtype=np.uint64),
        np.array([8, 7, 0], dtype=np.uint64),
    ]
    ids, frequencies = hit_sim.select_rows(counts, metadata, capacity=4)
    assert ids.tolist() == [0, 4, 1, 2]
    assert frequencies.tolist() == [10**12, 8, 7, 7]


def test_hit_sim_selection_matches_the_previous_ordering():
    script = Path(__file__).resolve().parents[4] / "scripts/ple_disk/hit_sim.py"
    spec = importlib.util.spec_from_file_location("qwen4_ple_hit_sim_order", script)
    hit_sim = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hit_sim)
    metadata = PLEMetadata(
        multipliers=np.array([3, 5, 7], dtype=np.int64),
        vocab_sizes=np.array([5, 4], dtype=np.int64),
        offsets=np.array([0, 5], dtype=np.int64),
        eos_token_id=2,
    )
    counts = [
        np.array([0, 3, 9, 3, 1], dtype=np.uint64),
        np.array([3, 7, 0, 3], dtype=np.uint64),
    ]

    def previous_selection(capacity):
        maximum = max(int(array.max()) for array in counts)
        histogram = np.zeros(maximum + 1, dtype=np.int64)
        for array in counts:
            local = np.bincount(
                np.asarray(array, dtype=np.int64), minlength=maximum + 1
            )
            histogram[: local.size] += local
        selected_above = 0
        threshold = 0
        for frequency in range(maximum, 0, -1):
            if selected_above + int(histogram[frequency]) >= capacity:
                threshold = frequency
                break
            selected_above += int(histogram[frequency])
        tie_remaining = capacity - selected_above
        ids = []
        frequencies = []
        for array, offset in zip(counts, metadata.offsets):
            local_ids = np.flatnonzero(array > threshold)
            if tie_remaining:
                tied = np.flatnonzero(array == threshold)
                take = min(tie_remaining, tied.size)
                local_ids = np.concatenate((local_ids, tied[:take]))
                tie_remaining -= take
            ids.append((local_ids + int(offset)).astype(np.uint32))
            frequencies.append(np.asarray(array[local_ids], dtype=np.uint64))
        global_ids = np.concatenate(ids)
        global_frequencies = np.concatenate(frequencies)
        order = np.lexsort((global_ids, np.bitwise_not(global_frequencies)))
        return global_ids[order], global_frequencies[order]

    expected_ids, expected_frequencies = previous_selection(5)
    actual_ids, actual_frequencies = hit_sim.select_rows(counts, metadata, 5)
    assert np.array_equal(actual_ids, expected_ids)
    assert np.array_equal(actual_frequencies, expected_frequencies)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
