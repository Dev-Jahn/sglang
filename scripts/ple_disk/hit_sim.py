#!/usr/bin/env python3
"""Build a PLE hot-row frequency file from a token-id stream.

The input token file has no header. It is a contiguous array of little-endian
signed 32-bit token IDs. Insert the model EOS token between documents so PLE
history does not cross document boundaries.

The metadata JSON contains ``multipliers`` (3 int64 values), ``vocab_sizes``
(16 int64 values), ``offsets`` (16 int64 values), and ``eos_token_id``. These
values can be copied from the loaded PLE module or checkpoint tensors. The
output is the runtime ``PLHOT001`` file accepted by
``--ple-disk-hot-frequency-file``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sglang.srt.models.qwen4_ple_disk import ROW_BYTES, write_hot_frequency_file
from sglang.srt.models.qwen4_ple_hash import PLEMetadata, hash_token_stream_numpy


def load_metadata(path: Path) -> PLEMetadata:
    document = json.loads(path.read_text())
    if "ngram_size" not in document:
        raise ValueError("PLE metadata must record ngram_size")
    ngram_size = int(document["ngram_size"])
    if ngram_size != 3:
        raise ValueError(
            f"Qwen4 PLE hit simulation requires ngram_size=3, got {ngram_size}"
        )
    return PLEMetadata(
        multipliers=np.asarray(document["multipliers"], dtype=np.int64),
        vocab_sizes=np.asarray(document["vocab_sizes"], dtype=np.int64),
        offsets=np.asarray(document["offsets"], dtype=np.int64),
        eos_token_id=int(document["eos_token_id"]),
        ngram_size=ngram_size,
    )


def open_counts(work_dir: Path, metadata: PLEMetadata) -> list[np.memmap]:
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for head, size in enumerate(metadata.vocab_sizes):
        path = work_dir / f"head{head:02d}.u64"
        array = np.memmap(path, mode="w+", dtype=np.uint64, shape=(int(size),))
        array[:] = 0
        counts.append(array)
    return counts


def count_rows(
    token_path: Path,
    metadata: PLEMetadata,
    counts: list[np.memmap],
    chunk_tokens: int,
) -> int:
    tokens = np.memmap(token_path, mode="r", dtype="<i4")
    history = np.empty(0, dtype=np.int64)
    for start in range(0, tokens.size, chunk_tokens):
        end = min(start + chunk_tokens, tokens.size)
        chunk = np.asarray(tokens[start:end], dtype=np.int64)
        combined = np.concatenate((history, chunk)) if history.size else chunk
        row_ids = hash_token_stream_numpy(combined, metadata)[history.size :]
        for head, count_array in enumerate(counts):
            local = row_ids[:, head] - metadata.offsets[head]
            np.add.at(count_array, local, 1)
        history = combined[-(metadata.ngram_size - 1) :].copy()
        chunk_index = start // chunk_tokens
        if chunk_index == 0 or end == tokens.size or chunk_index % 100 == 0:
            print(f"counted {end:,}/{tokens.size:,} tokens", flush=True)
    for array in counts:
        array.flush()
    return int(tokens.size)


def select_rows(
    counts: list[np.memmap], metadata: PLEMetadata, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
    nonzero_rows = sum(int(np.count_nonzero(array)) for array in counts)
    capacity = min(capacity, nonzero_rows)
    if capacity == 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint64)
    max_frequency = max(int(array.max()) for array in counts)
    histogram = np.zeros(max_frequency + 1, dtype=np.int64)
    for array in counts:
        local = np.bincount(
            np.asarray(array, dtype=np.int64), minlength=max_frequency + 1
        )
        histogram[: local.size] += local
    selected_above = 0
    threshold = 0
    for frequency in range(max_frequency, 0, -1):
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


def split_ranks(
    ids: np.ndarray,
    frequencies: np.ndarray,
    total_rows: int,
    tp_size: int,
    divisor: int,
) -> dict[int, np.ndarray]:
    padded_rows = (total_rows + divisor - 1) // divisor * divisor
    if padded_rows % tp_size:
        raise ValueError(
            f"padded PLE row count {padded_rows} is not divisible by TP={tp_size}"
        )
    rows_per_rank = padded_rows // tp_size
    result = {}
    for rank in range(tp_size):
        start = rank * rows_per_rank
        end = min(total_rows, start + rows_per_rank)
        mask = (ids >= start) & (ids < end)
        rank_ids = ids[mask]
        rank_frequencies = frequencies[mask]
        order = np.lexsort((rank_ids, np.bitwise_not(rank_frequencies)))
        result[rank] = rank_ids[order]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--budget-gib", type=float, default=8.0)
    parser.add_argument("--padding-divisor", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=250_000)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    counts = open_counts(args.work_dir, metadata)
    token_count = count_rows(args.tokens, metadata, counts, args.chunk_tokens)
    capacity = int(args.budget_gib * (1 << 30) // ROW_BYTES)
    ids, frequencies = select_rows(counts, metadata, capacity)
    ranks = split_ranks(
        ids,
        frequencies,
        int(metadata.vocab_sizes.sum()),
        args.tp_size,
        args.padding_divisor,
    )
    write_hot_frequency_file(
        args.output,
        ranks,
        fingerprint=args.fingerprint,
        total_rows=int(metadata.vocab_sizes.sum()),
        tp_size=args.tp_size,
        padding_divisor=args.padding_divisor,
    )
    print(
        json.dumps(
            {
                "tokens": token_count,
                "selected_rows": int(ids.size),
                "per_rank": {
                    str(rank): int(values.size) for rank, values in ranks.items()
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
