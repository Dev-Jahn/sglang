"""Checkpoint-independent NumPy helpers for Qwen4 PLE hashing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007


@dataclass(frozen=True)
class PLEMetadata:
    multipliers: np.ndarray
    vocab_sizes: np.ndarray
    offsets: np.ndarray
    eos_token_id: int
    ngram_size: int = 3


def _splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_layer_multipliers(
    size: int,
    *,
    vocab_size: int,
    seed: int = 1234,
    ple_layer_index: int = 0,
) -> np.ndarray:
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(vocab_size, 1)) // 2)
    base_seed = seed + PRIME_1 * ple_layer_index
    values = []
    for index in range(size):
        initial = (base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64
        values.append(2 * (_splitmix64(initial) % half_bound) + 1)
    return np.asarray(values, dtype=np.int64)


def hash_contexts_numpy(contexts: np.ndarray, metadata: PLEMetadata) -> np.ndarray:
    """Return PLE row IDs for independent three-token contexts."""
    contexts = np.asarray(contexts, dtype=np.int64)
    heads = int(metadata.vocab_sizes.size)
    if metadata.ngram_size != 3:
        raise ValueError(
            f"Qwen4 PLE hashing requires ngram_size=3, got {metadata.ngram_size}"
        )
    if contexts.ndim != 2 or contexts.shape[1] != 3:
        raise ValueError(f"contexts must have shape [N, 3], got {contexts.shape}")
    if metadata.multipliers.shape != (3,):
        raise ValueError("multipliers must contain three values")
    if heads <= 0 or heads % 2 or metadata.offsets.shape != (heads,):
        raise ValueError("vocab_sizes and offsets must contain an even number of heads")

    current, previous, oldest = contexts[:, 2], contexts[:, 1], contexts[:, 0]
    eos = np.int64(metadata.eos_token_id)
    oldest = np.where((oldest == eos) | (previous == eos), eos, oldest)
    first, second, third = metadata.multipliers
    with np.errstate(over="ignore"):
        bigram = np.bitwise_xor(current * first, previous * second)
        trigram = np.bitwise_xor(bigram, oldest * third)
    heads_per_ngram = heads // 2
    mixed = np.concatenate(
        (
            np.repeat(bigram[:, None], heads_per_ngram, axis=1),
            np.repeat(trigram[:, None], heads_per_ngram, axis=1),
        ),
        axis=1,
    )
    return np.remainder(mixed, metadata.vocab_sizes) + metadata.offsets


def hash_token_stream_numpy(tokens: np.ndarray, metadata: PLEMetadata) -> np.ndarray:
    """Hash a token stream; EOS tokens reset history for following tokens."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    contexts = np.full((tokens.size, 3), metadata.eos_token_id, dtype=np.int64)
    contexts[:, 2] = tokens
    if tokens.size > 1:
        contexts[1:, 1] = tokens[:-1]
    if tokens.size > 2:
        contexts[2:, 0] = tokens[:-2]
    return hash_contexts_numpy(contexts, metadata)
