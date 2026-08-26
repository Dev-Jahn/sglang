"""Correctness and graph tests for Qwen4-Exp QSA MQA scoring."""

import os

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.kernels.ops.attention.qsa.mqa import (
    triton_qsa_mqa_decode,
    triton_qsa_mqa_prefill,
)
from sglang.srt.layers.attention.qsa import mqa as mqa_module
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
    torch_qsa_mqa_decode,
    torch_qsa_mqa_prefill,
)

HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
TOPK = 512


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the production Triton dispatch is specific to SM120")


def _assert_logits(actual, expected):
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=2e-2, rtol=2e-2)


def _assert_topk_sets(actual, expected, starts, ends, topk=TOPK):
    """Require exact sets unless the reference cutoff is numerically tied."""

    finite = torch.isfinite(expected)
    max_error = (actual[finite] - expected[finite]).abs().max()
    for row in range(expected.shape[0]):
        start = int(starts[row])
        end = int(ends[row])
        width = min(topk, end - start)
        if width <= 0:
            continue
        actual_idx = torch.topk(actual[row, start:end], width).indices + start
        expected_values, expected_local = torch.topk(expected[row, start:end], width)
        expected_idx = expected_local + start
        if set(actual_idx.tolist()) == set(expected_idx.tolist()):
            continue
        cutoff = expected_values[-1]
        tied = (expected[row, start:end] - cutoff).abs() <= 2 * max_error
        stable = torch.nonzero((expected[row, start:end] > cutoff) & ~tied).flatten()
        stable_expected = set((stable + start).tolist())
        assert stable_expected.issubset(set(actual_idx.tolist()))


def _make_prefill_case():
    torch.manual_seed(41)
    lengths = torch.tensor([1, 127, 513, 2049, 8192], dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    sequence_ids = torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.long)
    starts = offsets[:-1].index_select(0, sequence_ids).cuda()
    ends = offsets[1:].index_select(0, sequence_ids).cuda()
    q = torch.randn(
        sequence_ids.numel(), HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(int(offsets[-1]), 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return q, k, starts, ends


def _make_decode_case():
    torch.manual_seed(42)
    lengths = torch.tensor(
        [1, 513, 2047, 8193, 32768], device="cuda", dtype=torch.int32
    )
    max_len = int(lengths.max())
    max_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE
    batch = lengths.numel()
    page_table = torch.arange(
        batch * max_pages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_pages)
    cache = torch.randn(
        batch * max_pages,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q, torch.zeros_like(q)], dim=1)
    return q, cache, page_table, lengths, max_len


def test_qsa_mqa_triton_ragged_prefill_logits_and_topk():
    _require_sm120()
    case = _make_prefill_case()
    expected = torch_qsa_mqa_prefill(*case)
    actual = triton_qsa_mqa_prefill(*case)
    _assert_logits(actual, expected)
    _assert_topk_sets(actual, expected, case[2], case[3])
    os.environ["SGLANG_QSA_MQA_BACKEND"] = "triton"
    try:
        _assert_logits(qsa_mqa_prefill(*case), expected)
    finally:
        os.environ.pop("SGLANG_QSA_MQA_BACKEND", None)


def test_qsa_mqa_triton_ragged_decode_logits_topk_and_graph():
    _require_sm120()
    case = _make_decode_case()
    expected = torch_qsa_mqa_decode(*case)
    actual = triton_qsa_mqa_decode(*case)
    _assert_logits(actual, expected)
    starts = torch.zeros_like(case[3])
    _assert_topk_sets(actual, expected, starts, case[3])

    for _ in range(3):
        triton_qsa_mqa_decode(*case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = triton_qsa_mqa_decode(*case)
    graph.replay()
    torch.cuda.synchronize()
    first = replay_output.clone()
    case[0].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(first, replay_output)
    scaled_expected = torch_qsa_mqa_decode(*case)
    _assert_logits(replay_output, scaled_expected)


def test_qsa_mqa_sm120_auto_uses_triton(monkeypatch):
    _require_sm120()
    case = _make_decode_case()
    monkeypatch.delenv("SGLANG_QSA_MQA_BACKEND", raising=False)
    expected = triton_qsa_mqa_decode(*case)
    actual = qsa_mqa_decode(*case)
    assert torch.equal(actual, expected)
    assert not mqa_module._IMPORT_TILELANG


def test_qsa_mqa_sm120_does_not_fall_back_from_triton(monkeypatch):
    _require_sm120()
    case = _make_prefill_case()
    monkeypatch.delenv("SGLANG_QSA_MQA_BACKEND", raising=False)
    monkeypatch.setattr(mqa_module, "is_sm120_supported", lambda: True)

    def fail(*args, **kwargs):
        raise RuntimeError("triton launch failed")

    monkeypatch.setattr(mqa_module, "triton_qsa_mqa_prefill", fail)
    with pytest.raises(RuntimeError, match="triton launch failed"):
        qsa_mqa_prefill(*case)
