"""Benchmark the SM120 QSA direct-paged sparse decode path."""

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
POC_PROBE = Path("/home/jahn/workspace/qwen3.8-poc/scripts/qsa_trtllm_sm120_probe.py")
sys.path.insert(0, str(REPO_ROOT / "python"))

# Load this before the probe inserts the primary checkout into sys.path.
from sglang.srt.layers.attention.qsa.sparse_attn import (  # noqa: E402
    qsa_sparse_decode_triton,
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("qsa_sm120_probe", POC_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {POC_PROBE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()


def _triton_call(case):
    return qsa_sparse_decode_triton(
        case["q"],
        case["k"],
        case["v"],
        case["req_to_token"],
        case["req_indices"],
        case["indices"],
        case["seq_lens"],
        PROBE.SOFTMAX_SCALE,
    )


def _graph_check(case):
    for _ in range(3):
        _triton_call(case)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _triton_call(case)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _triton_call(case)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    graph.replay()
    torch.cuda.synchronize()
    drift = (output.float() - first.float()).abs().max().item()
    return {
        "finite": bool(torch.isfinite(output).all()),
        "replay_max_abs_delta": drift,
    }


def _check_case(rows, args):
    case = PROBE.make_case(rows)
    reference = PROBE.reference_fp32(case)
    floor = PROBE.bf16_noise_floor(reference)

    first = _triton_call(case)
    second = _triton_call(case)
    torch.cuda.synchronize()
    triton_error = PROBE.errors(first, reference)
    deterministic = bool(torch.equal(first, second))
    triton_ms = PROBE.bench(
        lambda: _triton_call(case), warmup=args.warmup, iters=args.iterations
    )

    xqa = PROBE.TrtllmPath(case, backend="auto")
    xqa.full()
    torch.cuda.synchronize()
    xqa_full_ms = PROBE.bench(xqa.full, warmup=args.warmup, iters=args.iterations)
    xqa_core_ms = PROBE.bench(xqa.attn, warmup=args.warmup, iters=args.iterations)

    flash = PROBE.FlashAttnPath(case)
    flash.full()
    torch.cuda.synchronize()
    flash_ms = PROBE.bench(flash.full, warmup=args.warmup, iters=args.iterations)

    graph = _graph_check(case) if rows in (1, 16, 64) else None
    numeric_ok = (
        triton_error["finite"]
        and deterministic
        and triton_error["max_abs"] <= floor["max_abs"] + args.floor_slack
    )
    latency_ok = rows not in (1, 16) or triton_ms <= xqa_full_ms
    batch64_ok = rows != 64 or triton_ms <= args.batch64_limit_ms
    graph_ok = graph is None or (
        graph["finite"] and graph["replay_max_abs_delta"] == 0.0
    )
    return {
        "rows": rows,
        "bf16_floor_max_abs": floor["max_abs"],
        "triton_max_abs": triton_error["max_abs"],
        "finite": triton_error["finite"],
        "deterministic": deterministic,
        "triton_ms": triton_ms,
        "xqa_full_ms": xqa_full_ms,
        "xqa_core_ms": xqa_core_ms,
        "flash_ms": flash_ms,
        "cuda_graph": graph,
        "numeric_ok": numeric_ok,
        "latency_ok": latency_ok,
        "batch64_ok": batch64_ok,
        "graph_ok": graph_ok,
        "passed": numeric_ok and latency_ok and batch64_ok and graph_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--floor-slack", type=float, default=2e-3)
    parser.add_argument("--batch64-limit-ms", type=float, default=0.200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"kernel={inspect.getfile(qsa_sparse_decode_triton)}")
    print(PROBE.env_line())
    results = [_check_case(rows, args) for rows in args.rows]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(
            "| rows | Triton ms | XQA full ms | XQA core ms | flash ms | "
            "max abs | bf16 floor | graph drift | result |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
        for result in results:
            graph = result["cuda_graph"]
            drift = "n/a" if graph is None else f'{graph["replay_max_abs_delta"]:.3g}'
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f'| {result["rows"]} | {result["triton_ms"]:.4f} | '
                f'{result["xqa_full_ms"]:.4f} | {result["xqa_core_ms"]:.4f} | '
                f'{result["flash_ms"]:.4f} | {result["triton_max_abs"]:.6f} | '
                f'{result["bf16_floor_max_abs"]:.6f} | {drift} | {status} |'
            )
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
