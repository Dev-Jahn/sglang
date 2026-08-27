#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import platform
import socket

import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

from sglang.kernels.ops.moe.nvfp4_moe_sm120 import (
    Nvfp4MoeWorkspace,
    nvfp4_moe_sm120,
)
from sglang.srt.layers.moe.topk import fused_topk


SPIKE = pathlib.Path(
    "/home/jahn/workspace/qwen3.8-poc/spikes/nvfp4_moe_small_m/bench.py"
)


def load_spike():
    spec = importlib.util.spec_from_file_location("nvfp4_moe_spike", SPIKE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    spike = load_spike()
    torch.manual_seed(20260827)
    weights = spike.load_checkpoint_rank(torch.device("cuda"))
    input_scale_1 = weights["w13_input"].max()
    input_scale_2 = weights["w2_input"].max()
    input_quant_1 = (1.0 / input_scale_1).float()
    input_quant_2 = (1.0 / input_scale_2).float()
    g1_alpha = (input_scale_1 * weights["w13_s2"]).float()
    g2_alpha = (input_scale_2 * weights["w2_s2"]).float()
    expert_map = torch.arange(spike.EXPERTS, dtype=torch.int32, device="cuda")
    workspace = Nvfp4MoeWorkspace.allocate(
        max_tokens=16,
        top_k=spike.TOP_K,
        hidden_size=spike.HIDDEN,
        intermediate_size=spike.INTERMEDIATE,
        device=torch.device("cuda"),
    )
    x_all = (
        torch.randn((16, spike.HIDDEN), dtype=torch.bfloat16, device="cuda") / 4
    ).contiguous()
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    rows = []
    outputs = {}

    for tokens in (1, 4, 16):
        x = x_all[:tokens]
        logits = F.linear(x, weights["router"])
        topk_weights, topk_ids = fused_topk(
            x, logits, spike.TOP_K, True, scoring_func="softmax"
        )
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.contiguous()
        current_out = torch.empty_like(x)

        def current():
            return cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids,
                token_final_scales=topk_weights,
                fc1_expert_weights=weights["w13"].view(torch.long),
                fc2_expert_weights=weights["w2"].view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[
                    input_quant_1,
                    weights["s13_swizzled"].view(torch.int32),
                    g1_alpha,
                    input_quant_2,
                    weights["s2_swizzled"].view(torch.int32),
                    g2_alpha,
                ],
                output=current_out,
                tp_size=spike.TP,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1 << math.ceil(math.log2(tokens)),
                use_fused_finalize=True,
            )[0]

        def candidate():
            return nvfp4_moe_sm120(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13_weight=weights["w13"],
                w2_weight=weights["w2"],
                w13_scale=weights["s13_swizzled"],
                w2_scale=weights["s2_swizzled"],
                input_scale_1=input_quant_1,
                input_scale_2=input_quant_2,
                g1_alpha=g1_alpha,
                g1_alpha_up=g1_alpha,
                g2_alpha=g2_alpha,
                expert_map=expert_map,
                workspace=workspace,
            )

        current_samples, current_graph = spike.graph_samples(
            current, flush, args.repeats
        )
        candidate_samples, candidate_graph = spike.graph_samples(
            candidate, flush, args.repeats
        )
        current()
        outputs[("current", tokens)] = current_out.clone()
        outputs[("candidate", tokens)] = candidate().clone()
        torch.cuda.synchronize()
        row = {
            "tokens": tokens,
            "unique_experts": int(torch.unique(topk_ids).numel()),
            "current": spike.summarize(current_samples),
            "candidate": spike.summarize(candidate_samples),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del current_graph, candidate_graph

    correctness = {}
    for tokens in (1, 16):
        x = x_all[:tokens]
        logits = F.linear(x, weights["router"])
        routing, ids = fused_topk(
            x, logits, spike.TOP_K, True, scoring_func="softmax"
        )
        reference = spike.reference_moe(x, ids, routing, weights)
        correctness[str(tokens)] = {
            "cutlass_vs_fp32_dequant": spike.max_errors(
                outputs[("current", tokens)], reference
            ),
            "b12x_vs_fp32_dequant": spike.max_errors(
                outputs[("candidate", tokens)], reference
            ),
            "b12x_vs_cutlass": spike.max_errors(
                outputs[("candidate", tokens)],
                outputs[("current", tokens)].float(),
            ),
        }

    correctness_acceptance = {
        "criterion": (
            "candidate max_abs and relative_l2 are each no worse than current "
            "plus 1e-3"
        ),
        "passed": all(
            correctness[str(tokens)]["b12x_vs_fp32_dequant"][metric]
            <= correctness[str(tokens)]["cutlass_vs_fp32_dequant"][metric]
            + 1e-3
            for tokens in (1, 16)
            for metric in ("max_abs", "relative_l2")
        ),
    }
    for row in rows:
        row["routed_pairs"] = row["tokens"] * spike.TOP_K
        row["cuda_graph"] = {
            "cutlass_fused": row["current"],
            "b12x_sm12x": row["candidate"],
        }

    root = pathlib.Path(__file__).resolve().parents[5]
    source = root / "python/sglang/kernels/jit/csrc/moe/nvfp4_moe_sm120.cuh"
    props = torch.cuda.get_device_properties(0)
    result = {
        "schema": "nvfp4_moe_small_m_v1",
        "environment": {
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": spike.sh(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                "4",
            ),
            "cuda_toolkit": spike.sh("nvcc", "--version").splitlines()[-1],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "flashinfer": __import__("flashinfer").__version__,
            "sglang_commit": spike.sh("git", "-C", str(root), "rev-parse", "HEAD"),
            "kernel": platform.release(),
            "host": socket.gethostname(),
        },
        "model": {
            "checkpoint": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "snapshot": spike.snapshot_dir().name,
            "layer": spike.LAYER,
            "global_experts": spike.EXPERTS,
            "local_experts": spike.EXPERTS,
            "top_k": spike.TOP_K,
            "hidden": spike.HIDDEN,
            "global_intermediate": spike.GLOBAL_INTERMEDIATE,
            "rank_intermediate": spike.INTERMEDIATE,
            "tp": spike.TP,
            "tp_rank": 0,
            "ep": 1,
            "quant": (
                "NVFP4 W4A4 group 16, E4M3 block scales, FP32 global scales"
            ),
            "w13_input_scale_max": input_scale_1.item(),
            "w2_input_scale_max": input_scale_2.item(),
        },
        "timing": {
            "method": (
                "CUDA events, 256 MiB L2 flush before each sample, compilation "
                "and two warmups excluded"
            ),
            "repeats": args.repeats,
            "cuda_graph": (
                "one fused MoE call per captured graph; L2 flush outside graph"
            ),
        },
        "rows": rows,
        "correctness": correctness,
        "correctness_acceptance": correctness_acceptance,
        "graph_capture": {
            "cutlass": {
                str(tokens): {"ok": True} for tokens in (1, 4, 16)
            },
            "b12x_sm12x": {
                str(tokens): {"ok": True} for tokens in (1, 4, 16)
            },
        },
        "candidate": {
            "name": "SGLang SM120 grouped NVFP4 fused MoE",
            "source": str(source),
            "source_sha256": spike.sha256(source),
            "dispatch_source_sha256": spike.sha256(
                root
                / "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py"
            ),
            "prior_art": "SGLang PR 36043 SM120 OMMA path",
            "backend_cutovers": "one through sixteen tokens; FlashInfer above sixteen",
            "weight_preparation": (
                "checkpoint W4A4 tensors and swizzled scales used without transcode"
            ),
            "scale_convention": (
                "weight and activation scale products cached at layer processing"
            ),
        },
    }
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
