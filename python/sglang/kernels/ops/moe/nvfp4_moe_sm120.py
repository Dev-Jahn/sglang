from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.common import is_sm120_supported

MAX_LOCAL_EXPERTS = 512


@dataclass
class Nvfp4MoeWorkspace:
    x_q: torch.Tensor
    x_scale: torch.Tensor
    fc1: torch.Tensor
    fc1_split: torch.Tensor
    act_q: torch.Tensor
    act_scale: torch.Tensor
    fc2: torch.Tensor
    output: torch.Tensor
    pair_experts: torch.Tensor
    group_rows: torch.Tensor
    group_pairs: torch.Tensor
    expert_counts: torch.Tensor
    group_experts: torch.Tensor
    group_offsets: torch.Tensor
    num_groups: torch.Tensor
    barriers: torch.Tensor
    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
    ) -> "Nvfp4MoeWorkspace":
        if max_tokens <= 0 or max_tokens > 16:
            raise ValueError(f"max_tokens must be in [1, 16], got {max_tokens}")
        if hidden_size % 256 or intermediate_size % 64:
            raise ValueError(
                "hidden_size must be divisible by 256 and intermediate_size by 64"
            )
        pairs = max_tokens * top_k
        return cls(
            x_q=torch.empty(
                max_tokens, hidden_size // 2, dtype=torch.uint8, device=device
            ),
            x_scale=torch.empty(
                max_tokens, hidden_size // 16, dtype=torch.uint8, device=device
            ),
            fc1=torch.empty(
                pairs, 2 * intermediate_size, dtype=torch.float32, device=device
            ),
            fc1_split=torch.empty(
                pairs, 2 * intermediate_size, dtype=torch.float32, device=device
            ),
            act_q=torch.empty(
                pairs, intermediate_size // 2, dtype=torch.uint8, device=device
            ),
            act_scale=torch.empty(
                pairs, intermediate_size // 16, dtype=torch.uint8, device=device
            ),
            fc2=torch.empty(
                pairs, hidden_size, dtype=torch.float32, device=device
            ),
            output=torch.empty(
                max_tokens, hidden_size, dtype=torch.bfloat16, device=device
            ),
            pair_experts=torch.empty(pairs, dtype=torch.int32, device=device),
            group_rows=torch.empty(pairs, dtype=torch.int32, device=device),
            group_pairs=torch.empty(
                MAX_LOCAL_EXPERTS, pairs, dtype=torch.int32, device=device
            ),
            expert_counts=torch.empty(
                MAX_LOCAL_EXPERTS, dtype=torch.int32, device=device
            ),
            group_experts=torch.empty(pairs, dtype=torch.int32, device=device),
            group_offsets=torch.empty(pairs, dtype=torch.int32, device=device),
            num_groups=torch.empty(1, dtype=torch.int32, device=device),
            barriers=torch.zeros(12, dtype=torch.int32, device=device),
            max_tokens=max_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(
            getattr(self, field.name).data_ptr()
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        )


@cache_once
def _jit_nvfp4_moe_module(hidden_size: int, intermediate_size: int, top_k: int):
    if not is_sm120_supported():
        raise RuntimeError("nvfp4_moe_sm120 requires an SM120 GPU")
    args = make_cpp_args(hidden_size, intermediate_size, top_k)
    return load_jit(
        "nvfp4_moe_sm120",
        *args,
        cuda_files=["moe/nvfp4_moe_sm120.cuh"],
        cuda_wrappers=[
            ("nvfp4_moe_sm120", f"Nvfp4MoeKernel<{args}>::run"),
        ],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=["-O3"],
    )


def nvfp4_moe_sm120(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    input_scale_1: torch.Tensor,
    input_scale_2: torch.Tensor,
    g1_alpha: torch.Tensor,
    g1_alpha_up: torch.Tensor,
    g2_alpha: torch.Tensor,
    expert_map: torch.Tensor,
    workspace: Nvfp4MoeWorkspace,
) -> torch.Tensor:
    if x.shape[0] > workspace.max_tokens:
        raise ValueError(
            f"workspace holds {workspace.max_tokens} tokens, got {x.shape[0]}"
        )
    if (
        workspace.top_k != topk_ids.shape[1]
        or workspace.hidden_size != x.shape[1]
        or workspace.intermediate_size != w2_weight.shape[2] * 2
    ):
        raise ValueError("workspace does not match the MoE shape")
    module = _jit_nvfp4_moe_module(
        workspace.hidden_size, workspace.intermediate_size, workspace.top_k
    )
    module.nvfp4_moe_sm120(
        x,
        topk_ids,
        topk_weights,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        input_scale_1,
        input_scale_2,
        g1_alpha,
        g1_alpha_up,
        g2_alpha,
        expert_map,
        workspace.x_q,
        workspace.x_scale,
        workspace.fc1,
        workspace.fc1_split,
        workspace.act_q,
        workspace.act_scale,
        workspace.fc2,
        workspace.output,
        workspace.pair_experts,
        workspace.group_rows,
        workspace.group_pairs,
        workspace.expert_counts,
        workspace.group_experts,
        workspace.group_offsets,
        workspace.num_groups,
        workspace.barriers,
    )
    return workspace.output[: x.shape[0]]


def prepare_nvfp4_moe_sm120(
    *,
    max_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> Nvfp4MoeWorkspace:
    _jit_nvfp4_moe_module(hidden_size, intermediate_size, top_k)
    return Nvfp4MoeWorkspace.allocate(
        max_tokens=max_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )


__all__ = [
    "Nvfp4MoeWorkspace",
    "nvfp4_moe_sm120",
    "prepare_nvfp4_moe_sm120",
]
