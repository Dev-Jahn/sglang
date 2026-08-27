# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from unittest import mock

import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

from sglang.kernels.ops.moe.nvfp4_moe_sm120 import (
    Nvfp4MoeWorkspace,
    nvfp4_moe_sm120,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import dequantize_nvfp4_to_dtype
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 256
INTERMEDIATE = 320
LOCAL_EXPERTS = 8
GLOBAL_EXPERTS = 16
TOP_K = 4
MAX_TOKENS = 16


def _quantize_weight(weight: torch.Tensor):
    global_scale = torch.tensor(
        2688.0 / weight.abs().max().item(), dtype=torch.float32, device="cuda"
    )
    packed, scale = fp4_quantize(weight, global_scale)
    dequant = dequantize_nvfp4_to_dtype(
        packed, scale, global_scale, torch.float32
    )
    return packed, scale.view(torch.float8_e4m3fn), 1.0 / global_scale, dequant


def _quant_roundtrip(value: torch.Tensor, global_scale: torch.Tensor):
    packed, scale = fp4_quantize(value.to(torch.bfloat16), global_scale)
    return dequantize_nvfp4_to_dtype(
        packed, scale, global_scale, torch.float32
    )


class _Fixture:
    def __init__(self):
        torch.manual_seed(20260827)
        w13_q = []
        w2_q = []
        w13_sf = []
        w2_sf = []
        w13_alpha = []
        w2_alpha = []
        w13_ref = []
        w2_ref = []
        for _ in range(LOCAL_EXPERTS):
            up = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
            ) / 8
            gate = torch.randn(
                INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
            ) / 8
            down = torch.randn(
                HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
            ) / 8
            q13, sf13, alpha13, ref13 = _quantize_weight(
                torch.cat((up, gate), dim=0)
            )
            q2, sf2, alpha2, ref2 = _quantize_weight(down)
            w13_q.append(q13)
            w2_q.append(q2)
            w13_sf.append(sf13)
            w2_sf.append(sf2)
            w13_alpha.append(alpha13)
            w2_alpha.append(alpha2)
            w13_ref.append(ref13)
            w2_ref.append(ref2)

        self.w13 = torch.stack(w13_q)
        self.w2 = torch.stack(w2_q)
        self.w13_sf = torch.stack(w13_sf)
        self.w2_sf = torch.stack(w2_sf)
        self.w13_weight_scale = torch.stack(w13_alpha).float()
        self.w2_weight_scale = torch.stack(w2_alpha).float()
        self.w13_ref = torch.stack(w13_ref)
        self.w2_ref = torch.stack(w2_ref)
        self.input_scale_1 = torch.tensor(1024.0, dtype=torch.float32, device="cuda")
        self.input_scale_2 = torch.tensor(64.0, dtype=torch.float32, device="cuda")
        self.g1_alpha = self.w13_weight_scale / self.input_scale_1
        self.g2_alpha = self.w2_weight_scale / self.input_scale_2
        self.expert_map = torch.full(
            (GLOBAL_EXPERTS,), -1, dtype=torch.int32, device="cuda"
        )
        self.expert_map[
            torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], device="cuda")
        ] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
        self.workspace = Nvfp4MoeWorkspace.allocate(
            max_tokens=MAX_TOKENS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            device=torch.device("cuda"),
        )

    def run(self, x, ids, weights):
        return nvfp4_moe_sm120(
            x=x,
            topk_ids=ids,
            topk_weights=weights,
            w13_weight=self.w13,
            w2_weight=self.w2,
            w13_scale=self.w13_sf,
            w2_scale=self.w2_sf,
            input_scale_1=self.input_scale_1,
            input_scale_2=self.input_scale_2,
            g1_alpha=self.g1_alpha,
            g1_alpha_up=self.g1_alpha,
            g2_alpha=self.g2_alpha,
            expert_map=self.expert_map,
            workspace=self.workspace,
        )

    def reference(self, x, ids, weights):
        x_dequant = _quant_roundtrip(x, self.input_scale_1)
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device="cuda")
        for token in range(x.shape[0]):
            for slot in range(TOP_K):
                global_expert = int(ids[token, slot])
                if global_expert < 0 or global_expert >= GLOBAL_EXPERTS:
                    continue
                local_expert = int(self.expert_map[global_expert])
                if local_expert < 0:
                    continue
                fc1 = x_dequant[token] @ self.w13_ref[local_expert].T
                up, gate = fc1.split(INTERMEDIATE)
                act = torch.nn.functional.silu(gate) * up
                act = _quant_roundtrip(act[None], self.input_scale_2)[0]
                out[token] += (
                    float(weights[token, slot])
                    * (act @ self.w2_ref[local_expert].T)
                )
        return out


@unittest.skipUnless(get_device_sm() == 120, "requires SM120")
class TestNvfp4MoeSm120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _Fixture()

    @classmethod
    def tearDownClass(cls):
        del cls.fixture
        torch.cuda.empty_cache()

    @staticmethod
    def _routes(tokens: int, case: str):
        base = torch.tensor(
            [
                [0, 2, 4, 6],
                [8, 10, 12, 14],
                [0, 8, 4, 12],
                [2, 10, 6, 14],
            ],
            dtype=torch.int32,
            device="cuda",
        ).repeat(math.ceil(tokens / 4), 1)[:tokens]
        if case == "duplicate":
            base[:, 1] = base[:, 0]
        elif case == "skewed":
            base.fill_(0)
        elif case == "nonlocal_invalid":
            base[:, 1] = 1
            base[:, 2] = -1
            base[:, 3] = GLOBAL_EXPERTS + 3
        weights = torch.rand(tokens, TOP_K, dtype=torch.float32, device="cuda")
        weights /= weights.sum(dim=-1, keepdim=True)
        return base, weights

    def test_numerics_and_layouts(self):
        torch.manual_seed(17)
        for tokens in (1, 4, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            for case in ("balanced", "duplicate", "skewed", "nonlocal_invalid"):
                with self.subTest(tokens=tokens, case=case):
                    ids, weights = self._routes(tokens, case)
                    actual = self.fixture.run(x, ids, weights).float()
                    reference = self.fixture.reference(x, ids, weights)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0.20, atol=0.025)

    def test_no_worse_than_cutlass(self):
        torch.manual_seed(29)
        for tokens in (1, 4, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            ids, weights = self._routes(tokens, "balanced")
            identity_ids = torch.div(ids, 2, rounding_mode="floor").to(torch.int32)
            current = torch.empty(
                tokens, HIDDEN, dtype=torch.bfloat16, device="cuda"
            )
            cutlass_fused_moe(
                input=x,
                token_selected_experts=identity_ids,
                token_final_scales=weights,
                fc1_expert_weights=self.fixture.w13.view(torch.long),
                fc2_expert_weights=self.fixture.w2.view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[
                    self.fixture.input_scale_1,
                    self.fixture.w13_sf.view(torch.int32),
                    self.fixture.g1_alpha,
                    self.fixture.input_scale_2,
                    self.fixture.w2_sf.view(torch.int32),
                    self.fixture.g2_alpha,
                ],
                output=current,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1 << (tokens - 1).bit_length(),
                use_fused_finalize=True,
            )
            candidate = self.fixture.run(x, ids, weights).float()
            reference = self.fixture.reference(x, ids, weights)
            current_error = current.float() - reference
            candidate_error = candidate - reference
            self.assertLessEqual(
                candidate_error.norm() / reference.norm(),
                current_error.norm() / reference.norm() + 1e-3,
            )
            self.assertLessEqual(
                candidate_error.abs().max(), current_error.abs().max() + 1e-3
            )

    def test_graph_replay_keeps_addresses(self):
        torch.manual_seed(41)
        addresses = self.fixture.workspace.data_ptrs()
        for tokens in (1, 4, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            ids, weights = self._routes(tokens, "balanced")
            self.fixture.run(x, ids, weights)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.fixture.run(x, ids, weights)
            expected = output.clone()
            allocated = torch.cuda.memory_allocated()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(addresses, self.fixture.workspace.data_ptrs())
            self.assertEqual(allocated, torch.cuda.memory_allocated())
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_flashinfer_cutlass_small_row_dispatch(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(4, "balanced")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=self.fixture.w13,
            w2_weight=self.fixture.w2,
            output_dtype=torch.bfloat16,
            quant_scales=[
                self.fixture.input_scale_1,
                self.fixture.w13_sf,
                self.fixture.g1_alpha,
                self.fixture.input_scale_2,
                self.fixture.w2_sf,
                self.fixture.g2_alpha,
            ],
            apply_routed_scaling_factor=False,
            g1_alpha_up=self.fixture.g1_alpha,
            smallm_workspace=self.fixture.workspace,
            smallm_expert_map=self.fixture.expert_map,
        )
        runner_config = MoeRunnerConfig(
            num_experts=GLOBAL_EXPERTS,
            num_local_experts=LOCAL_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            top_k=TOP_K,
            activation="silu",
            is_gated=True,
        )
        cutlass = mock.Mock(side_effect=AssertionError("unexpected cutlass fallback"))
        with mock.patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
            "_flashinfer_cutlass_fused_moe",
            return_value=(cutlass, object()),
        ):
            actual = _run_flashinfer_cutlass(
                dispatch_output=dispatch,
                quant_info=quant_info,
                runner_config=runner_config,
            )
        self.assertFalse(cutlass.called)
        reference = self.fixture.reference(x, ids, weights)
        torch.testing.assert_close(actual.float(), reference, rtol=0.20, atol=0.025)


if __name__ == "__main__":
    unittest.main()
