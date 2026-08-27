"""The PLE offload table lives in host RAM and is read by a GPU kernel.

Two properties have to hold for that to work, and both broke in production:

  1. The pages must not all land on the NUMA node local to the GPU. A
     multi-GiB node-local pinned allocation can exhaust a small node, which
     shows up as an OOM kill, or as the NVIDIA driver failing to allocate the
     GPU page tables that map the pinning. A partially mapped pinning then
     faults asynchronously (Xid 31, MMU FAULT_PDE at a host address) and
     reaches Python as "CUDA error: an illegal memory access was encountered"
     at whatever unrelated call happens to synchronize next.
  2. The raw host pointer must stay readable from a device kernel, so the
     mapping has to outlive every view of it.

Neither needs a model, a checkpoint, or tensor parallelism to exercise.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.numa_utils import (
    allocate_interleaved_pinned_table,
    numa_page_counts,
    online_numa_nodes,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-a-test-cuda")

# Large enough that the interleave is visible over page-granularity noise,
# small enough to allocate on a busy host.
_TABLE_ROWS = 1 << 20
_TABLE_DIM = 160


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestInterleavedPinnedTable(CustomTestCase):
    def test_pages_span_every_numa_node(self):
        nodes = online_numa_nodes()
        if len(nodes) < 2:
            self.skipTest(f"single NUMA node host: {nodes}")
        tensor, buffer = allocate_interleaved_pinned_table(
            (_TABLE_ROWS, _TABLE_DIM), torch.float8_e4m3fn
        )
        try:
            self.assertIsNotNone(
                buffer, "interleaved allocation fell back to node-local pinning"
            )
            counts = numa_page_counts(buffer.ptr, buffer.nbytes)
            self.assertGreater(
                len(counts),
                1,
                f"table pinned on a single NUMA node: {counts}",
            )
        finally:
            del tensor
            if buffer is not None:
                buffer.release()

    def test_device_kernel_reads_the_host_table(self):
        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        tensor, buffer = allocate_interleaved_pinned_table(
            (_TABLE_ROWS, _TABLE_DIM), torch.float8_e4m3fn
        )
        try:
            self.assertTrue(tensor.is_pinned(), "table is not page-locked")
            rows = [0, _TABLE_ROWS // 2, _TABLE_ROWS - 1]
            reference = torch.arange(_TABLE_DIM, dtype=torch.bfloat16)
            for offset, row in enumerate(rows):
                tensor[row] = (reference + offset).to(torch.float8_e4m3fn)

            ids = torch.tensor(rows, dtype=torch.long, device="cuda")
            out = torch.empty(
                (len(rows), _TABLE_DIM), dtype=torch.bfloat16, device="cuda"
            )
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                tensor.data_ptr(),
                ids,
                out,
                embedding_dim=_TABLE_DIM,
                tp_vocab_start=0,
                tp_vocab_end=_TABLE_ROWS,
                is_fp8=True,
                BLOCK_D=256,
            )
            torch.cuda.synchronize()
            for offset in range(len(rows)):
                expected = (reference + offset).to(torch.float8_e4m3fn).to(
                    torch.bfloat16
                )
                self.assertTrue(torch.equal(out[offset].cpu(), expected))
        finally:
            del tensor
            if buffer is not None:
                buffer.release()

    def test_disabled_flag_uses_node_local_pinning(self):
        with envs.SGLANG_PLE_OFFLOAD_NUMA_INTERLEAVE.override(False):
            tensor, buffer = allocate_interleaved_pinned_table(
                (1024, _TABLE_DIM), torch.float8_e4m3fn
            )
        try:
            self.assertIsNone(buffer)
            self.assertTrue(tensor.is_pinned())
        finally:
            del tensor


if __name__ == "__main__":
    unittest.main()
