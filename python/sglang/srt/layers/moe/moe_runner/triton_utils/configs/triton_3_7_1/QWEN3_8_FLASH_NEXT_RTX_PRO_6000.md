# Qwen3.8-Flash-Next on RTX PRO 6000 Blackwell

These FP8 blockwise configs were tuned on an NVIDIA RTX PRO 6000 Blackwell
Server Edition with Triton 3.7.1, hidden size 2560, MoE intermediate size 640,
top-k 10, and block shape `[128, 128]`. Each file contains results for the
standard token-count grid plus 8192.

The valid expert-parallel layouts are E=64 (TP8/EP8), E=128 (TP4/EP4), and
E=256 (TP2/EP2). TP4 without expert parallelism would use E=512/N=160, but its
320-wide gate-up output is not divisible by the FP8 activation group size 128;
the Triton path rejects that shape before configuration selection, so no
E=512/N=160 FP8 config is included.

The tuner used a bounded 32-config search seeded from the existing Qwen/H200
tiles and including the SGLang default heuristic. On TP4/EP4, the selected
configs beat the default at token counts 1, 16, 64, 256, and 2048.
