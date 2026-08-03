# Qwen3-235B Forced 32K MXFP8 Backend Comparison

CuTeDSL achieved 620.73 tokens/sec/GPU. TRTLLM Adaptive achieved 571.69 tokens/sec/GPU (0.921x, 7.9% lower).

Both arms generated 64 x 32,768 tokens. The forced-32K trace observed five unique signatures, all covered by the qualified table. Longer OSL therefore did not expose missing shapes or recover an Adaptive performance advantage.
