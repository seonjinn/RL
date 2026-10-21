# DFlash on Qwen3.5 pays for 46 KV cache groups, not for drafting

## Why This Experiment

On Qwen3.5-35B-A3B GRPO at gpu_memory_utilization 0.8, DSpark K4 cut the step time by 4.4% and DFlash K3 added 6.4%. Both drafters have the same size and depth and similar acceptance (about 2.6 tokens per step). One structural difference stands out. DFlash has one full-attention layer whose KV spec matches no target layer. vLLM 0.29's group-size rule then falls to a group size of 1, so the KV pool splits into 46 groups (30 of them Gated DeltaNet). DSpark gets 8 groups and the baseline gets 4. The V2 model runner builds attention metadata for each group on every step. The hypothesis: **this per-group cost, not the drafter, makes DFlash slow on hybrid targets.**

`bench.py` runs vLLM 0.29 in-process on one GB200 per arm, with the GRPO engine settings and 2048-token decodes at batch 32, 64 and 128.

---

## Fewer groups recover the full DFlash speedup

| Arm | Groups | tok/s B32 | B64 | B128 | step ms B32 | B128 | attn-md CPU ms/step | accept |
|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 3564 | 5432 | 8740 | 9.0 | 14.6 | 0.23-0.30 | - |
| dflash_k3 | 46 | 2942 | 3746 | 5289 | 20.5 | 26.6 | 13.8-16.2 | 2.51-2.69 |
| dflash_k3_g5 | 10 | 4939 | 6077 | 8220 | 12.2 | 16.1 | 2.9-3.4 | 2.51-2.69 |

Job 8122498. `dflash_k3_g5` changes only one thing: vLLM's group size is forced to 5. The acceptance stays identical, and the KV pool shrinks a little (381K to 348K tokens). Throughput goes up by 1.55-1.68x, and DFlash goes from 17% slower than the baseline at B32 to 39% faster.

The step time falls by about the same amount as the attention-metadata CPU time (20.5 to 12.2 ms vs 14.2 to 2.9 ms). **The GPU waits on Python for most of a DFlash step.** The grouping is the cause.

---

## The cost is about 0.45 ms of CPU per GDN group

| Arm | Groups | GDN groups | tok/s B32 | B64 | B128 | attn-md CPU ms/step | accept |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 3 | 3565 | 5433 | 8737 | 0.22-0.23 | - |
| dspark_k4 | 8 | 5 | 4207 | 5905 | 9727 | 2.4-2.7 | 2.57-2.64 |
| dflash_k3 | 46 | 30 | 2939 | 3653 | 5754 | 14.2-14.6 | 2.51-2.69 |
| dflash_k7 | 46 | 30 | 2974 | 3337 | 4746 | 13.8-14.4 | 3.11-3.37 |

Job 8123219. With SpecDec on, the GDN builder takes its speculative path, and each build costs about 0.45-0.5 ms. DSpark pays this 5 times per step and DFlash 30 times. DFlash K7 accepts more tokens than K3 but is not faster, because the per-step CPU cost does not change with K.

Round 1 showed DSpark K4 dropping at B128 (5579 tok/s). That was a bench artifact. The arm used the K3 capture list, whose largest size (512) is below 128 x (1+4). With its own K4 list, DSpark K4 reaches 9727 tok/s at B128. The GRPO runs always used the matching list.

`VLLM_KV_CACHE_LAYOUT=BLHNC` selects vLLM's packed grouping planner, which would avoid the 1-layer group. Every BLHNC arm fails at startup on this target. Hybrid models set the attention block to about 1056-1088 tokens so that one attention page covers one mamba page. The kernels then split each manager block into smaller kernel blocks, and BLHNC interleaves the layers. vLLM rejects this layout because it "does not store blocks as dense, unpadded pages". The block size cannot shrink on a hybrid target, so the layout route is closed.

---

## Upstream vLLM removed the per-group cost

vLLM main already fixes this in three PRs:

- #57102 (v0.30.0) shares `token_to_req_indices` across groups.
- #58762 (v0.31.0) builds the GDN metadata once per spec and copies only the state indices for the other groups.
- #58763 (v0.32.0) slices pure spec-decode rows.

`git apply` does not work on 0.29. [`../vllm_gdn_metadata_reuse`](../../vllm_gdn_metadata_reuse/README.md) backports the three changes as an import-time source patch.

---

## The backport makes DFlash 1.4-1.8x faster

| Arm | Groups | tok/s B32 | B64 | B128 | step ms B32 | B128 | attn-md CPU ms/step | accept B32 |
|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 3519 | 5384 | 8727 | 9.1 | 14.7 | 0.22-0.24 | - |
| baseline_reuse | 4 | 3561 | 5433 | 8784 | 9.0 | 14.6 | 0.22-0.25 | - |
| dflash_k3 | 46 | 3284 | 3882 | 4702 | 19.8 | 25.8 | 13.6-15.8 | 2.71 |
| dflash_k3_reuse | 46 | 5184 | 6790 | 8331 | 12.5 | 15.1 | 2.1-2.3 | 2.71 |
| dflash_k7 | 46 | 3224 | 4353 | 4932 | 20.0 | 23.4 | 14.0-14.7 | 3.56 |
| dflash_k7_reuse | 46 | 4984 | 5397 | 6980 | 12.9 | 19.4 | 2.3-2.7 | 3.53 |
| dspark_k4 | 8 | 3961 | 6249 | 9786 | 11.8 | 16.3 | 2.5-2.7 | 2.72 |
| dspark_k4_reuse | 8 | 4315 | 7198 | 9370 | 12.0 | 15.4 | 0.75-0.87 | 2.59 |

Job 8123912 runs each arm with and without the backport. The groups stay the same, and only the per-group metadata build goes away. The DFlash attention-metadata time drops from about 14 ms to about 2.2 ms per step, and DFlash K3 throughput goes up by 1.58x at B32, 1.75x at B64 and 1.77x at B128. **This recovers the same speedup as the forced group size of 5, without a change to the KV layout.** DFlash K3 with the backport is 1.47x the baseline at B32 and 5% below it at B128.

The baseline has no speculative path, and its GDN build costs only about 0.07 ms per group, so the backport does not change it. DSpark K4 drops from 2.6 ms to 0.8 ms of metadata time and gains 9-15% at B32 and B64. At B128 it is 4% slower, within the variance of a single bench run.

Greedy decoding is the correctness check: 16 prompts, 256 tokens, run twice in the same process.

| Run | Run 1 vs run 2 | vs base arm |
|---|---|---|
| dflash_k3_reuse | 16/16 | 16/16, bit-exact |
| dflash_k7_reuse | 16/16 | 16/16, bit-exact |
| baseline_reuse | 16/16 | 2/16, mean matched prefix 75 tokens |
| dspark_k4_reuse | 16/16 | 1/16, mean matched prefix 84 tokens |

Every arm repeats itself exactly inside one process. DFlash with the backport matches DFlash without it token for token. The baseline and DSpark pairs split after about 80 tokens. Two unrelated arms (dflash_k3 vs dflash_k7, which must give the same tokens because the target model verifies every token) also match only 2/16 with a prefix of 78. So the split has the size of the noise between two processes, and the outputs stay coherent. The open question was whether that noise explains all of it, or whether the reused groups read a buffer they should not.

---

## The backport is bit-exact, and the baseline splits from itself

Job 8124837 runs each arm again in a new process (`@2` on another GPU, `@3` on the same GPU) and records the top-2 logprobs at every greedy step. The `*_reuse_verify` arms also rebuild each reused GDN group with the full `build()` and compare every metadata field with the `update_block_table()` result on the same step.

| Run | Reference | Same tokens | Mean matched prefix | Max logprob gap before split | Max top-2 margin at split |
|---|---|---|---|---|---|
| baseline@2 | baseline | 5/16 | 138/256 | 0.21 | 0.125 |
| baseline@3 | baseline | 3/16 | 143/256 | 0.12 | 0.250 |
| baseline_reuse | baseline | 3/16 | 102/256 | 0.17 | 0.250 |
| baseline_reuse_verify | baseline | 3/16 | 142/256 | 0.12 | 0.250 |
| dspark_k4@2 | dspark_k4 | 16/16 | 256/256 | 6.3e-05 | no split |
| dspark_k4_reuse | dspark_k4 | 16/16 | 256/256 | 0 | no split |
| dspark_k4_reuse_verify | dspark_k4 | 16/16 | 256/256 | 0 | no split |

| Run | Field checks | Mismatched fields |
|---|---|---|
| baseline_reuse_verify | 1512 | none |
| dspark_k4_reuse_verify | 2320 | none |
| dflash_k3_reuse_verify | 17284 | none |

**The baseline does not repeat itself across processes.** A second run of the unchanged baseline matches only 3-5 of 16 sequences, and its logprobs drift by up to 0.21 before the split. The baseline with the backport falls in the same band. Every split happens at a top-2 margin of 0.25 or less, which is a near-tie. The baseline runs the target on FlashInfer attention, and the spec arms run it on Triton attention. So the drift most likely comes from the FlashInfer kernels, not from the backport.

DSpark repeats itself across processes to within 6e-05, and DSpark with the backport matches DSpark without it with a logprob gap of exactly 0. The round 3 DSpark split did not repeat. The field checks find no difference between the reused metadata and a full rebuild in 21,116 comparisons over all three targets, including the CUDA graph capture steps. **The backport changes the CPU cost and nothing else.** The Q35 GRPO arms now run with it (jobs 8125226-8125231).

---

## Key Takeaway

**On a hybrid GDN target, DFlash's slowdown comes from vLLM 0.29 building GDN metadata once per KV cache group (30 groups), not from the drafter.** With 10 groups, DFlash K3 is 1.39x the baseline at B32 and 6% below it at B128. The backport of the upstream fix gives the same result with 46 groups (1.47x at B32, 5% below at B128), and its greedy tokens and logprobs match the unpatched path exactly. Any hybrid SSM target with a drafter whose layers match no target spec (Qwen3.5, Nemotron-H) pays this cost on vLLM before 0.31. Compare SpecDec on these targets with the upstream fix or the backport.
