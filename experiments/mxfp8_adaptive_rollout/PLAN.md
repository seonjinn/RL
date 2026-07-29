# Adaptive MXFP8 rollout execution plan

## Source and image gate

- [ ] Commit and push the NeMo-RL experiment branch.
- [ ] Build the ARM64 image from `seonjinn/vllm` commit `bc588192`.
- [ ] Verify vLLM `0.20.2`, FlashInfer `0.6.8.post1`, custom source, loader,
  and the package-relative bootstrap JSON.
- [ ] Run OCI-HSG FairShare and `sbatch --test-only`.
- [ ] Stage the registry image to an immutable `.sqsh`.
- [ ] Record registry source, source commit, staging job, retrieval time,
  squashfs SHA256, and immutable path.
- [ ] Pass the one-node/four-GPU smoke and monitor it for five minutes.

## Qwen applicability gate

- [ ] Pull the pushed NeMo-RL commit on OCI-HSG.
- [ ] Run the exact Task 3 Qwen TP1 4n4g trace recipe for one short step.
- [ ] Confirm every trace row records the bootstrap config SHA256.
- [ ] Run inventory over every process trace file.
- [ ] If inventory is empty, record `not-applicable`, skip Qwen A/B, and move
  the kernel-efficacy work to Nemotron 3 Ultra TP4.

## Qualification gate

- [ ] Shmoo every traced physical shape with at least three repeats.
- [ ] Require BF16/reference correctness and cosine similarity at least 0.999.
- [ ] Promote only median speedup of at least 1.02 versus tactic `-1`.
- [ ] Reproduce the qualified JSON bytes with `validate --check`.
- [ ] Install the qualified package-relative JSON in a rebuilt immutable image.
- [ ] Repeat the one-node/four-GPU smoke against the qualified JSON.

## Matched A/B gate

- [ ] Run one original and one adaptive warmup.
- [ ] Run at least three alternating measured original/adaptive repeats.
- [ ] Confirm the resolved-config guard passes for every pair.
- [ ] Confirm every qualified shape hits its tactic and unseen shapes use `-1`.
- [ ] Parse stable JSON and CSV summaries.
- [ ] Require correctness, higher median output-token throughput, lower median
  rollout latency, p95 rollout latency regression no greater than 2%, and no
  end-to-end step-time regression.
- [ ] Save job IDs, raw artifacts, summary tables, and the conclusion under
  the ignored local `report/` directory.
