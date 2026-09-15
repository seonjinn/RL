# New Draft online-cadence rerun plan

Goal: repeat the approved Qwen3-8B 200-step, eleven-arm comparison with the
rp25-44000 B8 DFlash/DSpark exports, without reusing old run identities.

Architecture: reuse the verified cadence workload and renderer, override only
new-checkpoint contracts and the approved horizon, and gate production on real
model update/refit and checkpoint/resume. No policy/runtime API changes here.

Source: `README.md` and the user's approved 200-step eleven-condition request.
Implementation remains in this isolated worktree; stable branches are untouched.

## Fixed comparison

- Qwen3-8B target snapshot b968826d9c46dd6066d109eabc6255188de91218.
- DAPOMath17K, seed42, PPS2/GPS4/GBS8, OSL1024, total4096, input2048.
- One node/four GB200, train TP2/CP1, generation TP1, packing=false.
- Preserve the old PIECEWISE/eager/S8 configuration for the first compatibility
  gate. This is not a CUDA Graph coverage certificate or the Q30 DAPO40K study.
- Baseline no SpecDec plus each method frozen/always/fixed5/10/20; no adaptive.
- New B8: DFlash gamma7, DSpark block8, serving K5, window2048; clear public revisions.
- W&B nvidia/sna-specdec, separate New Draft group, short names and fresh IDs.

## Tasks and gates

September 15 update: the user explicitly requested parallel submissions to
classify working/failing configurations. Submit the eleven fresh 200-step arms
independently, without resume-job dependencies. Resume validation remains a
separate correctness task, not a claim that continuation is validated. Retain
the patched checkpoint memory transition and checkpoints at 50/100/150/200.
The earlier production-blocking resume gate below is superseded by this request.

1. Add `study.py` and `test_study.py`. Test eleven unique arms, 200-step horizon,
   exact update steps, unchanged workload, no-SpecDec baseline, new paths,
   cleared revisions, and independent B8 training/K5 serving.
   Run `python -m unittest research.qwen3_8b_rp25_swa.test_study -v`.
2. Add `run_online_canary.sbatch`: same immutable archive/bundle/container as
   passed gate7150387; independent node-local actor environments; direct GRPO
   using study overrides. Two-step always-online for both methods. Capture
   checkpoint and cadence receipts; do not label process exit as study success.
3. Commit/sign-off/push exact files, pull remote, preflight assets/auth/quota,
   run sbatch test-only, then submit two independent canaries. Monitor startup.
4. Before 200-step submission, verify model changes, update/refit receipts,
   next rollout, generation-side window semantics, and actual checkpoint resume.
   Keep production blocked if any evidence is missing. No automatic production
   submission is hidden in the canary script.

## Current evidence

7150387 completed exit0 after42m33s: 58 provider/config/body tests and 12 selected
attention tests passed. Source cdbaa1767, NVIDIA GB200, torch2.11.0+cu130.
This is not yet a full Qwen train/refit/rollout result.
