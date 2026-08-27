# Nemotron3 Super EP32 Without M-LM #5515 Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Nemotron3 Super with HybridEP EP=32 completes 20 GRPO steps without hanging when `moe_hybridep_pad_uneven_dispatch_inputs=True` and M-LM PR #5515 is absent.

**Architecture:** Use the existing OCI-HSG 32-node x 4-GPU Super performance recipe and change only the HybridEP dispatcher, EP size, and uneven-dispatch toggle. Pin a validation-only Megatron-Bridge commit to an M-LM main commit that contains #5008 and #6114 but not #5515, then record every dependency SHA in the job log before training.

**Tech Stack:** NeMo-RL, Megatron-Bridge, Megatron-LM, DeepEP/HybridEP, Hydra, SLURM, OCI-HSG GB200

**Spec:** `docs/superpowers/plans/2026-08-26-super-ep32-no-5515-validation.md`

## Global Constraints

- Use DeepEP commit `17cfb817bccec3a9c247013360cc550c2bac441e`.
- M-LM must contain merged PRs #5008 and #6114 and must not contain PR #5515.
- Set `policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=true` through the NeMo-RL setup path by keeping legacy NeMo-RL pre-padding disabled.
- Preserve the existing Super 32-node x 4-GPU performance recipe except for HybridEP enablement and `expert_model_parallel_size: 32`.
- Run 20 steps with checkpointing disabled and capture exact NeMo-RL, Bridge, M-LM, and DeepEP provenance.
- Keep source under `/home`, caches under `/raid/scratch`, and durable logs/container/model data under `/lustre`.

---

### Task 1: Reproducible dependency stack

**Files:**
- Modify: `.gitmodules`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`

**Interfaces:**
- Consumes: NeMo-RL validation base `bf61938e957a5bd21a5992869003f1df06a43a53`.
- Produces: A pushed Bridge commit and NeMo-RL gitlink that reproduce the exact no-#5515 M-LM source.

- [ ] Fetch M-LM main and check out `e79cb4c1bae1afd04322d979d08cb63832991ebe`.
- [ ] Verify `git merge-base --is-ancestor 81770cb015eab05785ecd540ba929d1400a52f67 HEAD` succeeds for #5008.
- [ ] Verify `git merge-base --is-ancestor 723db5a72790aefc02f5a0228e6607eef70c0533 HEAD` succeeds for #6114.
- [ ] Verify `git merge-base --is-ancestor 278cc9128 HEAD` fails so the #5515 implementation is absent.
- [ ] Commit and push the Bridge gitlink update, then point NeMo-RL at that pushed commit.

### Task 2: Super EP32 validation configuration

**Files:**
- Create: `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-ep32-hybridep-no5515.yaml`
- Create: `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-ep32-no5515.env`
- Create: `scripts/experiments/oci-hsg/hybridep/submit_super_ep32_no5515.sh`

**Interfaces:**
- Consumes: Existing `grpo-nemotron3-super-120BA12B-32n4g.yaml` values and pushed dependency stack from Task 1.
- Produces: A secret-free, 20-step SLURM submission that asserts dependency and resolved-config provenance.

- [ ] Inherit the existing 32n4g recipe and override only dispatcher `flex`, backend `hybridep`, HybridEP SM count `32`, EP size `32`, and legacy pre-padding `false`.
- [ ] Set `MAX_STEPS=20`, `TIME_LIMIT=04:00:00`, and checkpointing disabled in the experiment environment.
- [ ] Log the resolved config and assert the installed DeepEP source contains `17cfb817bccec3a9c247013360cc550c2bac441e`.
- [ ] Run shell syntax checks and Hydra config composition locally.

### Task 3: Submit and observe the runtime test

**Files:**
- Produce remotely: `/lustre/.../super-ep32-no5515/<job-id>/`

**Interfaces:**
- Consumes: Pushed NeMo-RL validation branch from Task 2.
- Produces: SLURM job ID, first-five-minute health evidence, final step count, exit state, and hang/error classification.

- [ ] Push the signed NeMo-RL validation commit.
- [ ] SSH to `sna@sna-oci-hsg-cs.park.nvidia.com` and pull the exact branch under `/home`.
- [ ] Run `sbatch --test-only` and save the scheduler response.
- [ ] Submit once using `--gpus-per-node=4`; the repository `ray.sub` requests the full four-GPU OCI-HSG node exclusively.
- [ ] Query only this job no more than once per 60 seconds for the first five minutes.
- [ ] After completion, record `sacct` state, exit code, completed GRPO steps, and any HybridEP/NCCL/Ray fatal signature.
