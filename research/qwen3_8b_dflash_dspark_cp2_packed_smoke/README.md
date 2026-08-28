# Qwen3-8B DFlash/DSpark packed CP1/CP2 online smoke

This experiment validates DFlash and DSpark training at TP2, PP1, CP1 or CP2
with target sequence parallelism and sequence packing enabled. Generation
remains TP1 and PP1. Each provider runs for two GRPO steps so the log must prove a nonzero draft
loss, an actual draft parameter update, draft refit manifest construction, the
post-step-1 target-and-draft refit, CUDA Graph capture, and completion of step 2.

The harness is intentionally functional rather than a performance comparison.
It uses the public Qwen3-8B DFlash K5 and DSpark block-7 snapshots and changes no
NeMo RL product code.

## Fixed topology and runtime contract

- Product base: `26603aedd28fae9db852e4b84d85d6cfcd7729e8`
- Target: `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`
- DFlash: `z-lab/Qwen3-8B-DFlash-b16` revision `9b41424b7109f9c5413454f481b09a82b85333f4`, K5
- DSpark: `deepseek-ai/dspark_qwen3_8b_block7` revision `03326e5043815da1f81b109078b2889737c26017`, K7
- Training: one OCI-HSG GB200 node, four GPUs, TP2, PP1, CP2, sequence parallelism enabled
- Packed sequence divisibility: 16, which is a multiple of `2 * TP * CP = 8`
- Generation: colocated vLLM, TP1, PP1, drafter TP1, PIECEWISE CUDA Graph mode
- Drafter attention backend: `FLASH_ATTN`
- GRPO: two prompts, four generations per prompt, global batch size 8, two steps

## Provenance prerequisites

The checkout must be a clean recursive-submodule-complete clone under `/home`.
Its HEAD must descend from the product base above, and every changed path must
remain under this experiment directory. The runner also requires an immutable
container SHA256, exact local Hugging Face snapshots, a W&B API key inherited by
`sbatch --export=ALL`, and a fresh `/lustre` result directory.

The known OCI-HSG paths used by the public snapshot contract are:

```bash
export TARGET_SNAPSHOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
export DFLASH_SNAPSHOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
export DSPARK_SNAPSHOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/snapshots/03326e5043815da1f81b109078b2889737c26017
export CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh
export CONTAINER_SHA256=6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44
```

Verify all four paths on OCI-HSG before forecasting. The container was used by
the earlier PR11/CP validation harness, but this experiment still requires a
fresh import/runtime smoke because the product base includes a later main merge.
All four paths and the recorded container SHA were confirmed from the OCI-HSG
login node on 2026-08-27.

## Forecast and submit

After committing and pushing this experiment-only branch, prepare the exact
remote checkout and environment on OCI-HSG:

```bash
cd /home/sna/nemo-rl-q8-cp2-packed-smoke
git pull --ff-only
git submodule update --init --recursive
export REMOTE_REPO=$PWD
export EXPECTED_HEAD=$(git rev-parse HEAD)
export FINAL_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/results/q8-cp2-packed-online-smoke-$(date +%Y%m%d-%H%M%S)
export SBATCH_ACCOUNT=nemotron_n3_post
export WANDB_PROJECT=sna-specdec-cp2-validation
export CONTEXT_PARALLEL_SIZE=1  # use 2 for the CP2 qualification pair
test -n "${WANDB_API_KEY}"
```

Export the four immutable artifact variables from the previous section, then
run the scheduler forecast:

```bash
bash research/qwen3_8b_dflash_dspark_cp2_packed_smoke/submit_oci_hsg.sh --test-only
```

Only after both forecasts pass, submit the two independent provider jobs:

```bash
bash research/qwen3_8b_dflash_dspark_cp2_packed_smoke/submit_oci_hsg.sh
```

The submitter prints two job IDs and two W&B run IDs. Monitor both jobs with one
filtered scheduler query no more often than once per minute. A successful result
has `terminal_phase=complete` in each provider's `result.txt` and
`status=passed` in `evidence.json`.

## Known validation gap

No completed CP2 packed DFlash or DSpark E2E receipt is stored locally yet; the
two jobs above are the missing runtime gate. The focused AST provider contract
has six passing checks, but its literal-capability check is stale: commit
`27cdb2d1c7b9b8ae0854271f8512a34f984397c2` intentionally moved the three
capability values to `DFlashDraftConfig` and `DSparkDraftConfig`, while the test
still requires literal booleans in the provider classes. Runtime semantics are
unchanged, but the test needs a separate core/test-scope correction before a
fully green product test claim can be made.

## Local contract verification

```bash
python3 -m pytest -q \
  --confcutdir=research/qwen3_8b_dflash_dspark_cp2_packed_smoke/tests \
  research/qwen3_8b_dflash_dspark_cp2_packed_smoke/tests
bash -n \
  research/qwen3_8b_dflash_dspark_cp2_packed_smoke/run_oci_hsg.sbatch \
  research/qwen3_8b_dflash_dspark_cp2_packed_smoke/submit_oci_hsg.sh
```
