# Qwen3-30B-A3B Thinking SWE trajectory-collection rollout-only benchmark

This experiment compares a matched no-speculation baseline with DFlash K5/K7
and DSpark K5/K7 on the official NeMo-Gym SWE1 pivot workload. It pins the
first 500 rows of NVIDIA's `Nemotron-RL-Super-Training-Blends/swe1.jsonl`,
inherits the native Thinking SWE trajectory-collection path from PR #3733, and
uses five checked-in YAML overlays. It is not the PR #3243 generation-only eval
benchmark. It changes no full-workload semantics outside the speculative
configuration, the legacy SWE1 agent alias, and derived CUDA Graph capture
sizes.

Official-SWE baseline and DFlash K5 canaries completed, and the first full
baseline/DFlash jobs reached trajectory collection. Those FULL-graph results
are preliminary because vLLM 0.25.1 cannot safely replay non-causal FlashInfer
draft attention. DSpark K5 recovery canary `6485632` completed with PIECEWISE
graphs and custom all-reduce disabled. The final comparison applies those two
settings to all five arms before K7 canary and the 500-task matrix. No final
speedup is recorded yet. PR #3733 explicitly did not validate the 30B runtime
or performance path; successful experiment canaries are not retroactive proof
attached to either PR.

## Contract

- Source history contains exact PR #3733 head
  `b580dd8927b88c996470d315e74d57bf0cb4090e`.
- The authoritative SWE base, PR overlay, and PR launcher are unchanged from
  that head.
- Target, DFlash, DSpark, data, source commit, and container bytes are verified
  before scheduling.
- The pinned `ray.sub` preserves the container image home with
  `--no-container-mount-home`; no host Python or uv cache is mounted over the
  image runtime.
- Before Ray starts, every head and worker container imports the pinned driver
  dependencies through `/opt/nemo_rl_venv/bin/python` and `vllm` through the
  exact prebuilt async vLLM actor interpreter under `/opt/ray_venvs`.
- Output and state are bound beneath the approved user Lustre `experiments`
  prefix; alternate state directories are rejected.
- Baseline is the same rollout-only overlay with `speculative_config=null`.
- Scheduler topology is two nodes, four GPUs per node, segment size one; the
  inherited generation resources use one node at TP2.
- W&B routes to `nvidia/sna-specdec`.
- Every arm must pass the identical `sbatch --test-only` contract before its
  exclusive pre-submission reservation can be created.
- Full submission remains locked until baseline and DFlash K5 canaries complete
  successfully.

## OCI-HSG sequence

Run from a clean recursive checkout under `/home/sna` at the signed experiment
commit. Use an output root on the user's Lustre allocation. Replace the shell
variables with the FairShare-selected account and compute-verified immutable
container identity.

Run `preflight` on an allocated compute node: it streams the target, draft, and
container bytes to SHA256 and must not perform those heavy reads on a login
node. The manifest intentionally fails closed while any expected digest is a
placeholder.

Any manual Pyxis preflight or diagnostic must use
`--no-container-mount-home`, matching `ray.sub`. Omitting it can replace the
image's `/root` with the host home. This image's uv-managed venv contains
absolute links into `/root/.local/share/uv/python` and `/root/.cache/uv`, so a
home-mounted diagnostic does not test the production bootstrap.

The driver environment intentionally does not contain vLLM. Do not force
`NEMO_RL_PY_EXECUTABLES_SYSTEM=1`; the scheduler retains actor-tier selection,
points `NEMO_RL_VENV_DIR` at `/opt/ray_venvs`, and reuses the immutable image's
prebuilt vLLM, Megatron, and NeMo Gym environments.

```bash
experiment=experiments/qwen3_30ba3b_swe_rollout_pr3733
source_commit=$(git rev-parse HEAD)
container=/lustre/path/to/immutable-nemo-rl.sqsh
container_sha256=<64-lowercase-hex>
output_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-swe-pr3733
state_dir=${output_root}/state
account=<fairshare-selected-account>

python3 ${experiment}/benchmark.py preflight \
  --repo-root "$PWD" \
  --source-commit "${source_commit}" \
  --container "${container}" \
  --container-sha256 "${container_sha256}" \
  --record "${state_dir}/preflight.json"

python3 ${experiment}/benchmark.py materialize-canary \
  --destination "${output_root}/inputs/swe1_first1.jsonl" \
  --record "${state_dir}/canary-input.json"

python3 ${experiment}/benchmark.py plan \
  --profile canary \
  --source-commit "${source_commit}" \
  --container "${container}" \
  --container-sha256 "${container_sha256}" \
  --output-root "${output_root}" \
  --preflight-record "${state_dir}/preflight.json" \
  --canary-record "${state_dir}/canary-input.json" \
  > "${output_root}/canary-plan.json"

for arm in baseline dflash_k5; do
  python3 ${experiment}/submit.py test-only \
    --plan "${output_root}/canary-plan.json" --arm "${arm}" \
    --repo-root "$PWD" --state-dir "${state_dir}" \
    --preflight-record "${state_dir}/preflight.json" \
    --account "${account}" --partition batch --time 04:00:00
done
```

The canary bounds the dataset to its canonical first prompt and preserves the
native trajectory-collection validation sampling contract of one generation.

Inspect the two test-only records before changing the explicit action to
`submit`. Collect the returned job IDs and monitor them with one filtered query
per minute over five minutes:

```bash
python3 ${experiment}/submit.py monitor \
  --job-id <baseline-job-id> --job-id <dflash-k5-job-id> \
  --state-dir "${state_dir}" \
  --campaign-id <campaign-id-from-plan> --profile canary
```

After both jobs terminate successfully, their job wrappers create matched
completion records. `benchmark.py unlock-full --state-dir "${state_dir}"
--campaign-id <campaign-id-from-plan>` must return `full-unlocked` before a full
plan can pass scheduler submission gates.

## Local verification

The repository lock targets Linux, so the authoritative locked test command is
run inside the selected OCI container. A macOS-safe dependency-free fallback is:

```bash
pytest --noconftest tests/unit/test_qwen3_30ba3b_swe_rollout_benchmark.py -q
ruff check experiments/qwen3_30ba3b_swe_rollout_pr3733 \
  tests/unit/test_qwen3_30ba3b_swe_rollout_benchmark.py
```
