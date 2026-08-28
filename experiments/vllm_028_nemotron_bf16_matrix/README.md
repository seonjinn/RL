# vLLM 0.28 Nemotron-3 BF16 DynamicMTP matrix

This experiment evaluates standalone offline generation on Lyris GB200 with
vLLM 0.28.0. It keeps target weights in BF16 and the KV cache in FP8.

## Fixed contract

| Field | Super | Ultra |
| --- | --- | --- |
| Checkpoint | `NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` |
| Topology | TP2, DP1, 1 node | TP8, DP1, EP, Ray, 2 nodes |
| Weight / KV precision | BF16 / FP8 | BF16 / FP8 |

Both models use the following workload contract:

- ISL/OSL: `1000/10000` and `10000/1000`
- concurrency: `1, 2, 4, 8, 16, 32, 128, 512`
- methods: baseline, static MTP K=`1,2,3,4,5`, and DynamicMTP max-K=`5`
- DynamicSD schedule: `1:4:5,5:16:3,17:64:2,65:128:1,129:512:0`
- exact generation: `min_tokens == max_tokens == OSL`, `ignore_eos=true`
- prefix cache off, chunked prefill on, `max_num_seqs=512`
- `max_num_batched_tokens=32768`, temperature/top-p=`1.0/1.0`

The full cross product contains 448 runs. The 224-run `mrv1` slice with
`PIECEWISE` CUDA graphs is the correctness gate. `mrv2` with
`FULL_AND_PIECEWISE` is a separate canary
because the v0.28 MTP/DynamicSD path still needs runtime validation. DynamicSD
is restricted to DP1.

DynamicSD selects K from the actively scheduled batch at each engine iteration,
not from the total number of prompts passed to the offline benchmark. Offered
concurrency can therefore move through several K ranges as requests are
admitted and drained. Results record the requested-batch schedule lookup only
as provenance; they do not label it as the effective runtime K.

## Reproducible runtime

The container staging job pins the official ARM64 image digest
`sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4`
and vLLM release commit `2cf0a69`. It writes an immutable SQSH plus JSON
provenance and updates a stable symlink only after validation.

Ultra's multi-node executor uses Ray 2.48.0, matching the exact vLLM 0.28
CUDA CI pin. Its Python 3.12 Linux ARM64 dependency closure is constrained by
the vLLM `2cf0a69` CUDA test lock, fully hash-pinned in
`ray248-aarch64.lock`, and staged as an immutable compressed sidecar. Each
Ultra node expands the verified bundle into `/raid/scratch`; Ray runtime state
also stays in a job-specific node-local directory.

Super BF16 is pinned to revision
`d51eab0d1f979ebc26b546e634a04f450d99158e`; staging validates `config.json`
and all 50 weight shards. Ultra BF16 is pinned to the existing revision
`624ba927cfbef0427354998700de3d51173c8c04`.

All source and scripts run from `/home`. Container/checkpoint/result artifacts
are durable on `/lustre`; compilation and Hugging Face caches use
`/raid/scratch`.

## Gate order

1. Stage and validate the vLLM container.
2. Stage and validate Super BF16, dependent on the container job.
3. Run baseline, static K5, and DynamicMTP at BS 1/32/128.
4. Run DynamicMTP at BS512 and verify the observed active-batch/K behavior;
   use an explicit all-K0 schedule if a zero-draft control is required.
5. Expand to the 448-run matrix only after MRV1 and MRV2 gates pass.

Render the non-submitting gate plan with:

```bash
python3 experiments/vllm_028_nemotron_bf16_matrix/submit_smoke.py --dry-run
```

After `sbatch --test-only` preflight, submit the dependency-gated container and
checkpoint staging jobs with `submit_smoke.py --submit-staging`. Internally,
`build_stage_submission_plan` resolves the committed Ray lock to an absolute
path and exports it to SLURM, so invocation from the repository root is safe.

The renderer intentionally does not bulk-submit jobs. Each completed result
must report exact output-token completion and immutable runtime provenance
before it is included in the final comparison table.
