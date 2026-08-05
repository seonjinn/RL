# Handoff

## Resume From Here

The credential issue is fixed. Jobs `496459` and `496460` then exposed a vLLM
0.25 runtime assertion because the modular MXFP8 MoE kernel was uninitialized.
The two existing smoke-tested fixes are cherry-picked as `bf3005a7a` and
`29ac96193`; push, pull, and rerun with warm driver and worker environments.

## Next Actions

- Push the two vLLM runtime fixes and updated session notes.
- Pull on GCP-NRT, resubmit `submit_pair.sh`, and monitor through step 2.

## Watch Outs

- Do not attribute the historical trainer-prequant result to current PR 3477.
- Keep PR 3478's transfer/update effect separate from total-refit and E2E claims.
