# Nemotron3 Super HybridEP GB200 validation

This experiment validates the existing 32-node, 4-GPU Nemotron3 Super
performance recipe with only HybridEP enabled. The source baseline is an exact
NeMo-RL `main` commit and the runtime uses an immutable import of the NeMo-RL
nightly image.

## Validation matrix

| Field | Value |
|---|---|
| Base recipe | `grpo-nemotron3-super-120BA12B-32n4g.yaml` |
| Model topology | TP2, EP16, PP1 |
| Hardware | GB200, 4 GPUs per node |
| Nodes | 32 |
| Steps | 20 |
| Dispatcher | Flex + HybridEP |
| Sequence packing alignment | Enabled |
| NVLink domain | 72 GPUs, MNNVL enabled |

The submitter writes a private runtime manifest next to the job logs. Cluster
paths, account names, and scheduler identifiers are intentionally excluded from
this repository.

`submit_ptyche.sh` defaults to the Ptyche `36x2` constraint. Set
`CONSTRAINT=` when reusing it on an NVL72 cluster that does not expose that
feature.
