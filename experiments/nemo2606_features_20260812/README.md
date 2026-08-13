# NeMo 26.06 policy performance features

This experiment evaluates CuTeDSL fused GroupedGEMM, full-iteration CUDA Graph,
and expert-parallel A2A overlap in NeMo-RL PolicyTraining and current-policy
Logprob. The source of truth for status and evidence is
[`report/index.html`](report/index.html).

Performance claims require a clean exact source, an immutable container, a
dependency-matched baseline, finite optimizer updates, identical workload
settings, and replicated alternating arms. Functional smoke runs are not used
as speedup evidence.
