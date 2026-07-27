# Qwen3-30B-A3B Run Provenance

- Container:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260726_2503237.sqsh`
- Container SHA256:
  `6ca7d60e10fa584f476e439419c97bc9185c321f727af8962efce1c633ae86fe`
- Container source: `nvcr.io/nvidian/nemo-rl:nightly`
- Container source commit:
  `f366f041a9d23f3bfee2151ae5c76103d5785e4c`
- MXFP8 arm commit:
  `4289bf61f01165e01b22ba85440e86829dea0ff1`
- BF16 arm commit:
  `90a35f8bea8842fd0a65ac14ef09056d5b605579`
- vLLM: 0.20.0
- Model: `Qwen/Qwen3-30B-A3B`

The BF16 commit is the MXFP8 commit plus only the matched BF16 experiment
launcher, README, and test. Runtime NeMo-RL code is unchanged.

Submodules:

| Path | Commit |
| --- | --- |
| `3rdparty/Automodel-workspace/Automodel` | `24b47e856263d313b942f0ed666c63fff83306b4` |
| `3rdparty/Gym-workspace/Gym` | `d67ad6611cfe21dbaeb301c59e59df32ce22ec50` |
| `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge` | `a056408d29ad1070fb926512991075998c9e023f` |
| `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM` | `cf2f07d7b1315c96c05554c670c43207c6783e5e` |

Per-arm recipes and optimization switches are in `run_matrix.csv`.
