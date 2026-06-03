# SpecDec Decode-Heavy Investigation — Handoff (2026-06-02, PT)

Self-contained handoff for continuing this work in a fresh agent (Codex). No prior conversation
context is assumed. Everything needed to reproduce, harvest, and extend is below.

---

## 1. The question being investigated

Speculative decoding (Eagle3 drafters) has shown little/no end-to-end speedup for NeMo-RL GRPO
rollout on Qwen3 (8B / 30B-A3B / 235B). The working hypothesis from the user was:

1. Generation is a small fraction of the E2E RL step, so SpecDec's E2E effect is Amdahl-limited.
2. **Therefore**, making generation dominate (long decode / "decode-heavy") should expose a real speedup.

This session tested that hypothesis with **vLLM standalone `LLM.generate` decode-heavy benchmarks**
(pure generation, no RL loop) and harvested existing NeMo-RL data.

**Verdict: hypothesis is half right, half wrong.**
- Part 1 (gen fraction small) — CONFIRMED.
- Part 2 (decode-heavy → speedup) — REFUTED. Decode-heavy makes SpecDec a *slowdown* at K=1.

---

## 2. Headline findings (all measured this session)

### 2a. Standalone decode-heavy, greedy, ISL=1024 OSL=10000, OpenMath prompts

235B target = `Qwen/Qwen3-235B-A22B-Thinking-2507`, TP=4, draft_tp=1, CUDA graph on, custom-all-reduce off.

| config | bs1 | bs2 | bs4 |
|--------|-----|-----|-----|
| baseline tok/s/gpu | 23.05 | 42.41 | 75.35 |
| in-house 500K K=1 | 0.486x | 0.578x | 0.606x |
| in-house 500K K=3 | 0.553x | 0.681x | 0.564x |
| public HF K=1 | 0.652x | 0.550x | 0.554x |
| public HF K=3 | 1.013x | 0.575x | 0.591x |

235B K=3 acceptance reruns (jobs 3122231 in-house / 3122235 public, with acceptance logging; slight
throughput variance vs the pre-fix rows above is bs1 single-prompt noise):

| config | bs | speedup | acceptance | mean_accept_len | per-position acc | overhead_ratio |
|--------|----|---------|-----------|-----------------|------------------|----------------|
| in-house K=3 | 1 | 0.678x | 0.340 | 2.02 | [0.593, 0.313, 0.114] | 2.98x |
| in-house K=3 | 2 | 0.590x | 0.313 | 1.94 | [0.556, 0.268, 0.113] | 3.28x |
| in-house K=3 | 4 | 0.557x | 0.353 | 2.06 | [0.578, 0.325, 0.158] | 3.70x |
| public K=3 | 1 | 0.981x | 0.606 | 2.82 | [0.867, 0.626, 0.326] | 2.87x |
| public K=3 | 2 | 0.575x | 0.426 | 2.28 | [0.670, 0.396, 0.213] | 3.96x |
| public K=3 | 4 | 0.572x | 0.363 | 2.09 | [0.615, 0.309, 0.164] | 3.65x |

**235B is BOTH overhead-bound AND acceptance-limited (unlike 8B which was pure overhead).** 235B
acceptance is low (~31-35% in-house, ~36-61% public) and per-position collapses steeply (pos2 only
11-33%), so mean_accept_len is only ~2.0-2.8 — below the ~2.9 break-even set by the overhead ratio.
**public > in-house on acceptance** (bs1 pos0 0.87 vs 0.59), so the in-house mlen8193 drafter quality
IS a real secondary factor (the earlier K=1-throughput-only read that said "public ≈ in-house, not a
drafter problem" was masked by overhead). Model check: public K=1 bs1 = pos0 acc 0.87 → mean_len 1.87
/ overhead 2.87 = 0.65x = measured public K=1 bs1 0.652x ✓.

8B target = `Qwen/Qwen3-8B`, drafter `RedHatAI/Qwen3-8B-speculator.eagle3`, TP=1:

| config | bs1 | bs2 | bs4 | acceptance | mean_accept_len |
|--------|-----|-----|-----|-----------|------|
| baseline tok/s/gpu | 207.98 | 405.58 | 763.61 | — | — |
| K=1 | 0.701x | 0.717x | 0.762x | **0.999** | 2.00 |
| K=3 | 1.343x | 1.351x | 1.455x | **0.998** | 3.99 |

### 2b. The mechanism — it is OVERHEAD-bound, NOT acceptance-bound

8B acceptance is **99.9%** (near-perfect at greedy) yet K=1 is still **0.70x (slower)**.

`speedup = tokens_emitted_per_step / per_step_cost_ratio`:
- 8B K=1: 2.00 tokens / **2.85x** baseline-decode cost = 0.70x
- 8B K=3: 3.99 tokens / **2.97x** baseline-decode cost = 1.34x

The SpecDec per-step machinery (drafter forward + verify + rejection sampling + lost CUDA-graph
efficiency + scheduling) costs ~2.9x a baseline decode step, **roughly constant in K**. Break-even is
~mean_len 2.9 (≈K=2). So K=1 (2 tokens) always loses; K=3 (≈4 tokens) wins by amortizing the fixed
overhead. The win is NOT from better acceptance.

**235B does not recover at K=3** (still 0.55-0.68x; public K=3 bs1 only 0.98x). Two compounding causes
(see acceptance table in §2a): (1) per-step overhead ~2.9-4x (draft_tp=1 single-GPU drafter vs TP=4
target), AND (2) low acceptance with steep per-position collapse → mean_accept_len only ~2.0-2.8,
below the ~2.9 break-even. The public drafter has higher acceptance than in-house, so in-house drafter
quality (mlen8193, possibly overshot by OSL=10000) is a real secondary factor. 235B needs BOTH levers:
cut overhead (draft_tp) and raise effective accepted length (better drafter, or stop drafting the dead
3rd position). 8B, by contrast, was pure overhead (99.9% acceptance) and K=3 alone fixed it.

Caveat: the 99.9% is inflated by `ignore_eos=True` + `min_tokens=OSL` forcing generation past natural
EOS into degenerate repetition (trivially draftable). Real RL (temperature=1.0) acceptance is ~57%
(30B NeMo-RL), so the overhead conclusion is conservative.

### 2c. NeMo-RL generation fraction (from existing 30B data)

`experiments/eagle3_qwen3_235b/qwen30ba3b_500k_live_summary.json`, Qwen3-30B-A3B GRPO, in-house 500K:

| K | gen_time | E2E | gen frac | gen speedup | E2E speedup | acceptance |
|---|----------|-----|----------|-------------|-------------|------------|
| baseline (3056050) | 96.33 | 236.43 | **40.7%** | 1.0 | 1.0 | — |
| K=1 (3058167) | 71.74 | 215.31 | 33.3% | 1.344x | **1.099x** | 57.5% |
| K=2 (3058168) | 80.36 | 222.93 | 36.0% | 1.200x | 1.062x | 42.1% |
| K=3 (3058169) | 81.91 | 221.46 | 37.0% | 1.177x | 1.068x | 31.9% |

- Generation is only **40.7%** of E2E (and **15.5%** in the large-batch alltoall full-shape run
  3078629/3078630 → E2E 0.997x). Amdahl ceiling ≤ 1.68x even with infinite gen speedup.
- **In NeMo-RL (temperature=1.0), K=3 is WORSE than K=1** because acceptance collapses with depth
  (57→42→32%). Opposite of standalone greedy 8B where K=3 wins. The greedy K=3 standalone win does
  NOT transfer to temp=1.0 RL.

### 2d. Config clarification (235B NeMo-RL)

There is no literal `grpo-qwen3-235b-32n4g.yaml` in the repo. The 235B NeMo-RL SpecDec runs use
**`grpo-qwen3-235b-16n8g.yaml`** (TP=2, PP=8, EP=16) via the GB200 launch scripts
(`Qwen235B_GB200_Main_PublicHF_Eagle3*.sh`), with effective layout **32 nodes × 4 GPU = 128 GPU on
GB200** (= the `official_32n4g_async` resource profile — that is where "32n4g" comes from). The math
throughput recipe uses `grpo-qwen3-235b-32n8g-async-1off.yaml`. Both have
`max_total_sequence_length=8192`, which ≈ the in-house drafter training length (mlen8193). So the
decode-heavy OSL=10000 (total 11024) **overshoots** the drafter's trained context — a confound that is
absent in the real RL config.

---

## 3. Infrastructure / paths (all on cluster `oci-hsg-cs-001-vscode-02`)

| Asset | Path |
|-------|------|
| Local repo (this checkout) | `/Users/sna/Nemo-RL_Qwen3_Roadmap` (git `specdec_rl.git`) |
| Benchmark py (source of truth, has acceptance fix) | `experiments/eagle3_qwen3_235b/standalone_vllm_specdec_breakdown.py` |
| 235B standalone submit (run LOCALLY; SSHes + scp's local py) | `experiments/eagle3_qwen3_235b/submit_vllm_standalone_specdec_breakdown.sh` |
| 8B standalone submit (run ON cluster; uses cluster py) | `experiments/eagle3_qwen3_8b/submit_qwen3_8b_vllm_standalone.sh` |
| vllm-benchmark repo (cluster, separate git `dastokes/vllm-benchmark`) | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark` |
| Cluster benchmark py (acceptance fix DEPLOYED 2026-06-02; backup `.bak_20260602_preaccept`) | `.../vllm-benchmark/standalone_vllm_specdec_breakdown.py` |
| Run output dirs | `.../vllm-benchmark/vllm-runs/<TAG>/breakdown.json` |
| NeMo-RL repo (SpecDec) | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/SpecDec-RL` |
| In-house 235B drafter | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers94_mlen8193/0` |
| Public 235B drafter (Thinking) | `nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3` (HF cached) |
| Public 8B drafter | `RedHatAI/Qwen3-8B-speculator.eagle3` (HF cached) |
| OpenMath prompts JSONL | `/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/openmath_reasoning_cot_conversations_50k.jsonl` |
| 235B container (vllm 0.17.0) | `/lustre/fsw/portfolios/coreai/users/guyueh/rl_projects/vllm/vllm-runs/vllm-hsg-nightly-nsys.sqsh` |
| 8B container (vllm 0.20.2) | `/lustre/fsw/portfolios/coreai/users/sna/containers/vllm-hsg-ultra-rl-v0.20.2-nemo-speed-pr24.sqsh` |
| HF_HOME | `/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home` |
| SLURM account / partition | `coreai_dlalgo_nemorl` / `batch` |

### The acceptance-logging fix (IMPORTANT)
- Root cause: completed standalone runs had empty `spec_decode_metrics` because `build_llm()` did not
  set `disable_log_stats`, so vLLM offline defaulted to stats OFF and `llm.get_metrics()` returned no
  spec counters.
- Fix applied 2026-06-02 in `build_llm()` of the local benchmark py:
  added `"disable_log_stats": False,` to the `LLM(**kwargs)` dict. This enables vLLM v1 stat collection
  so `vllm:spec_decode_num_accepted_tokens` etc. are exposed.
- Deployment: the **235B submit script auto-scp's the local py** at submit time (so new 235B runs get
  the fix automatically). The **8B submit script uses the cluster copy** — the fixed py was manually
  scp'd to `.../vllm-benchmark/standalone_vllm_specdec_breakdown.py` (verify with
  `grep -c disable_log_stats <path>`). Jobs submitted BEFORE 17:25 PT 2026-06-02 do NOT have acceptance.

---

## 4. Every job launched this session

All COMPLETED unless noted. Result dir = `.../vllm-benchmark/vllm-runs/<TAG>/breakdown.json`.

| Job | What | Acceptance logged? | Status |
|-----|------|--------------------|--------|
| 3121332 | 8B baseline decode-heavy | n/a | DONE |
| 3121333 | 8B K=1 decode-heavy (no acc) | no (pre-fix) | DONE |
| 3121334 | 8B K=3 decode-heavy (no acc) | no (pre-fix) | DONE |
| 3121336 | 235B public K=1 decode-heavy | no (pre-fix) | DONE |
| 3121564 | 235B in-house K=3 decode-heavy | no (pre-fix) | DONE |
| 3121566 | 235B public K=3 decode-heavy | no (pre-fix) | DONE |
| 3121598 | 8B K=1 decode-heavy (acc rerun) | **yes** | DONE |
| 3121599 | 8B K=3 decode-heavy (acc rerun) | **yes** | DONE |
| 3122231 | 235B in-house K=3 decode-heavy (acc rerun) | yes | DONE (results in §2a) |
| 3122235 | 235B public K=3 decode-heavy (acc rerun) | yes | DONE (results in §2a) |

Pre-existing (earlier session, baseline reuse): 235B in-house K=1 = 3120704, baseline = 3120705.
3120648 (235B in-house K=1 OSL=20000) was still running earlier — check if relevant.

TAG naming for the acc reruns:
- `qwen235b_thinking2507_vllm_inhouse500k_eagle3_k3_acc_openmath_decodeheavy_1024x10000_bs1-4_mbt16384_cuda_graph_no_custom_ar_20260602`
- `qwen235b_thinking2507_vllm_publichf_eagle3_k3_acc_openmath_decodeheavy_1024x10000_bs1-4_mbt16384_cuda_graph_no_custom_ar_20260602`

---

## 5. Harvest command (3122231/3122235 already harvested — results in §2a)

These two finished and their acceptance results are recorded in §2a. Re-run this to re-harvest any
breakdown.json (also the template for harvesting future runs):

```bash
ssh oci-hsg-cs-001-vscode-02 "python3 - <<'PY'
import json,os
base='/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/vllm-benchmark/vllm-runs'
def load(t):
    p=os.path.join(base,t,'breakdown.json')
    return {r['bs']:r for r in json.load(open(p))['results']} if os.path.exists(p) else None
bl=load('qwen235b_thinking2507_vllm_baseline_openmath_decodeheavy_1024x10000_bs1-4_mbt16384_cuda_graph_no_custom_ar_20260602')
for name,t in {
 'in-house K=3':'qwen235b_thinking2507_vllm_inhouse500k_eagle3_k3_acc_openmath_decodeheavy_1024x10000_bs1-4_mbt16384_cuda_graph_no_custom_ar_20260602',
 'public   K=3':'qwen235b_thinking2507_vllm_publichf_eagle3_k3_acc_openmath_decodeheavy_1024x10000_bs1-4_mbt16384_cuda_graph_no_custom_ar_20260602'}.items():
    d=load(t)
    if not d: print(name,'NO JSON'); continue
    for bs in sorted(d):
        tg=d[bs]['output_tok_s_per_gpu']; sp=tg/bl[bs]['output_tok_s_per_gpu'] if bl else None
        sd=d[bs].get('spec_decode_metrics') or {}
        acc=' acc=%.3f mean_len=%.2f per_pos=%s'%(sd['acceptance_rate'],sd.get('mean_acceptance_length',0),sd.get('acceptance_rate_per_pos')) if sd.get('acceptance_rate') is not None else ' (acc NA)'
        print('%s bs=%s speedup=%.3fx%s'%(name,bs,sp,acc))
PY"
```

Interpretation: if 235B acc is ~99% like 8B but still 0.55x → pure overhead (confirms draft_tp=1
hypothesis). If 235B acc is low (≪99%) → overhead + acceptance collapse (mlen8193 overshoot / thinking
drift). Either way the fix is draft_tp.

---

## 6. THE decisive next experiment — draft_tp sweep (not yet launched)

Hypothesis: 235B's slowdown is dominated by the **single-GPU drafter** (`draft_tensor_parallel_size=1`
while target TP=4). Sharding the drafter should cut the overhead ratio and let K=3 win.

The submit script hardcodes `draft_tensor_parallel_size: 1` in the speculative_config JSON
(`submit_vllm_standalone_specdec_breakdown.sh`, the `SPECULATIVE_CONFIG` python heredoc ~lines 86-95).
To sweep, add a `DRAFT_TP` env override:

```bash
# in submit_vllm_standalone_specdec_breakdown.sh, change the heredoc line
#   "draft_tensor_parallel_size": 1,
# to
#   "draft_tensor_parallel_size": int("${DRAFT_TP:-1}"),
```

Then launch (235B in-house K=3, sweep DRAFT_TP=1,2,4, OSL=10000, same OpenMath setup):

```bash
cd /Users/sna/Nemo-RL_Qwen3_Roadmap
for DTP in 2 4; do
  MODEL="Qwen/Qwen3-235B-A22B-Thinking-2507" \
  DRAFT_MODEL="/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/speculators/eagle3_qwen3_235b_mixed_math_nonopenmath_500k_parallel/checkpoints_train_500k_layers94_mlen8193/0" \
  ENABLE_SPECDEC=true NUM_SPECULATIVE_TOKENS=3 DRAFT_TP=$DTP \
  TP=4 PP=1 GPUS=4 ISL=1024 OSL=10000 BATCH_SIZES="1 2 4" \
  MAX_NUM_BATCHED_TOKENS=16384 ENFORCE_EAGER=false DISABLE_VLLM_PROFILER=true DISABLE_CUSTOM_ALL_REDUCE=true \
  PROMPT_JSONL=/lustre/fsw/portfolios/coreai/users/sna/qwen3_235b_eagle3/data/openmath_reasoning_cot_conversations_50k.jsonl PROMPT_OFFSET=0 \
  TAG=qwen235b_thinking2507_inhouse500k_k3_drafttp${DTP}_openmath_decodeheavy_1024x10000_bs1-4_20260602 \
  JOB_FILE=/Users/sna/Nemo-RL_Qwen3_Roadmap/latest_vllm_qwen235b_inhouse500k_k3_drafttp${DTP}_jobs.txt \
  bash experiments/eagle3_qwen3_235b/submit_vllm_standalone_specdec_breakdown.sh
done
```

Success criterion: if `draft_tp=4` moves 235B K=3 above ~1.0x, the single-GPU-drafter overhead is
confirmed as the binding constraint and draft_tp becomes the actionable fix for 235B SpecDec.

Caveat from §2a acceptance data: 235B mean_accept_len is only ~2.0-2.8 (acceptance ~35%, per-position
collapses), so cutting overhead via draft_tp may only reach break-even, not a large win — 235B also
needs higher effective accepted length. Pair the draft_tp sweep with: (a) public drafter (better
acceptance than in-house), (b) try K=2 (the 3rd draft position is accepted only 11-33% of the time, so
it mostly wastes work), and (c) an OSL≤8192 within-drafter-range run (§7 item 1) since OSL=10000
overshoots the mlen8193 in-house drafter.

---

## 7. Other recommended experiments (lower priority)

1. **OSL within drafter range** — repeat 235B in-house decode-heavy at OSL≈6000-7000 (total < 8192) to
   separate "context overflow beyond mlen8193" from "regime/overhead". If speedup recovers at OSL=7000,
   the in-house drafter's training length is a (secondary) cause.
2. **temperature=1.0 (RL-matched) variant** — the benchmark hardcodes `temperature=0.0` (greedy) in
   `SamplingParams` (~line 435). RL rollout uses temperature=1.0, top_p=1.0. Add a `--temperature`/
   `--top-p` arg and re-measure; expect lower acceptance and worse SpecDec (greedy is best-case).
   This is the number that actually predicts RL behavior.
3. **NeMo-RL E2E with the chosen config** — if draft_tp helps standalone, validate in NeMo-RL
   (`grpo-qwen3-235b-16n8g.yaml`) with load-aware gating (not always-on). The doc's central
   recommendation is gating, not always-on SpecDec.

---

## 8. Document to update (the user maintains this)

`docs/specdec_background_and_observations_google_doc.md` — its decode-heavy section is OUTDATED:
- It says the synthetic decode-heavy negative result was due to "dummy token IDs" and expects OpenMath
  to fix it. The OpenMath decode-heavy result (this session) is WORSE (0.486x), so that explanation is
  wrong. Replace with: "decode-heavy K=1 is a slowdown for both 8B and 235B; the binding constraint is
  per-step SpecDec overhead (8B 99.9% acceptance still gives 0.70x at K=1); K=3 amortizes overhead and
  wins on 8B but not 235B (draft_tp=1 single-GPU drafter)."
- The projection model `1 + 0.64*acceptance` is short-decode-only; add a regime caveat.
- Add per-position acceptance and the overhead-ratio framing (`speedup = mean_accept_len / overhead_ratio`).

---

## 9. How to re-derive the speedup numbers

`output_tok_s_per_gpu = bs * OSL / latency_s / total_gpus` (in the benchmark). Speedup = specdec
tok/s/gpu ÷ matched-baseline tok/s/gpu at the same bs. The baseline (no drafter) is shape-independent of
the drafter, so a single baseline run is reused across K and drafter variants.

`overhead_ratio = mean_acceptance_length / measured_speedup` = per-step SpecDec cost in units of a
baseline decode step (only computable when acceptance is logged).
