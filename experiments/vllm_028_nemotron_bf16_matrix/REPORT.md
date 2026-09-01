# vLLM 0.28 Nemotron-3 BF16 DynamicMTP report

## Canary configuration

| Field | Super | Ultra |
| --- | --- | --- |
| Checkpoint revision | `d51eab0d1f979ebc26b546e634a04f450d99158e` | `624ba927cfbef0427354998700de3d51173c8c04` |
| Weight / KV precision | BF16 / FP8 | BF16 / FP8 |
| Parallelism | TP2, DP1, 1 node | TP8, DP1, EP, Ray, 2 nodes |
| Workload | ISL 10000, OSL 1000, BS 1 | ISL 10000, OSL 1000, BS 1 |

The runtime was vLLM 0.28.0 at commit `2cf0a69`, using the official ARM64
image digest
`sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4`.
Both runners used exact OSL, ignored EOS, disabled prefix caching, enabled
chunked prefill, and set `max_num_seqs=512` and
`max_num_batched_tokens=32768`.

DynamicMTP used max-K 5 and the schedule
`1:4:5,5:16:3,17:64:2,65:128:1,129:512:0`.

Both checkpoint configs declare `num_nextn_predict_layers=1`. Super uses
`mtp_hybrid_override_pattern="*E"`; Ultra declares an attention-plus-MoE MTP
block. The architectural MTP depth is therefore one for both checkpoints.
Static K2-K5 does not address two to five independent MTP heads. vLLM 0.28
reuses the same one-layer MTP block for multiple forwards and emits an explicit
warning that acceptance can fall when `num_speculative_tokens > 1`.

The vLLM validator requires a positive global `num_speculative_tokens`. For an
MTP model, a value above the checkpoint's native `n_predict` must be divisible
by `n_predict`. Because these checkpoints have `n_predict=1`, vLLM accepts any
positive static K in principle. K1-K5 is the bounded practical sweep selected
for this experiment, not the complete unbounded set of accepted integers.

DynamicSD is a native vLLM 0.28 feature, not a harness-only patch. The exact
configuration is:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 5,
  "num_speculative_tokens_per_batch_size": [
    [1, 4, 5],
    [5, 16, 3],
    [17, 64, 2],
    [65, 128, 1],
    [129, 512, 0]
  ]
}
```

At every scheduler step, vLLM indexes the schedule with the number of actively
scheduled requests. It clamps each scheduled K to the global maximum K=5.
DynamicSD is disabled for DP greater than one because ranks can select
different K values and deadlock collectives. Under MRV1, vLLM downgrades full
CUDA Graph mode to `PIECEWISE` for reliability; this experiment requests
`PIECEWISE` explicitly and verifies the actual capture in every job log.

Relevant upstream changes are the merged DynamicSD implementation
[PR #32374](https://github.com/vllm-project/vllm/pull/32374) and the merged MRV2
full-CUDA-Graph infrastructure [PR #45953](https://github.com/vllm-project/vllm/pull/45953).
The v0.28.0 source used here already contains both. A later open MRV2 issue
[issue #51510](https://github.com/vllm-project/vllm/issues/51510) reports that
MRV2 can still ignore the scheduler's dynamic K on the drafter side; the
associated fix [PR #51575](https://github.com/vllm-project/vllm/pull/51575) was
not part of commit `2cf0a69`. This is another reason MRV1 is the validated path
for the published matrix.

## CUDA Graph verification

The completed BS512 baseline and DynamicMTP logs verify real CUDA Graph
capture rather than eager execution. All four runs report
`enforce_eager=False`, `CUDAGraphMode.PIECEWISE`, one graph warmup, 83 capture
sizes through 1024, and successful `83/83` mixed prefill/decode capture.

| Model | Method | Job | Capture | CUDA Graph pool memory |
| --- | --- | ---: | --- | ---: |
| Super | Baseline | `2816828` | `83/83` | 1.44 GiB |
| Super | DynamicMTP | `2816642` | `83/83` | 1.49 GiB |
| Ultra | Baseline | `2816829` | `83/83` | 2.39 GiB |
| Ultra | DynamicMTP | `2816643` | `83/83` | 2.45 GiB |

Super TP2 runs systematically emitted non-fatal `CUDACachingAllocator`
allocation warnings during pre-capture memory profiling. Every audited Super
baseline, DynamicMTP, and Static K5 job then completed `83/83` graph capture,
generation, exact token validation, and canonical publication. Their throughput
is retained with this caveat; representative repeats are required before the
matrix is marked final. The corresponding Ultra TP8 jobs did not emit these
allocator warnings.

## Patched MRV2 Dynamic-K canary

The vLLM 0.28.0 base was patched with the pinned #49652, #51575, and #52548
changes plus the local Mamba reduced-query-length guard. Optional #53426 was
excluded. The immutable patched image SHA256 is
`5ae5c3e3d630d95e1129b71384fe9c5c437a77288492ada30da94f93b8582066`;
the patchset manifest ID is `238e2ffcc14d`.

The controlled schedule
`1:1:5,2:2:3,3:4:2,5:8:1,9:512:0` exercised K5, K3, K2, K1, and K0 in one
engine per model at ISL/OSL 1000/128. Clean reruns Super `2820849` and Ultra
`2820713` both completed with SLURM exit code `0:0`, exact output-token
generation, and the same complete CUDA Graph set: target PIECEWISE `83/83`,
target FULL `375/375`, drafter prefill PIECEWISE `83/83`, drafter prefill FULL
`375/375`, and drafter decode FULL `51/51`.

| Model | BS | Selected K | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Exact tokens | Job |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Super | 1 | 5 | 123.91 | N/A | 26.18% | 2.3091 | 128/128 | `2820849` |
| Super | 2 | 3 | 188.51 | N/A | 55.10% | 2.6531 | 256/256 | `2820849` |
| Super | 4 | 2 | 53.63 | N/A | 44.78% | 1.8955 | 512/512 | `2820849` |
| Super | 8 | 1 | 577.81 | N/A | 74.62% | 1.7564 | 1024/1024 | `2820849` |
| Super | 16 | 0 | 868.80 | N/A | N/A | N/A | 2048/2048 | `2820849` |
| Ultra | 1 | 5 | 30.27 | N/A | 49.73% | 3.4865 | 128/128 | `2820713` |
| Ultra | 2 | 3 | 46.75 | N/A | 63.57% | 2.9213 | 256/256 | `2820713` |
| Ultra | 4 | 2 | 11.86 | N/A | 66.89% | 2.3594 | 512/512 | `2820713` |
| Ultra | 8 | 1 | 93.32 | N/A | 81.09% | 1.8253 | 1024/1024 | `2820713` |
| Ultra | 16 | 0 | 118.95 | N/A | N/A | N/A | 2048/2048 | `2820713` |

These are single-repeat correctness canaries, not matched performance rows.
No MRV2 K0 baseline was run, so no speedup is reported. BS4 includes a large
first-use JIT latency spike. The SpecDec Prometheus counters update
asynchronously: positive-K validation permits at most one offered batch at the
max-K counter width and records the skew per row; K0 acceptance is suppressed
because a small prior-step counter residual was observed. Exact-token
validation remains strict, and substantive max-K work hidden under reduced K
is still rejected. Publication sweeps must use one fresh engine per workload
cell so this cross-row counter bleed cannot affect matched performance rows.

The canonical canary CSV and standalone HTML page are generated by
`build_mrv2_patch_canary_report.py` from the committed result and CUDA Graph
evidence artifacts.

## Patched MRV2 full BF16 matrix

The 64-cell matched matrix completed on 2026-08-28: two models, two ISL/OSL
shapes, eight offered concurrencies, and baseline versus DynamicMTP. All 64
jobs exited `0:0`, generated exactly `concurrency * OSL` measured output
tokens, and passed the patched-image, checkpoint, topology, and CUDA Graph
provenance gates. Baseline rows captured target PIECEWISE and FULL graphs;
DynamicMTP rows additionally captured drafter prefill PIECEWISE/FULL and
drafter decode FULL graphs.

The results are preliminary single-repeat measurements. DynamicMTP wins all 24
matched cells at concurrency 1–32 and loses all eight cells at concurrency
128/512. The best result is Ultra 10K/1K at C=1 (`2.932x`); the weakest is
Super 1K/10K at C=512 (`0.382x`). High acceptance at high concurrency does not
imply good throughput: actual mean draft width has already fallen close to K1,
while drafter and synchronization overhead remain.

### Super BF16 · 1000/10000

| C | Baseline tok/s/GPU | Dynamic tok/s/GPU | Speedup | Acceptance | Mean accepted length | Mean draft width |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 101.425 | 241.440 | 2.380x | 65.32% | 4.266 | 5.000 |
| 2 | 170.069 | 288.838 | 1.698x | 54.78% | 3.739 | 5.000 |
| 4 | 338.588 | 520.414 | 1.537x | 61.11% | 4.055 | 5.000 |
| 8 | 603.742 | 1176.543 | 1.949x | 88.56% | 3.715 | 3.066 |
| 16 | 1184.445 | 2157.104 | 1.821x | 93.29% | 3.833 | 3.037 |
| 32 | 2185.212 | 3462.096 | 1.584x | 96.18% | 2.925 | 2.002 |
| 128 | 4530.499 | 2760.820 | 0.609x | 92.77% | 2.278 | 1.377 |
| 512 | 5675.331 | 2169.613 | 0.382x | 96.09% | 2.027 | 1.068 |

### Super BF16 · 10000/1000

| C | Baseline tok/s/GPU | Dynamic tok/s/GPU | Speedup | Acceptance | Mean accepted length | Mean draft width |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98.651 | 231.274 | 2.344x | 69.64% | 4.482 | 5.000 |
| 2 | 163.207 | 264.454 | 1.620x | 44.72% | 3.236 | 5.000 |
| 4 | 280.891 | 465.132 | 1.656x | 45.40% | 3.270 | 5.000 |
| 8 | 459.657 | 605.938 | 1.318x | 59.38% | 2.840 | 3.098 |
| 16 | 634.527 | 877.504 | 1.383x | 69.78% | 3.126 | 3.047 |
| 32 | 873.264 | 1159.527 | 1.328x | 74.05% | 2.506 | 2.034 |
| 128 | 1568.573 | 969.411 | 0.618x | 79.65% | 2.118 | 1.404 |
| 512 | 1946.143 | 768.179 | 0.395x | 80.67% | 1.891 | 1.104 |

### Ultra BF16 · 1000/10000

| C | Baseline tok/s/GPU | Dynamic tok/s/GPU | Speedup | Acceptance | Mean accepted length | Mean draft width |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.269 | 33.973 | 2.381x | 52.67% | 3.633 | 5.000 |
| 2 | 25.909 | 71.474 | 2.759x | 75.11% | 4.755 | 5.000 |
| 4 | 44.262 | 98.643 | 2.229x | 62.89% | 4.144 | 5.000 |
| 8 | 73.982 | 171.119 | 2.313x | 86.32% | 3.620 | 3.035 |
| 16 | 122.340 | 252.202 | 2.061x | 85.89% | 3.602 | 3.030 |
| 32 | 201.111 | 356.845 | 1.774x | 78.10% | 2.570 | 2.010 |
| 128 | 571.998 | 469.072 | 0.820x | 95.25% | 2.147 | 1.204 |
| 512 | 1068.199 | 496.263 | 0.465x | 92.77% | 1.980 | 1.056 |

### Ultra BF16 · 10000/1000

| C | Baseline tok/s/GPU | Dynamic tok/s/GPU | Speedup | Acceptance | Mean accepted length | Mean draft width |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.951 | 40.906 | 2.932x | 74.43% | 4.722 | 5.000 |
| 2 | 24.701 | 59.818 | 2.422x | 70.20% | 4.510 | 5.000 |
| 4 | 42.202 | 80.430 | 1.906x | 60.34% | 4.017 | 5.000 |
| 8 | 67.562 | 110.453 | 1.635x | 70.80% | 3.205 | 3.114 |
| 16 | 105.588 | 165.954 | 1.572x | 75.46% | 3.277 | 3.018 |
| 32 | 153.806 | 219.905 | 1.430x | 81.93% | 2.669 | 2.037 |
| 128 | 301.470 | 290.141 | 0.962x | 84.26% | 2.050 | 1.246 |
| 512 | 361.630 | 307.028 | 0.849x | 85.90% | 1.868 | 1.010 |

The offered-concurrency schedule lookup is K5/K3/K2/K1/K0, but the runtime
selection key is the active scheduled batch. For example, offered C=512 has a
nominal lookup of K0 but measured mean draft widths of 1.010–1.104, proving
that most admitted decode work occurred in the active K1 range. The next
tuning experiment should set K0 at active batch 65 and above; a separate
#53426 cohort is required before attributing any K0 benefit to sync-forward
skipping.

The four C=512 DynamicMTP reruns use harness commit `f3a842c0` while their
matched baselines use `85cfe1a8`. The intervening commit changes the
active-scheduled-batch result validator, not benchmark execution. The report
builder requires every performance-affecting config field, checkpoint,
container, patched vLLM revision, Ray provenance, and CUDA Graph mode to match;
it permits only this exact C=512 old-baseline/new-Dynamic harness pair and
records `harness_pair_exception=true` on those four Dynamic rows.

## MRV1 correctness canary

| Model | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Generation latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Super | Baseline | 38.9932 | 1.0000x | - | - | 12.8227 s |
| Super | DynamicMTP | 99.9251 | 2.5626x | 61.38% | 4.0691 | 5.0037 s |
| Ultra | Baseline | 7.2625 | 1.0000x | - | - | 17.2116 s |
| Ultra | DynamicMTP | 20.6467 | 2.8429x | 59.52% | 3.9762 | 6.0542 s |

All four jobs completed exactly 1000 output tokens and atomically published a
validated canonical `result.json`. These are single-concurrency canary results,
not the final matrix averages.

| Model | Baseline job | DynamicMTP job |
| --- | ---: | ---: |
| Super | `2815633` | `2815634` |
| Ultra | `2815504` | `2815505` |

The DynamicMTP logs for both BF16 checkpoints report `Detected MTP model` and
share the target embedding and LM-head weights with the built-in drafter. No
external draft checkpoint was used.

## MRV1 1K/10K gate expansion

All 42 jobs below completed exact output-token validation with real PIECEWISE
CUDA Graph capture (`83/83`) and `enforce_eager=false`.

### Super BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 38.9835 | 1.0000x | - | - | `2817792` |
| 1 | Static K5 | 147.7266 | 3.7895x | 96.99% | 5.8497 | `2817798` |
| 1 | DynamicMTP | 115.4559 | 2.9617x | 70.29% | 4.5144 | `2817795` |
| 2 | Baseline | 79.8033 | 1.0000x | - | - | `2817971` |
| 2 | Static K5 | 255.1441 | 3.1972x | 82.91% | 5.1454 | `2817979` |
| 2 | DynamicMTP | 241.1963 | 3.0224x | 79.42% | 4.9709 | `2817975` |
| 4 | Baseline | 156.0840 | 1.0000x | - | - | `2817972` |
| 4 | Static K5 | 508.6326 | 3.2587x | 88.31% | 5.4153 | `2817980` |
| 4 | DynamicMTP | 361.8401 | 2.3182x | 75.97% | 4.7983 | `2817976` |
| 8 | Baseline | 310.9371 | 1.0000x | - | - | `2817973` |
| 8 | Static K5 | 1010.7722 | 3.2507x | 89.87% | 5.4937 | `2817981` |
| 8 | DynamicMTP | 709.2838 | 2.2811x | 93.43% | 3.8352 | `2817977` |
| 16 | Baseline | 625.7017 | 1.0000x | - | - | `2817970` |
| 16 | Static K5 | 1862.3229 | 2.9764x | 91.30% | 5.5649 | `2817978` |
| 16 | DynamicMTP | 1558.4634 | 2.4907x | 97.88% | 3.9418 | `2817974` |
| 32 | Baseline | 1208.1857 | 1.0000x | - | - | `2817794` |
| 32 | Static K5 | 2725.5178 | 2.2559x | 84.62% | 5.2309 | `2817800` |
| 32 | DynamicMTP | 2516.0816 | 2.0825x | 95.81% | 2.9162 | `2817797` |
| 128 | Baseline | 4184.4883 | 1.0000x | - | - | `2817793` |
| 128 | Static K5 | 4837.0664 | 1.1560x | 85.22% | 5.2610 | `2817799` |
| 128 | DynamicMTP | 3486.4307 | 0.8332x | 95.35% | 2.3444 | `2817796` |

### Ultra BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 7.3179 | 1.0000x | - | - | `2817801` |
| 1 | Static K5 | 18.5959 | 2.5411x | 52.29% | 3.6144 | `2817807` |
| 1 | DynamicMTP | 29.0142 | 3.9648x | 93.25% | 5.6625 | `2817804` |
| 2 | Baseline | 14.7509 | 1.0000x | - | - | `2817983` |
| 2 | Static K5 | 44.6310 | 3.0256x | 78.77% | 4.9385 | `2817991` |
| 2 | DynamicMTP | 54.9156 | 3.7229x | 91.62% | 5.5808 | `2817987` |
| 4 | Baseline | 29.1579 | 1.0000x | - | - | `2817984` |
| 4 | Static K5 | 66.9339 | 2.2956x | 61.54% | 4.0768 | `2817992` |
| 4 | DynamicMTP | 59.1264 | 2.0278x | 66.69% | 4.3347 | `2817988` |
| 8 | Baseline | 57.5838 | 1.0000x | - | - | `2817985` |
| 8 | Static K5 | 130.0255 | 2.2580x | 57.29% | 3.8644 | `2817993` |
| 8 | DynamicMTP | 126.3834 | 2.1948x | 77.55% | 3.3672 | `2817989` |
| 16 | Baseline | 111.8123 | 1.0000x | - | - | `2817982` |
| 16 | Static K5 | 229.4710 | 2.0523x | 62.14% | 4.1069 | `2817990` |
| 16 | DynamicMTP | 250.4263 | 2.2397x | 78.06% | 3.3456 | `2817986` |
| 32 | Baseline | 180.8105 | 1.0000x | - | - | `2817803` |
| 32 | Static K5 | 398.0442 | 2.2014x | 67.07% | 4.3534 | `2817809` |
| 32 | DynamicMTP | 403.0922 | 2.2294x | 95.14% | 2.9241 | `2817806` |
| 128 | Baseline | 537.0231 | 1.0000x | - | - | `2817802` |
| 128 | Static K5 | 672.8166 | 1.2529x | 53.87% | 3.6934 | `2817808` |
| 128 | DynamicMTP | 534.4071 | 0.9951x | 92.75% | 2.1928 | `2817805` |

DynamicMTP gives the strongest low-concurrency result for Ultra BS1, but its
benefit disappears at offered BS128 for both models. Static K5 retains positive
BS128 throughput speedup in this long-output workload. As in the 10K/1K runs,
DynamicMTP acceptance must be interpreted with its active-scheduled-batch K
selection rather than the offered batch size alone.

### Complete static-K ladder for BS1/2/4/8/16/32/128/512

Each cell reports `tok/s/GPU (speedup versus matched baseline)`. All 112 rows
published canonical results with exact output-token validation,
`PIECEWISE` CUDA Graph mode, `enforce_eager=false`, and successful `83/83`
graph capture.

#### Super BF16

| BS | K0 baseline | Static K1 | Static K2 | Static K3 | Static K4 | Static K5 | DynamicMTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.98 (1.00x) | 61.18 (1.57x) | 82.26 (2.11x) | 89.66 (2.30x) | 130.72 (3.35x) | **147.73 (3.79x)** | 115.46 (2.96x) |
| 2 | 79.80 (1.00x) | 118.73 (1.49x) | 148.68 (1.86x) | 166.39 (2.08x) | 157.88 (1.98x) | **255.14 (3.20x)** | 241.20 (3.02x) |
| 4 | 156.08 (1.00x) | 235.91 (1.51x) | 288.99 (1.85x) | 340.24 (2.18x) | 443.78 (2.84x) | **508.63 (3.26x)** | 361.84 (2.32x) |
| 8 | 310.94 (1.00x) | 478.32 (1.54x) | 596.53 (1.92x) | 757.88 (2.44x) | 763.73 (2.46x) | **1010.77 (3.25x)** | 709.28 (2.28x) |
| 16 | 625.70 (1.00x) | 928.50 (1.48x) | 1238.04 (1.98x) | 1683.45 (2.69x) | 1282.63 (2.05x) | **1862.32 (2.98x)** | 1558.46 (2.49x) |
| 32 | 1208.19 (1.00x) | 1608.87 (1.33x) | 2244.14 (1.86x) | **3111.31 (2.58x)** | 2719.30 (2.25x) | 2725.52 (2.26x) | 2516.08 (2.08x) |
| 128 | 4184.49 (1.00x) | **5562.85 (1.33x)** | 3902.50 (0.93x) | 4577.69 (1.09x) | 5183.83 (1.24x) | 4837.07 (1.16x) | 3486.43 (0.83x) |
| 512 | 5087.72 (1.00x) | 5303.29 (1.04x) | 4607.40 (0.91x) | **5348.90 (1.05x)** | 4523.52 (0.89x) | 4759.11 (0.94x) | 3453.71 (0.68x) |

#### Ultra BF16

| BS | K0 baseline | Static K1 | Static K2 | Static K3 | Static K4 | Static K5 | DynamicMTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.32 (1.00x) | 12.07 (1.65x) | 17.00 (2.32x) | 17.90 (2.45x) | 24.85 (3.40x) | 18.60 (2.54x) | **29.01 (3.96x)** |
| 2 | 14.75 (1.00x) | 23.95 (1.62x) | 32.71 (2.22x) | 33.44 (2.27x) | 45.74 (3.10x) | 44.63 (3.03x) | **54.92 (3.72x)** |
| 4 | 29.16 (1.00x) | 43.94 (1.51x) | 60.51 (2.08x) | 71.66 (2.46x) | **74.81 (2.57x)** | 66.93 (2.30x) | 59.13 (2.03x) |
| 8 | 57.58 (1.00x) | 88.14 (1.53x) | 111.79 (1.94x) | 140.68 (2.44x) | **154.46 (2.68x)** | 130.03 (2.26x) | 126.38 (2.19x) |
| 16 | 111.81 (1.00x) | 166.63 (1.49x) | 222.69 (1.99x) | 234.74 (2.10x) | **253.59 (2.27x)** | 229.47 (2.05x) | 250.43 (2.24x) |
| 32 | 180.81 (1.00x) | 295.66 (1.64x) | 434.71 (2.40x) | 425.24 (2.35x) | **449.94 (2.49x)** | 398.04 (2.20x) | 403.09 (2.23x) |
| 128 | 537.02 (1.00x) | 848.77 (1.58x) | **1048.44 (1.95x)** | 763.57 (1.42x) | 849.72 (1.58x) | 672.82 (1.25x) | 534.41 (1.00x) |
| 512 | 934.03 (1.00x) | 991.77 (1.06x) | **1083.01 (1.16x)** | 998.59 (1.07x) | 999.98 (1.07x) | 824.71 (0.88x) | 558.25 (0.60x) |

Super prefers K5 through BS16, K3 at BS32, and K1 at BS128. Ultra is more
sensitive to K: DynamicMTP wins BS1/2, static K4 wins BS4/8/16/32, and static
K2 wins BS128. At BS512, static K3 is best for Super and static K2 is best for
Ultra; DynamicMTP falls to 0.68x and 0.60x baseline respectively. Every BS512
job produced exactly 5,120,000 measured output tokens. The new static
K1/K2/K3 job provenance is Super
`2818103`-`2818114`, `2818228`-`2818236` and Ultra `2818115`-`2818126`,
`2818237`-`2818245`. The complete BS512 jobs are Super `2818340`-`2818345`
and Ultra `2818346`-`2818351`.

## MRV1 10K/1K gate expansion

### Super BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 38.9932 | 1.0000x | - | - | `2815633` |
| 1 | Static K5 | 102.7185 | 2.6343x | 61.38% | 4.0691 | `2815986` |
| 1 | DynamicMTP | 99.9251 | 2.5626x | 61.38% | 4.0691 | `2815634` |
| 2 | Baseline | 77.1876 | 1.0000x | - | - | `2817283` |
| 2 | Static K5 | 158.7226 | 2.0563x | 49.02% | 3.4509 | `2817361` |
| 2 | DynamicMTP | 143.3739 | 1.8575x | 39.12% | 2.9558 | `2817288` |
| 4 | Baseline | 154.4845 | 1.0000x | - | - | `2817284` |
| 4 | Static K5 | 283.1409 | 1.8328x | 44.20% | 3.2101 | `2817362` |
| 4 | DynamicMTP | 330.7121 | 2.1407x | 64.89% | 4.2444 | `2817289` |
| 8 | Baseline | 287.1353 | 1.0000x | - | - | `2817286` |
| 8 | Static K5 | 534.1882 | 1.8604x | 55.08% | 3.7539 | `2817363` |
| 8 | DynamicMTP | 470.0615 | 1.6371x | 57.08% | 2.7484 | `2817290` |
| 16 | Baseline | 511.5928 | 1.0000x | - | - | `2817282` |
| 16 | Static K5 | 765.7723 | 1.4968x | 54.23% | 3.7117 | `2817360` |
| 16 | DynamicMTP | 794.3084 | 1.5526x | 56.96% | 2.7127 | `2817287` |
| 32 | Baseline | 824.4557 | 1.0000x | - | - | `2815983` |
| 32 | Static K5 | 1155.8046 | 1.4019x | 51.84% | 3.5919 | `2815988` |
| 32 | DynamicMTP | 1124.0057 | 1.3633x | 76.39% | 2.5622 | `2815985` |
| 128 | Baseline | 1562.5967 | 1.0000x | - | - | `2815982` |
| 128 | Static K5 | 1563.6908 | 1.0007x | 51.93% | 3.5967 | `2815987` |
| 128 | DynamicMTP | 1401.2702 | 0.8968x | 75.56% | 2.5345 | `2815984` |

### Ultra BF16

| BS | Method | tok/s/GPU | Speedup | Acceptance | Mean accepted length | Job |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 7.2625 | 1.0000x | - | - | `2815504` |
| 1 | Static K5 | 16.7987 | 2.3131x | 47.34% | 3.3670 | `2815993` |
| 1 | DynamicMTP | 20.6467 | 2.8429x | 59.52% | 3.9762 | `2815505` |
| 2 | Baseline | 14.3217 | 1.0000x | - | - | `2817292` |
| 2 | Static K5 | 34.6345 | 2.4183x | 53.36% | 3.6679 | `2817365` |
| 2 | DynamicMTP | 40.6689 | 2.8397x | 68.33% | 4.4163 | `2817296` |
| 4 | Baseline | 28.0637 | 1.0000x | - | - | `2817293` |
| 4 | Static K5 | 65.2620 | 2.3255x | 67.61% | 4.3803 | `2817366` |
| 4 | DynamicMTP | 69.4982 | 2.4764x | 66.07% | 4.3036 | `2817297` |
| 8 | Baseline | 52.5442 | 1.0000x | - | - | `2817294` |
| 8 | Static K5 | 101.3524 | 1.9289x | 64.64% | 4.2319 | `2817367` |
| 8 | DynamicMTP | 104.9324 | 1.9970x | 73.54% | 3.2606 | `2817298` |
| 16 | Baseline | 95.6500 | 1.0000x | - | - | `2817291` |
| 16 | Static K5 | 166.6426 | 1.7422x | 61.81% | 4.0905 | `2817364` |
| 16 | DynamicMTP | 163.2649 | 1.7069x | 76.52% | 3.3102 | `2817295` |
| 32 | Baseline | 163.7139 | 1.0000x | - | - | `2815990` |
| 32 | Static K5 | 241.9931 | 1.4781x | 60.09% | 4.0046 | `2815995` |
| 32 | DynamicMTP | 213.9473 | 1.3068x | 81.44% | 2.6588 | `2815992` |
| 128 | Baseline | 334.6108 | 1.0000x | - | - | `2815989` |
| 128 | Static K5 | 324.6044 | 0.9701x | 60.68% | 4.0342 | `2815994` |
| 128 | DynamicMTP | 289.2306 | 0.8644x | 88.15% | 2.1732 | `2815991` |

The requested-batch lookup maps BS2/4 to K5, BS8/16 to K3, BS32 to K2, and
BS128 to K1, but runtime K is selected from the actively scheduled batch. The
acceptance percentages are therefore not directly comparable to static K5
acceptance. DynamicMTP did not recover its overhead at offered BS128; the BS512
gate was used to inspect the actual scheduler behavior.

### Complete static-K ladder for BS1/2/4/8/16/32/128/512

Each cell reports `tok/s/GPU (speedup versus matched baseline)`. All 112 rows
below published canonical results with exact output-token validation,
`PIECEWISE` CUDA Graph mode, `enforce_eager=false`, and successful `83/83`
graph capture.

#### Super BF16

| BS | K0 baseline | Static K1 | Static K2 | Static K3 | Static K4 | Static K5 | DynamicMTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.99 (1.00x) | 59.72 (1.53x) | 65.04 (1.67x) | 88.84 (2.28x) | 70.56 (1.81x) | **102.72 (2.63x)** | 99.93 (2.56x) |
| 2 | 77.19 (1.00x) | 115.78 (1.50x) | 122.96 (1.59x) | 156.20 (2.02x) | 152.37 (1.97x) | **158.72 (2.06x)** | 143.37 (1.86x) |
| 4 | 154.48 (1.00x) | 206.24 (1.34x) | 263.73 (1.71x) | 243.86 (1.58x) | 222.18 (1.44x) | 283.14 (1.83x) | **330.71 (2.14x)** |
| 8 | 287.14 (1.00x) | 369.76 (1.29x) | 422.10 (1.47x) | 453.44 (1.58x) | 479.38 (1.67x) | **534.19 (1.86x)** | 470.06 (1.64x) |
| 16 | 511.59 (1.00x) | 627.72 (1.23x) | 781.41 (1.53x) | 831.48 (1.63x) | **852.72 (1.67x)** | 765.77 (1.50x) | 794.31 (1.55x) |
| 32 | 824.46 (1.00x) | 1080.58 (1.31x) | 1146.32 (1.39x) | **1168.52 (1.42x)** | 1075.29 (1.30x) | 1155.80 (1.40x) | 1124.01 (1.36x) |
| 128 | 1562.60 (1.00x) | **1865.63 (1.19x)** | 1655.41 (1.06x) | 1574.55 (1.01x) | 1529.61 (0.98x) | 1563.69 (1.00x) | 1401.27 (0.90x) |
| 512 | 1941.36 (1.00x) | **2002.07 (1.03x)** | 1859.32 (0.96x) | 1760.88 (0.91x) | 1677.55 (0.86x) | 1635.39 (0.84x) | 1511.09 (0.78x) |

#### Ultra BF16

| BS | K0 baseline | Static K1 | Static K2 | Static K3 | Static K4 | Static K5 | DynamicMTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7.26 (1.00x) | 11.60 (1.60x) | 15.94 (2.19x) | 17.46 (2.40x) | 19.28 (2.66x) | 16.80 (2.31x) | **20.65 (2.84x)** |
| 2 | 14.32 (1.00x) | 21.35 (1.49x) | 30.30 (2.12x) | 32.79 (2.29x) | 33.60 (2.35x) | 34.63 (2.42x) | **40.67 (2.84x)** |
| 4 | 28.06 (1.00x) | 40.96 (1.46x) | 58.48 (2.08x) | 57.15 (2.04x) | 61.06 (2.18x) | 65.26 (2.33x) | **69.50 (2.48x)** |
| 8 | 52.54 (1.00x) | 77.48 (1.47x) | 91.99 (1.75x) | 106.56 (2.03x) | **107.05 (2.04x)** | 101.35 (1.93x) | 104.93 (2.00x) |
| 16 | 95.65 (1.00x) | 132.17 (1.38x) | 152.30 (1.59x) | 162.07 (1.69x) | **167.10 (1.75x)** | 166.64 (1.74x) | 163.26 (1.71x) |
| 32 | 163.71 (1.00x) | 194.55 (1.19x) | 224.03 (1.37x) | 236.92 (1.45x) | **243.79 (1.49x)** | 241.99 (1.48x) | 213.95 (1.31x) |
| 128 | 334.61 (1.00x) | **353.30 (1.06x)** | 348.86 (1.04x) | 319.78 (0.96x) | 336.17 (1.00x) | 324.60 (0.97x) | 289.23 (0.86x) |
| 512 | **383.31 (1.00x)** | 343.53 (0.90x) | 359.87 (0.94x) | 361.03 (0.94x) | 364.75 (0.95x) | 354.21 (0.92x) | 271.04 (0.71x) |

The best fixed K is workload-, model-, and concurrency-dependent. Super uses
K5 most effectively at BS1/2/8, K4 at BS16, K3 at BS32, and K1 at BS128/512;
DynamicMTP wins only BS4. Ultra DynamicMTP wins BS1/2/4, K4 wins BS8/16/32,
K1 wins BS128, and baseline itself wins BS512. The original ladder job
provenance is Super `2817700`-`2817711` and Ultra `2817712`-`2817717`,
`2817719`-`2817724`; the edge/high-concurrency expansion is `2818742`-`2818768`
excluding `2818753`. The K4 jobs are `2819454`-`2819487`, excluding unassigned
IDs `2819455` and `2819459`. Across both workload shapes, all 224 requested
rows are canonical, exact-token complete, and CUDA Graph verified.

## Offered BS512 active-batch finding

| Model | Baseline tok/s/GPU | DynamicMTP tok/s/GPU | Speedup | Acceptance | Mean accepted length | Baseline job | Dynamic job |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Super | 1941.3559 | 1511.0894 | 0.7784x | 74.58% | 2.4976 | `2816828` | `2816642` |
| Ultra | 383.3092 | 271.0352 | 0.7071x | 87.87% | 1.8896 | `2816829` | `2816643` |

Both corrected-harness reruns completed exactly 512000 measured output tokens,
passed `tokens_ok=true`, exited with code zero, and atomically published a
canonical `result.json`. The matched baseline jobs also completed exactly
512000 output tokens with `tokens_ok=true`. DynamicMTP is slower than baseline
at offered BS512 for both models under this schedule.

The original diagnostic jobs (`2816110` for Super and `2816111` for Ultra)
completed generation but withheld canonical publication because the validator
had assumed offered BS512 implied K0. Logs disproved that assumption: Super
admitted about 63 active requests and selected the K2 range, while Ultra
admitted about 73 and selected the K1 range. As each offline batch drained, its
active batch and selected K continued to change.

The corrected harness records dynamic `effective_k` as unknown, records the
requested-batch schedule lookup separately as K0, and identifies
`active_scheduled_batch` as the runtime K-selection basis. A true zero-draft
control requires an explicit all-K0 schedule; ordinary DynamicSD runs must be
interpreted from their observed draft metrics.

## Historical unpatched MRV2 compatibility canary

| Model | Baseline tok/s/GPU | Baseline job | DynamicMTP job | DynamicMTP result |
| --- | ---: | ---: | ---: | --- |
| Super | 96.9387 | `2815808` | `2815809` | Failed during full CUDA graph capture |
| Ultra | 13.7748 | `2815810` | `2815811` | Failed during full CUDA graph capture |

Both MRV2 baselines completed exact output-token validation with
`FULL_AND_PIECEWISE`. Both DynamicMTP jobs failed before generation at the same
vLLM 0.28 assertion in `mamba_attn.py`:

```text
assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only
```

This historical failure motivated the pinned patch stack. It does not apply to
the patched MRV2 matrix above, which passed both FULL and PIECEWISE target and
drafter graph capture in all 64 canonical cells.

## Provenance

Published artifacts:

- [Patched MRV2 full-matrix CSV](../../public/data/vllm028_nemotron3_bf16_mrv2_patched_matrix_20260828/results.csv)
- [Patched MRV2 full-matrix HTML](../../public/reports/vllm028_nemotron3_bf16_mrv2_patched_matrix_20260828.html)
- [Patched MRV2 source manifest](../../public/data/vllm028_nemotron3_bf16_mrv2_patched_matrix_20260828/source_manifest.json)
- [Patched MRV2 Google Sheet](https://docs.google.com/spreadsheets/d/1A4-XehbdBiTRoKc3MTl8fHZ79s3DAx-1NuTu-bwZogw/edit)
- [Canonical CSV](../../public/data/vllm028_nemotron3_bf16_dynamicsd_20260827/results.csv)
- [Interactive HTML report](../../public/reports/vllm028_nemotron3_bf16_dynamicsd_20260827.html)
- [Native Google Sheet](https://docs.google.com/spreadsheets/d/1LD-a-yRJBcJ5L1e3IrZxQIywYTb64OpwPu-IxO6q5yU/edit)

- Canary harness commit: `76656d1f1159ccdd44b2290e74e85b755f2421ef`
- 10K/1K gate expansion harness commit: `6d4ac2da89abf64b28f4e6baa8df06798c4747dd`
- BS512 canonical rerun harness commit: `f0dd8af3110820c708de2ce7b0720970f4c8ef8c`
- BS512 canonical jobs: Super `2816642`, Ultra `2816643`
- BS512 matched baseline jobs: Super `2816828`, Ultra `2816829`
- Container staging job: `2815382`
- Super BF16 staging job: `2815383`
- Ray: 2.48.0, exact hash-locked ARM64 sidecar
- Result root: `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-benchmark-results/vllm028-nemotron-bf16-smoke`

The repository also retains the 64 curated canonical `result.json` files and
their 64 CUDA Graph evidence files under
`artifacts/mrv2_patch_matrix/curated_results`. The source manifest binds every
file to its matrix key and SLURM job ID with SHA256 so the normalized outputs
can be regenerated without access to the cluster.
