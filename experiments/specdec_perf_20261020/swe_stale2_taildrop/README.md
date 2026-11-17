# SWE tail-drop A/B at staleness 2

Does a drop of five slow, never-resolved instances make the Q30 SWE GBS 32 step faster at staleness 2?

## Why

The ready-first gate waits for the slowest group in the admission window. Exposed generation is close to the gate group span minus its age. Eight GBS 32 runs (stale2 A/B and run-ahead A/B, 10-30 to 11-01) saw 92 instances, 724 groups. The group span p50 is 174 s and the p90 is 294 s. Five instances resolved 0 of 64 attempts and had a mean span of 380-788 s. A group with zero resolves has zero advantage. Thus these groups add step time and give no gradient.

| Instance | Line in drop5 | Mean span (s) | Max span (s) | Resolved |
|---|---|---|---|---|
| singapore__renovate-527 | 8 | 788 | 813 | 0/64 (600 s init timeout) |
| jtablesaw__tablesaw-559 | 47 | 599 | 764 | 0/64 |
| kroxylicious__kroxylicious-586 | 58 | 483 | 637 | 0/64 |
| juliasymbolics__symbolics.jl-552 | 56 | 441 | 683 | 0/64 |
| graphhopper__graphhopper-1911 | 4 | 380 | 528 | 0/64 |

The rule: 0 resolves in all attempts and a mean span above the p90. 73 of 92 instances have 0 resolves, so a drop of all of them is not a system change. We drop only the five tail ones.

## Arms

| Arm | Config | Difference |
|---|---|---|
| s2 | `swe_traj_l0_staleness2_drop5.yaml` | run-ahead control + `rf_staleness2.yaml` |
| tail | `swe_traj_l0_staleness2_droptail.yaml` | s2 + `drop_tail_instances.yaml` (data path only) |

Both arms: 20 steps, GBS 32, 16 vLLM replicas, no drafter, no checkpoints. Two replicates per arm.

## Usage

```bash
# Once, on the login node: build the data file.
python filter_swe_instances.py $D/balanced_language_drop5.jsonl $D/balanced_language_drop5_tail5.jsonl \
  singapore__renovate-527 jtablesaw__tablesaw-559 kroxylicious__kroxylicious-586 \
  juliasymbolics__symbolics.jl-552 graphhopper__graphhopper-1911
WANT=<full sha> ARMS=both bash swe_stale2_taildrop/submit.sh
WANT=<full sha> ARMS=both SFX=-r2 bash swe_stale2_taildrop/submit.sh   # after the first pair ends
python3 swe_stale2_taildrop/swe_instance_tail.py <results roots...>      # tail table
```

## Caveats

- The drop changes the training data. Later lines move up into the first 92 slots. The new lines have unknown spans. Report the reward and the instance set next to the step time.
- The five instances can become solvable in a longer run. This A/B tests only the 20-step system effect.
- Compare steps 2-20. Step 1 holds the startup cost.