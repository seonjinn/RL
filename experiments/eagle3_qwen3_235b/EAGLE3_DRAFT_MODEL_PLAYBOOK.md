# Eagle3 Draft Model Playbook

This playbook describes the reusable path for building an Eagle3 draft model
for a verifier model, with Qwen3-235B Thinking as the current target. It is the
general answer to: "What do we need to do to make an Eagle3 draft for an
arbitrary model, and what changes in an RL context?"

## Scope

The path assumes the verifier is supported by ModelOpt Eagle3 training and by
the serving stack used for speculative decoding. The included scripts are tuned
for Qwen/LLaMA-like Hugging Face configs, where the verifier `config.json`
contains layer count, attention heads, KV heads, hidden size, FFN size, RMS norm
epsilon, and RoPE settings.

For unsupported architectures, the same high-level gates still apply, but the
architecture derivation and hidden-state capture code must be extended first.

## Required Inputs

Collect these before launching GPU work:

- verifier model or local verifier config directory,
- tokenizer config and chat template,
- target-domain prompts or rollout conversations with assistant responses,
- ModelOpt checkout and version/patch state,
- vLLM/NeMo-RL branch that will load the draft,
- artifact root with enough space for hidden states and checkpoints,
- Slurm account, partition, container image, and mounts.

For Qwen3-235B Thinking, the verifier is:

```text
Qwen/Qwen3-235B-A22B-Thinking-2507
```

The public draft `nvidia/Qwen3-235B-A22B-Eagle3` is useful only as a vLLM
compatibility smoke test because it was trained for the non-thinking verifier.

## Canonical Artifact Flow

For the Qwen3-235B Thinking RL path, the training packet is not complete just
because the architecture and scripts exist. The artifacts must advance in this
order:

```text
rollout_conversation_corpus
  -> verifier_hidden_states
  -> modelopt_checkpoint
  -> hf_eagle3_export
  -> vllm_eagle3_draft
  -> rl_vllm_draft_validation
```

The manifest field `artifact_flow_complete` may be true only when every
artifact row has `proof_status=pass`. Do not claim completion while any row is
open, even if the local wrappers and static validators pass.

The current handoff packet exposes a gate/action matrix. For this packet, the
ready actions are:

- `remote_hayate_reference_probe`: `probe_remote_hosts`
- `runtime_container`: `probe_remote_hosts`,
  `submit_vllm_source_build`, `poll_megatron_compat_probe`,
  `submit_container_preflight`

The rollout, hidden-state, train, export, and trained-draft sweep actions stay
future candidates until their producer gates close. In particular, do not run
a hidden-state dump or ModelOpt training before the actual Qwen3 SWE/RL rollout corpus exists.
The source-built vLLM ABI probe, Megatron compatibility probe, and container preflight must also have passing evidence.

Hayate artifacts remain reference-only. They can explain workflow shape and
cross-check architecture fields, but the local or remote current ModelOpt
checkout and the Thinking-2507 verifier config are the source of truth for this
training path.

## General Procedure

1. Derive the Eagle3 architecture from the verifier `config.json`.

   Use:

   ```bash
   python3 experiments/eagle3_qwen3_235b/derive_eagle3_architecture.py \
     --verifier-config /path/to/verifier/config.json \
     --json-out /path/to/eagle3_architecture.json \
     --env-out /path/to/eagle3_architecture.env \
     --dotlist-out /path/to/eagle3_architecture.dotlist
   ```

   The helper mirrors ModelOpt's default EAGLE-3 auxiliary-layer rule:

   ```text
   sorted({1, max(0, num_hidden_layers // 2 - 1), max(0, num_hidden_layers - 4)})
   ```

   For a 94-layer Qwen3-235B verifier this gives `[1, 46, 90]`.

2. Prepare target-domain conversations.

   The training JSONL must contain complete user/context plus assistant answer
   text. For RL, prefer actual rollout traces from the same SWE/task
   distribution. If rollouts are not available yet, generate responses from the
   current policy and treat that as a bootstrap dataset for pilot-only work.
   The final completion audit still requires the actual Qwen3 SWE/RL rollout
   corpus artifact.

   Validate before hidden-state dump:

   ```bash
   python3 experiments/eagle3_qwen3_235b/validate_training_conversations.py \
     /path/to/conversations.jsonl \
     --max-seq-len 16384 \
     --json-out /path/to/conversations.validation.json
   ```

3. Prepare and validate answer-only loss masking.

   For chat/instruction RL data, the draft should learn assistant tokens, not
   prompt, tool, or context tokens. Use a chat template with Transformers
   generation tags and validate that it produces a positive assistant-token mask:

   ```bash
   TOKENIZER_CONFIG=/path/to/tokenizer_config.json \
   OUTPUT_TEMPLATE=/path/to/generation_template.jinja2 \
   bash experiments/eagle3_qwen3_235b/prepare_qwen3_chat_template.sh

   python3 experiments/eagle3_qwen3_235b/validate_chat_template_loss_mask.py \
     --model-or-tokenizer /path/to/verifier_or_tokenizer \
     --chat-template /path/to/generation_template.jinja2
   ```

4. Choose offline or ModelOpt online training.

   Offline is the recommended first path for a 235B verifier:

   - dump hidden states once,
   - validate shapes and masks,
   - train the draft from saved tensors,
   - resume individual stages independently.

   ModelOpt online training uses `data.data_path` and keeps the verifier in the
   training job. It matches Hayate's later workflow more closely, but it is much
   heavier for 235B because every training step needs verifier forward work.

   Do not confuse ModelOpt online training with online RL draft training.
   NeMo-RL online draft training means the draft evolves inside the RL loop as
   the policy changes. It requires NeMo-RL support for trainer-owned draft loss,
   Megatron enabled, DTensor disabled, and sequence packing disabled.

5. Dump verifier hidden states.

   The dump must include the final hidden state plus the selected auxiliary
   hidden states, and it should include `loss_mask` when answer-only loss is
   enabled.

   ```bash
   ARCH_ENV_FILE=/path/to/eagle3_architecture.env \
   INPUT_DATA=/path/to/conversations.jsonl \
   HIDDEN_STATES_DIR=/path/to/hidden_states \
   BACKEND=trtllm \
   TP=8 \
   DP_WORLD_SIZE=8 \
   ANSWER_ONLY_LOSS=true \
   CHAT_TEMPLATE=/path/to/generation_template.jinja2 \
   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_dump_hidden_states.sh
   ```

   Validate the output before training:

   ```bash
   python3 experiments/eagle3_qwen3_235b/validate_hidden_state_dump.py \
     /path/to/hidden_states \
     --require-loss-mask \
     --require-positive-loss-mask \
     --expected-hidden-size 4096 \
     --expected-aux-count 3 \
     --max-seq-len 16384 \
     --validate-modelopt-loader \
     --modelopt-dir /path/to/Model-Optimizer
   ```

   For a non-Qwen model, set `--expected-hidden-size` from the derived
   architecture.

6. Train the Eagle3 draft.

   Offline training:

   ```bash
   ARCH_ENV_FILE=/path/to/eagle3_architecture.env \
   HIDDEN_STATES_DIR=/path/to/hidden_states \
   OUTPUT_DIR=/path/to/modelopt_ckpt \
   ANSWER_ONLY_LOSS=true \
   USE_FAKE_BASE_FOR_OFFLINE=true \
   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_offline_train.sh
   ```

   ModelOpt online training:

   ```bash
   ARCH_ENV_FILE=/path/to/eagle3_architecture.env \
   INPUT_DATA=/path/to/conversations.jsonl \
   OUTPUT_DIR=/path/to/modelopt_online_ckpt \
   ANSWER_ONLY_LOSS=true \
   CHAT_TEMPLATE=/path/to/generation_template.jinja2 \
   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_online_train.sh
   ```

   Start with a pilot: small conversation count, short max steps, and one saved
   checkpoint. Scale only after chat-template masks, hidden-state shapes,
   ModelOpt loader compatibility, and export all pass.

7. Export and compare configs.

   ```bash
   TRAINED_CKPT=/path/to/modelopt_ckpt \
   EXPORT_DIR=/path/to/exported_hf \
   VLLM_DRAFT_DIR=/path/to/vllm_draft \
   VERIFIER_CONFIG_DIR=/path/to/verifier \
   REFERENCE_ARCH=/path/to/eagle3_architecture.json \
   bash experiments/eagle3_qwen3_235b/modelopt_qwen3_235b_export_vllm.sh
   ```

   The exported draft config must match the verifier-derived architecture:
   hidden size, attention heads, KV heads, FFN size, aux layer ids, norm eps,
   and RoPE settings.

8. Validate in the RL generation path.

   Use a baseline-vs-specdec smoke pair before any long RL run. Then sweep
   `num_speculative_tokens` rather than assuming one value:

   ```bash
   SUBMIT=false \
   VLLM_DRAFT_DIR=/path/to/vllm_draft \
   SPEC_TOKENS_LIST="2 3 4" \
   MAX_NUM_STEPS=2 \
   bash experiments/eagle3_qwen3_235b/submit_trained_draft_spec_tokens_sweep.sh
   ```

   The pass condition is not standalone throughput. The RL gate must include
   exposed generation time, acceptance rate, reward, malformed output rate,
   environment/tool errors, and any task-specific regression signal.

## RL-Specific Decisions

Static draft:

- lowest-risk first step,
- train once against the current policy/verifier,
- load through `vllm_kwargs.speculative_config`,
- monitor acceptance decay as the policy changes.

Periodically refreshed draft:

- retrain from recent rollout traces or newer policy checkpoints,
- use when acceptance falls below the target range,
- requires artifact/version discipline but not deep NeMo-RL changes.

Online RL draft training:

- train draft weights inside the RL loop,
- validate the checkout with
  `validate_nemo_rl_specdec_integration.py --integration-mode online-draft-training`,
- capture policy hidden states and logits at Eagle3 layers,
- manage draft parameters under Megatron/FSDP,
- checkpoint/export draft weights separately from the verifier,
- sync draft weights to generation workers after policy updates.

This is a NeMo-RL feature project, not just a ModelOpt recipe change.

## Hayate Findings To Preserve

The accessible Hayate/Hiso ModelOpt branch showed workflow-level changes:

- response aggregation for many completions per prompt,
- hidden-state dump aggregation to reduce file fan-out,
- move from older offline scripts toward online ModelOpt training,
- tokenizer output handling fixes,
- Slurm chaining for repeated short jobs.

No committed core `modelopt/torch/...` speculative-decoding library changes
were found in the accessible checkout. The Qwen3 path here therefore keeps
ModelOpt changes narrowly scoped to the TRT-LLM hidden-state dumper patch needed
for RL answer-only loss masking.

An additional accessible Hayate/Hiso SpecForge checkout exists at:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/SpecForge
```

This is an SGLang ecosystem project and is a stronger reference for Eagle3
training shapes than the earlier inaccessible draft directory. It contains
Qwen3 8B/30B draft outputs and a `configs/qwen3-235B-A22B-eagle3.json` file.
The 235B config agrees with our 94-layer aux-layer selection `[1, 46, 90]`, but
does not match the Thinking-2507 verifier on RoPE theta, max position, or
intermediate size, so it is reference evidence, not a config source of truth.
See `HAYATE_SPECFORGE_REFERENCE.md` for the comparison.

The current Qwen3-235B RL rollout path also needs a Megatron-Bridge
compatibility shim that Hayate's SpecForge path does not exercise:

```text
experiments/eagle3_qwen3_235b/megatron_bridge_qwen3moe/
```

Probe `2860778` proved the shim registers `Qwen3MoeForCausalLM` with the
container's existing `/opt/Megatron-Bridge` and creates a synthetic Qwen3MoE
provider. Full newer Megatron-Bridge replacement is not the default because it
collides with the target NeMo container's Megatron-Core/Transformers stack.

## Completion Checklist

The draft path is complete only when these are all proven for the target model:

- verifier config and derived Eagle3 architecture are recorded,
- target-domain training conversations pass validation,
- chat template produces positive assistant-token masks,
- hidden-state dump passes shape, aux-count, loss-mask, and ModelOpt loader
  validation,
- ModelOpt training produces a checkpoint,
- export produces HF and vLLM draft artifacts,
- exported config matches verifier-derived architecture,
- baseline-vs-trained-draft smoke runs in the RL generation stack,
- `num_speculative_tokens` sweep selects a passing setting,
- manifest `artifact_flow_complete` is true and every artifact-flow row has
  `proof_status=pass`,
- completion audit reports PASS from the produced artifacts.
