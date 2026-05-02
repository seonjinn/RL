#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


OUT_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs" / "recipes" / "llm" / "performance"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.rstrip() + "\n"
    if path.exists() and path.read_text() == normalized:
        return
    path.write_text(normalized)


def yaml_list(items: list[str], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {item}" for item in items)


def maybe_seqpack_block(enabled: bool, indent: int = 4) -> str:
    pad = " " * indent
    inner = " " * (indent + 2)
    if enabled:
        return (
            f"{pad}sequence_packing:\n"
            f"{inner}enabled: true\n"
            f"{inner}train_mb_tokens: 4096\n"
            f"{inner}logprob_mb_tokens: 4096\n"
            f"{inner}algorithm: modified_first_fit_decreasing\n"
            f"{inner}sequence_length_round: 64\n"
        )
    return (
        f"{pad}sequence_packing:\n"
        f"{inner}enabled: false\n"
    )


def ll8_base(name: str, seqpack: bool) -> str:
    return (
        "defaults: ../../../grpo_math_1B.yaml\n"
        "grpo:\n"
        "  num_prompts_per_step: 64\n"
        "  num_generations_per_prompt: 16\n"
        "  max_num_steps: 20\n"
        "loss_fn:\n"
        "  use_importance_sampling_correction: true\n"
        "checkpointing:\n"
        "  enabled: false\n"
        f"  checkpoint_dir: results/{name}\n"
        "policy:\n"
        "  model_name: meta-llama/Llama-3.1-8B-Instruct\n"
        "  tokenizer:\n"
        "    name: meta-llama/Llama-3.1-8B-Instruct\n"
        "  train_global_batch_size: 64\n"
        "  train_micro_batch_size: 1\n"
        "  logprob_batch_size: 2\n"
        "  max_total_sequence_length: 4096\n"
        "  make_sequence_length_divisible_by: 1\n"
        "  dtensor_cfg:\n"
        "    enabled: false\n"
        "  optimizer: null\n"
        "  scheduler: null\n"
        f"{maybe_seqpack_block(seqpack, indent=2)}"
        "  megatron_cfg:\n"
        "    enabled: true\n"
        "    empty_unused_memory_level: 1\n"
        "    converter_type: LlamaForCausalLM\n"
        "    tensor_model_parallel_size: 1\n"
        "    pipeline_model_parallel_size: 1\n"
        "    context_parallel_size: 1\n"
        "    sequence_parallel: false\n"
        "    activation_checkpointing: false\n"
        "    defer_fp32_logits: true\n"
        "    apply_rope_fusion: true\n"
        "    bias_activation_fusion: false\n"
        "    optimizer:\n"
        "      lr: 5.0e-07\n"
        "      min_lr: 5.0e-08\n"
        "      weight_decay: 0.0\n"
        "      use_precision_aware_optimizer: true\n"
        "      use_distributed_optimizer: true\n"
        "    scheduler:\n"
        "      lr_warmup_iters: 2\n"
        "      lr_warmup_init: 5.0e-08\n"
        "    fp8_cfg:\n"
        "      enabled: false\n"
        "  generation:\n"
        "    max_new_tokens: 4096\n"
        "    stop_token_ids:\n"
        "    - 128009\n"
        "    vllm_cfg:\n"
        "      max_model_len: 4096\n"
        "      tensor_parallel_size: 1\n"
        "      gpu_memory_utilization: 0.5\n"
        "data:\n"
        "  max_input_seq_length: 2048\n"
        "logger:\n"
        f"  log_dir: logs/{name}\n"
        "  wandb_enabled: false\n"
        "  tensorboard_enabled: false\n"
        "  wandb:\n"
        f"    name: {name}\n"
        "cluster:\n"
        "  gpus_per_node: 4\n"
        "  num_nodes: 1\n"
    )


def qw8_base(name: str, seqpack: bool) -> str:
    return (
        "defaults: ../../../grpo_math_1B.yaml\n"
        "grpo:\n"
        "  num_prompts_per_step: 64\n"
        "  num_generations_per_prompt: 16\n"
        "  max_num_steps: 20\n"
        "loss_fn:\n"
        "  use_importance_sampling_correction: true\n"
        "checkpointing:\n"
        "  enabled: false\n"
        f"  checkpoint_dir: results/{name}\n"
        "policy:\n"
        "  model_name: Qwen/Qwen3-8B-Base\n"
        "  tokenizer:\n"
        "    name: Qwen/Qwen3-8B-Base\n"
        "  train_global_batch_size: 64\n"
        "  train_micro_batch_size: 1\n"
        "  logprob_batch_size: 2\n"
        "  max_total_sequence_length: 4096\n"
        "  make_sequence_length_divisible_by: 1\n"
        "  dtensor_cfg:\n"
        "    enabled: false\n"
        "  optimizer: null\n"
        "  scheduler: null\n"
        f"{maybe_seqpack_block(seqpack, indent=2)}"
        "  megatron_cfg:\n"
        "    enabled: true\n"
        "    empty_unused_memory_level: 1\n"
        "    converter_type: Qwen3ForCausalLM\n"
        "    tensor_model_parallel_size: 1\n"
        "    pipeline_model_parallel_size: 1\n"
        "    context_parallel_size: 1\n"
        "    sequence_parallel: false\n"
        "    activation_checkpointing: false\n"
        "    defer_fp32_logits: true\n"
        "    apply_rope_fusion: true\n"
        "    bias_activation_fusion: false\n"
        "    optimizer:\n"
        "      lr: 3.0e-07\n"
        "      min_lr: 3.0e-08\n"
        "      weight_decay: 0.0\n"
        "      use_precision_aware_optimizer: true\n"
        "      use_distributed_optimizer: true\n"
        "    scheduler:\n"
        "      lr_warmup_iters: 2\n"
        "      lr_warmup_init: 3.0e-08\n"
        "    fp8_cfg:\n"
        "      enabled: false\n"
        "  generation:\n"
        "    max_new_tokens: 4096\n"
        "    vllm_cfg:\n"
        "      max_model_len: 4096\n"
        "      tensor_parallel_size: 1\n"
        "      gpu_memory_utilization: 0.5\n"
        "data:\n"
        "  max_input_seq_length: 2048\n"
        "logger:\n"
        f"  log_dir: logs/{name}\n"
        "  wandb_enabled: false\n"
        "  tensorboard_enabled: false\n"
        "  wandb:\n"
        f"    name: {name}\n"
        "cluster:\n"
        "  gpus_per_node: 4\n"
        "  num_nodes: 1\n"
    )


def q30_nocg(name: str, seqpack: bool) -> str:
    return (
        "defaults: ./grpo-qwen3-30ba3b-4n4g.yaml\n"
        "grpo:\n"
        "  max_num_steps: 20\n"
        "loss_fn:\n"
        "  use_importance_sampling_correction: true\n"
        "checkpointing:\n"
        "  enabled: false\n"
        f"  checkpoint_dir: results/{name}\n"
        "policy:\n"
        f"{maybe_seqpack_block(seqpack, indent=2)}"
        "logger:\n"
        f"  log_dir: logs/{name}\n"
        "  wandb_enabled: false\n"
        "  tensorboard_enabled: false\n"
        "  wandb:\n"
        f"    name: {name}\n"
    )


def cg_variant_body(defaults: str, name: str, scope: str | list[str], warmup: int, cg_packed_seq: bool) -> str:
    scope_lines = (
        f"    cuda_graph_scope: {scope}\n"
        if isinstance(scope, str)
        else "    cuda_graph_scope:\n" + yaml_list(scope, 6) + "\n"
    )
    return (
        f"defaults: {defaults}\n"
        f"checkpointing:\n"
        f"  enabled: false\n"
        f"  checkpoint_dir: results/{name}\n"
        f"policy:\n"
        f"  megatron_cfg:\n"
        f"    cuda_graph_impl: transformer_engine\n"
        f"{scope_lines}"
        f"    cuda_graph_warmup_steps: {warmup}\n"
        f"    cuda_graph_packed_seq: {'true' if cg_packed_seq else 'false'}\n"
        f"    cuda_graph_buckets:\n"
        f"    - 4096\n"
        f"logger:\n"
        f"  log_dir: logs/{name}\n"
        f"  wandb_enabled: false\n"
        f"  tensorboard_enabled: false\n"
        f"  wandb:\n"
        f"    name: {name}\n"
    )


def alias_overlay(defaults: str, name: str) -> str:
    return (
        f"defaults: {defaults}\n"
        f"checkpointing:\n"
        f"  enabled: false\n"
        f"  checkpoint_dir: results/{name}\n"
        f"logger:\n"
        f"  log_dir: logs/{name}\n"
        f"  wandb_enabled: false\n"
        f"  tensorboard_enabled: false\n"
        f"  wandb:\n"
        f"    name: {name}\n"
    )


def main() -> None:
    # Base nocg files
    write(OUT_DIR / "grpo-llama3.1-8b-instruct-1n4g-nocg.yaml", ll8_base("grpo-llama3.1-8b-instruct-1n4g-nocg", True))
    write(OUT_DIR / "grpo-llama3.1-8b-instruct-1n4g-nocg-nopack.yaml", ll8_base("grpo-llama3.1-8b-instruct-1n4g-nocg-nopack", False))
    write(OUT_DIR / "grpo-qwen3-8b-1n4g-nocg.yaml", qw8_base("grpo-qwen3-8b-1n4g-nocg", True))
    write(OUT_DIR / "grpo-qwen3-8b-1n4g-nocg-nopack.yaml", qw8_base("grpo-qwen3-8b-1n4g-nocg-nopack", False))
    write(OUT_DIR / "grpo-qwen3-30ba3b-4n4g-nocg.yaml", q30_nocg("grpo-qwen3-30ba3b-4n4g-nocg", True))
    write(OUT_DIR / "grpo-qwen3-30ba3b-4n4g-nocg-nopack.yaml", q30_nocg("grpo-qwen3-30ba3b-4n4g-nocg-nopack", False))

    # Compatibility/base CG configs
    write(
        OUT_DIR / "grpo-llama3.1-8b-instruct-1n4g-cg.yaml",
        cg_variant_body("./grpo-llama3.1-8b-instruct-1n4g-nocg.yaml", "grpo-llama3.1-8b-instruct-1n4g-cg", "attn", 3, True),
    )
    write(
        OUT_DIR / "grpo-qwen3-8b-1n4g-cg.yaml",
        cg_variant_body("./grpo-qwen3-8b-1n4g-nocg.yaml", "grpo-qwen3-8b-1n4g-cg", ["attn", "mlp"], 6, False),
    )

    # Llama 8B variants
    for warmup in (3, 6):
        for seqpack, suffix, packed in ((True, "", True), (False, "-nopack", False)):
            defaults = f"./grpo-llama3.1-8b-instruct-1n4g-nocg{suffix}.yaml"
            combos = {
                f"grpo-llama3.1-8b-instruct-1n4g-cg-attn-w{warmup}{suffix}.yaml": "attn",
                f"grpo-llama3.1-8b-instruct-1n4g-cg-mlp-w{warmup}{suffix}.yaml": "mlp",
                f"grpo-llama3.1-8b-instruct-1n4g-cg-attn-mlp-w{warmup}{suffix}.yaml": ["attn", "mlp"],
            }
            for filename, scope in combos.items():
                stem = filename[:-5]
                write(OUT_DIR / filename, cg_variant_body(defaults, stem, scope, warmup, packed))

    # Qwen3-8B variants
    for warmup in (3, 6):
        for seqpack, suffix in ((True, "",), (False, "-nopack")):
            defaults = f"./grpo-qwen3-8b-1n4g-nocg{suffix}.yaml"
            combos = {
                f"grpo-qwen3-8b-1n4g-cg-attn-w{warmup}{suffix}.yaml": "attn",
                f"grpo-qwen3-8b-1n4g-cg-mlp-w{warmup}{suffix}.yaml": "mlp",
                f"grpo-qwen3-8b-1n4g-cg-attn-mlp-w{warmup}{suffix}.yaml": ["attn", "mlp"],
            }
            for filename, scope in combos.items():
                stem = filename[:-5]
                write(OUT_DIR / filename, cg_variant_body(defaults, stem, scope, warmup, False))

    # Preserve a few historically used Qwen8 files for convenience
    write(
        OUT_DIR / "grpo-qwen3-8b-1n4g-nocg-pack8192.yaml",
        dedent(
            """\
            defaults: ./grpo-qwen3-8b-1n4g-nocg.yaml
            checkpointing:
              enabled: false
              checkpoint_dir: results/grpo-qwen3-8b-1n4g-nocg-pack8192
            policy:
              sequence_packing:
                enabled: true
                train_mb_tokens: 8192
                logprob_mb_tokens: 8192
                algorithm: modified_first_fit_decreasing
                sequence_length_round: 64
            logger:
              log_dir: logs/grpo-qwen3-8b-1n4g-nocg-pack8192
              wandb_enabled: false
              tensorboard_enabled: false
              wandb:
                name: grpo-qwen3-8b-1n4g-nocg-pack8192
            """
        ),
    )
    write(
        OUT_DIR / "grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack8192.yaml",
        dedent(
            """\
            defaults: ./grpo-qwen3-8b-1n4g-cg-attn-mlp-w6.yaml
            checkpointing:
              enabled: false
              checkpoint_dir: results/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack8192
            policy:
              sequence_packing:
                enabled: true
                train_mb_tokens: 8192
                logprob_mb_tokens: 8192
                algorithm: modified_first_fit_decreasing
                sequence_length_round: 64
            logger:
              log_dir: logs/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack8192
              wandb_enabled: false
              tensorboard_enabled: false
              wandb:
                name: grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack8192
            """
        ),
    )
    write(
        OUT_DIR / "grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack-cgpackoff.yaml",
        dedent(
            """\
            defaults: ./grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack8192.yaml
            checkpointing:
              enabled: false
              checkpoint_dir: results/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack-cgpackoff
            policy:
              megatron_cfg:
                cuda_graph_packed_seq: false
            logger:
              log_dir: logs/grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack-cgpackoff
              wandb_enabled: false
              tensorboard_enabled: false
              wandb:
                name: grpo-qwen3-8b-1n4g-cg-attn-mlp-w6-pack-cgpackoff
            """
        ),
    )

    # Qwen3-30B-A3B variants
    q30_scopes: dict[str, str | list[str]] = {
        "attn": "attn",
        "mlp": "mlp",
        "moe-router": "moe_router",
        "attn-mlp": ["attn", "mlp"],
        "attn-moe-router": ["attn", "moe_router"],
        "mlp-moe-router": ["mlp", "moe_router"],
        "attn-mlp-moe-router": ["attn", "mlp", "moe_router"],
    }
    for warmup in (3, 6):
        for seqpack, suffix in ((True, ""), (False, "-nopack")):
            defaults = f"./grpo-qwen3-30ba3b-4n4g-nocg{suffix}.yaml"
            for tag, scope in q30_scopes.items():
                filename = f"grpo-qwen3-30ba3b-4n4g-cg-{tag}-w{warmup}{suffix}.yaml"
                stem = filename[:-5]
                write(OUT_DIR / filename, cg_variant_body(defaults, stem, scope, warmup, False))

    # Historical aliases without warmup in name
    write(
        OUT_DIR / "grpo-qwen3-30ba3b-4n4g-cg-attn.yaml",
        alias_overlay("./grpo-qwen3-30ba3b-4n4g-cg-attn-w6.yaml", "grpo-qwen3-30ba3b-4n4g-cg-attn"),
    )
    write(
        OUT_DIR / "grpo-qwen3-30ba3b-4n4g-cg-moe-router.yaml",
        alias_overlay("./grpo-qwen3-30ba3b-4n4g-cg-moe-router-w6.yaml", "grpo-qwen3-30ba3b-4n4g-cg-moe-router"),
    )
    write(
        OUT_DIR / "grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml",
        alias_overlay("./grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w6.yaml", "grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router"),
    )
    write(
        OUT_DIR / "grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-nopack.yaml",
        alias_overlay(
            "./grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-w6-nopack.yaml",
            "grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router-nopack",
        ),
    )
    write(
        OUT_DIR / "grpo-qwen3-30ba3b-4n4g-cg-attn-moe.yaml",
        dedent(
            """\
            defaults: ./grpo-qwen3-30ba3b-4n4g-cg-attn-moe-router.yaml
            checkpointing:
              enabled: false
              checkpoint_dir: results/grpo-qwen3-30ba3b-4n4g-cg-attn-moe
            logger:
              log_dir: logs/grpo-qwen3-30ba3b-4n4g-cg-attn-moe
              wandb_enabled: false
              tensorboard_enabled: false
              wandb:
                name: grpo-qwen3-30ba3b-4n4g-cg-attn-moe
            """
        ),
    )


if __name__ == "__main__":
    main()
