from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


EXPECTED_MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-30ba3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}


@dataclass(frozen=True)
class Case:
    run_key: str
    model: str
    config_name: str
    mode: str
    force_on_policy_ratio: bool
    nodes: int
    gpus_per_node: int
    segment: int
    global_batch_size: int
    steps: int
    time_limit: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Case:
        force_value = row["force_on_policy_ratio"]
        assert force_value in {"false", "true"}
        assert row["mode"] in {"sync", "async1off"}
        return cls(
            run_key=row["run_key"],
            model=row["model"],
            config_name=row["config_name"],
            mode=row["mode"],
            force_on_policy_ratio=force_value == "true",
            nodes=int(row["nodes"]),
            gpus_per_node=int(row["gpus_per_node"]),
            segment=int(row["segment"]),
            global_batch_size=int(row["global_batch_size"]),
            steps=int(row["steps"]),
            time_limit=row["time_limit"],
        )

    def overrides(self) -> list[str]:
        force_value = str(self.force_on_policy_ratio).lower()
        return [
            f"grpo.max_num_steps={self.steps}",
            "checkpointing.enabled=false",
            f"policy.train_global_batch_size={self.global_batch_size}",
            f"loss_fn.force_on_policy_ratio={force_value}",
            f"cluster.segment_size={self.segment}",
        ]


def load_cases(contract_path: Path) -> list[Case]:
    with contract_path.open(newline="") as contract_file:
        return [
            Case.from_row(row)
            for row in csv.DictReader(contract_file, delimiter="\t")
        ]


def resolve_case(case: Case) -> dict[str, object]:
    config_path = Path("examples/configs/recipes/llm/performance") / (
        f"{case.config_name}.yaml"
    )
    config = load_config(str(config_path))
    config = parse_hydra_overrides(config, case.overrides())
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def validate_case(case: Case, resolved: dict[str, object]) -> None:
    grpo = resolved["grpo"]
    policy = resolved["policy"]
    loss_fn = resolved["loss_fn"]
    cluster = resolved["cluster"]
    checkpointing = resolved["checkpointing"]

    assert isinstance(grpo, dict)
    assert isinstance(policy, dict)
    assert isinstance(loss_fn, dict)
    assert isinstance(cluster, dict)
    assert isinstance(checkpointing, dict)

    async_grpo = grpo["async_grpo"]
    assert isinstance(async_grpo, dict)
    expected_async = case.mode == "async1off"

    assert policy["model_name"] == EXPECTED_MODELS[case.model]
    assert policy["train_global_batch_size"] == case.global_batch_size == 2048
    assert policy["train_micro_batch_size"] == 1
    assert grpo["num_prompts_per_step"] == 64
    assert grpo["num_generations_per_prompt"] == 32
    assert (
        grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
        == policy["train_global_batch_size"]
    )
    assert async_grpo["enabled"] is expected_async
    assert async_grpo["in_flight_weight_updates"] is expected_async
    assert async_grpo["max_trajectory_age_steps"] == 1
    assert grpo["seq_logprob_error_threshold"] is None
    assert loss_fn["force_on_policy_ratio"] is case.force_on_policy_ratio
    assert loss_fn["reference_policy_kl_penalty"] == 0.01
    assert cluster["num_nodes"] == case.nodes
    assert cluster["gpus_per_node"] == case.gpus_per_node == 4
    assert cluster["segment_size"] == case.segment == case.nodes
    assert case.steps == 20
    assert case.time_limit in {"02:00:00", "03:00:00", "04:00:00"}
    assert "8g" not in case.config_name
    assert checkpointing["enabled"] is False


def normalized_pair_config(resolved: dict[str, object]) -> dict[str, object]:
    normalized = OmegaConf.create(resolved)
    normalized.loss_fn.force_on_policy_ratio = False
    normalized.logger.log_dir = "PAIR_NORMALIZED"
    normalized.logger.wandb.name = "PAIR_NORMALIZED"
    result = OmegaConf.to_container(normalized, resolve=True)
    assert isinstance(result, dict)
    return result


def main() -> None:
    register_omegaconf_resolvers()
    script_dir = Path(__file__).resolve().parent
    contract_path = script_dir.parent / "manifests" / "config_contract.tsv"
    cases = load_cases(contract_path)
    assert len(cases) == 8

    resolved_by_config: dict[str, list[tuple[Case, dict[str, object]]]] = {}
    for case in cases:
        resolved = resolve_case(case)
        validate_case(case, resolved)
        resolved_by_config.setdefault(case.config_name, []).append((case, resolved))
        print(
            f"CONFIG_OK {case.run_key} model={case.model} mode={case.mode} "
            f"force={str(case.force_on_policy_ratio).lower()} "
            f"nodes={case.nodes} gpus_per_node={case.gpus_per_node} "
            f"global_batch={case.global_batch_size}"
        )

    assert len(resolved_by_config) == 4
    for config_name, pair in resolved_by_config.items():
        assert len(pair) == 2
        assert {case.force_on_policy_ratio for case, _ in pair} == {False, True}
        control = next(
            resolved for case, resolved in pair if not case.force_on_policy_ratio
        )
        treatment = next(
            resolved for case, resolved in pair if case.force_on_policy_ratio
        )
        assert normalized_pair_config(control) == normalized_pair_config(treatment)
        print(f"PAIR_OK {config_name}")


if __name__ == "__main__":
    main()
