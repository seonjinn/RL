from __future__ import annotations

import argparse
import json
import os
import pprint
import time
from pathlib import Path

from omegaconf import OmegaConf

import nemo_rl.evals.eval as eval_module
from examples.run_eval import setup_data
from experiments.mxfp8_adaptive_rollout_v0251.generation_timing import (
    AsyncCallTimer,
    GenerationLengthAudit,
)
from experiments.mxfp8_adaptive_rollout_v0251.weight_source_guard import (
    require_valid_eval_weight_source,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets.eval_datasets import _is_multimodal_dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config


def _count_output_tokens(path: Path, tokenizer: object) -> int:
    data_path = path / "evaluation_data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    encode = getattr(tokenizer, "encode")
    return sum(
        len(encode(sample["response"], add_special_tokens=False))
        for sample in payload["evaluation_data"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("baseline", "trace", "trtllm_default", "adaptive"),
        required=True,
    )
    args, remaining = parser.parse_known_args()

    start = time.time()
    print(f"NEMORL_CANARY arm={args.arm} event=start epoch={start}", flush=True)
    config = load_config(str(args.config))
    if remaining:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(remaining))
    resolved_config = OmegaConf.to_container(config, resolve=True)
    require_valid_eval_weight_source(resolved_config)
    config = MasterConfig(**resolved_config)
    pprint.pprint(config)

    init_ray()
    is_multimodal = _is_multimodal_dataset(config.data["dataset_name"])
    tokenizer = get_tokenizer(config.tokenizer, get_processor=is_multimodal)
    config.generation = configure_generation_config(
        config.generation, tokenizer, is_eval=True
    )
    dataset, env, tokenizer = setup_data(
        tokenizer, config.data, config.env, is_multimodal=is_multimodal
    )
    generation, dataloader, master_config = setup(config, tokenizer, dataset)
    print(f"NEMORL_CANARY event=model_ready epoch={time.time()}", flush=True)
    generation_length_audit = GenerationLengthAudit()
    original_generate_text_async = generation.generate_text_async

    async def audited_generate_text_async(data, greedy=False):
        async for index, result in original_generate_text_async(data, greedy):
            lengths = result.get("generation_lengths")
            if lengths is None:
                raise RuntimeError("async text generation did not report token lengths")
            generation_length_audit.record(lengths.tolist())
            yield index, result

    generation.generate_text_async = audited_generate_text_async
    generation_timer = AsyncCallTimer()
    eval_module._generate_texts = generation_timer.wrap(  # noqa: SLF001
        eval_module._generate_texts  # noqa: SLF001
    )
    run_env_eval(generation, dataloader, env, master_config)
    print(
        "NEMORL_CANARY event=generation "
        f"seconds={generation_timer.elapsed_seconds} calls={generation_timer.calls}",
        flush=True,
    )
    print(
        "NEMORL_CANARY event=generated_outputs "
        f"requests={generation_length_audit.request_count} "
        f"min_tokens={generation_length_audit.min_tokens} "
        f"max_tokens={generation_length_audit.max_tokens} "
        f"tokens={generation_length_audit.total_tokens}",
        flush=True,
    )
    expected_requests = os.environ.get("CANARY_EXPECTED_REQUESTS")
    expected_tokens = os.environ.get("CANARY_EXPECTED_TOKENS_PER_RESPONSE")
    if (expected_requests is None) != (expected_tokens is None):
        raise RuntimeError(
            "CANARY_EXPECTED_REQUESTS and CANARY_EXPECTED_TOKENS_PER_RESPONSE "
            "must be set together"
        )
    if expected_requests is not None and expected_tokens is not None:
        generation_length_audit.validate(
            expected_requests=int(expected_requests),
            expected_tokens_per_response=int(expected_tokens),
        )

    output_dir = Path(os.environ["CANARY_OUTPUT_DIR"])
    output_tokens = _count_output_tokens(output_dir, tokenizer)
    if output_tokens <= 0:
        raise RuntimeError("canary produced no output tokens")
    print(f"NEMORL_CANARY event=outputs tokens={output_tokens}", flush=True)
    print(
        f"NEMORL_CANARY arm={args.arm} event=complete epoch={time.time()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
