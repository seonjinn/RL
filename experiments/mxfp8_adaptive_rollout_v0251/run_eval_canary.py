from __future__ import annotations

import argparse
import json
import os
import pprint
import time
from pathlib import Path

from omegaconf import OmegaConf

from examples.run_eval import setup_data
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
        "--arm", choices=("baseline", "trace", "adaptive"), required=True
    )
    args, remaining = parser.parse_known_args()

    start = time.time()
    print(f"NEMORL_CANARY arm={args.arm} event=start epoch={start}", flush=True)
    config = load_config(str(args.config))
    if remaining:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(remaining))
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
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
    run_env_eval(generation, dataloader, env, master_config)

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
