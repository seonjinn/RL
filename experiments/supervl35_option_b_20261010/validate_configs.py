"""Validate the composed SuperVL performance recipes against NeMo-RL."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.outdated_config_checks import check_outdated_config


def main() -> None:
    register_omegaconf_resolvers()
    config_dir = Path(sys.argv[1])
    configs = sorted(path for path in config_dir.glob("*.yaml") if path.stem != "te_precision")
    if len(configs) != 6:
        raise ValueError(f"Expected six performance recipes, found {len(configs)}")
    for path in configs:
        config = OmegaConf.to_container(load_config(path), resolve=True)
        check_outdated_config(config)
        MasterConfig(**config)
        if config["policy"]["generation"]["refit_transport"] != "nccl_reshard":
            raise ValueError(f"{path.name}: expected nccl_reshard refit transport")
        print(f"VALID {path.name}", flush=True)


if __name__ == "__main__":
    main()
