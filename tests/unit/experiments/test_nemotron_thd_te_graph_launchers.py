from __future__ import annotations

import base64
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    REPO_ROOT / "experiments" / "cuda_graph" / "nemotron_thd_te_graph_20260731"
)
BRIDGE_SHA = "8e8156896abf194b99b0ac5a90bf449bd75c07eb"
MCORE_SHA = "2d19c0e07d2e8d6f061e05d55af1445bcef120a9"
NEMORL_SHA = "0" * 40
TE_SHA = "e" * 40
CONTAINER_SHA256 = "32f07be22293d9a3979e8ba04772ad48a8157dad04fd92577063ed4e07ab1493"
PYTHON_VERSION = (REPO_ROOT / ".python-version").read_text().strip()
UV_VERSION_MATCH = re.search(
    r"^ARG UV_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$",
    (REPO_ROOT / "docker" / "Dockerfile").read_text(),
    re.MULTILINE,
)
assert UV_VERSION_MATCH is not None
UV_VERSION = UV_VERSION_MATCH.group(1)
CONTAINER_ENV_VARS = (
    "CONTAINER_PATH_PREFIX,UV_PROJECT,UV_PROJECT_ENVIRONMENT,UV_LINK_MODE,UV_PYTHON,"
    "UV_PYTHON_INSTALL_DIR,UV_MANAGED_PYTHON,UV_PYTHON_DOWNLOADS,"
    "UV_NO_EDITABLE,PINNED_UV_VERSION,UV_EXECUTABLE,RUNTIME_PYTHON,NEMO_RL_VENV_DIR,"
    "NRL_FORCE_REBUILD_VENVS,NEMO_RL_MCORE_PY_EXECUTABLE,"
    "NEMO_RL_VLLM_PY_EXECUTABLE,"
    "NRL_MEGATRON_CHECKPOINT_DIR,NVTE_WITH_NCCL_EP,NVTE_CUDA_ARCHS,"
    "TORCH_CUDA_ARCH_LIST,CMAKE_BUILD_PARALLEL_LEVEL,NRL_SLURM_JOB_ID,"
    "NRL_SLURM_RESTART_COUNT,HF_HOME,HF_HUB_CACHE,HF_DATASETS_CACHE,HF_MODULES_CACHE,"
    "HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,HF_DATASETS_OFFLINE,"
    "HF_HUB_DISABLE_IMPLICIT_TOKEN,HF_HUB_DISABLE_TELEMETRY"
)
DENSE_AXES = ("attn", "mlp", "mamba")
MOE_AXES = (
    (),
    ("moe",),
    ("moe_router",),
    ("moe_router", "moe_preprocess"),
)
VALID_TE_SCOPES = {
    tuple(
        module
        for enabled, module in zip(enabled_dense, DENSE_AXES, strict=True)
        if enabled
    )
    + moe_scope
    for enabled_dense in itertools.product((False, True), repeat=3)
    for moe_scope in MOE_AXES
}


def _run_script(
    relative_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(EXPERIMENT_DIR / relative_path)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_clean_git_repository(tmp_path: Path, name: str) -> tuple[Path, str]:
    repository = tmp_path / name
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test")
    (repository / "tracked.txt").write_text(f"{name}\n")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-qm", f"create {name}")
    return repository, _git(repository, "rev-parse", "HEAD")


def _load_experiment_module(name: str) -> ModuleType:
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nemotron_experiment_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _load_r3_router_graph_parity_driver() -> ModuleType:
    return _load_experiment_module("scripts/run_r3_router_graph_parity")


def _r3_parity_array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_r3_parity_sidecar(
    frozen: Path,
    *,
    token_ids: np.ndarray,
    input_lengths: np.ndarray,
    routed_experts: np.ndarray,
) -> Path:
    schema = "nemo_rl_r3_frozen_batch_v1"
    json_sha256 = hashlib.sha256(frozen.read_bytes()).hexdigest()
    token_input_sha256 = _r3_parity_array_digest(
        {"input_lengths": input_lengths, "token_ids": token_ids}
    )
    route_sha256 = _r3_parity_array_digest(
        {"input_lengths": input_lengths, "routed_experts": routed_experts}
    )
    content_sha256 = hashlib.sha256(
        (
            f"{schema}\0{json_sha256}\0{token_input_sha256}\0{route_sha256}"
        ).encode()
    ).hexdigest()
    sidecar = frozen.with_suffix(".r3-parity.npz")
    np.savez_compressed(
        sidecar,
        schema=np.asarray(schema),
        json_sha256=np.asarray(json_sha256),
        token_input_sha256=np.asarray(token_input_sha256),
        route_sha256=np.asarray(route_sha256),
        content_sha256=np.asarray(content_sha256),
        token_ids=token_ids,
        input_lengths=input_lengths,
        routed_experts=routed_experts,
    )
    return sidecar


def test_r3_router_graph_parity_driver_loads_exact_json_and_safe_sidecar(
    tmp_path: Path,
) -> None:
    driver = _load_r3_router_graph_parity_driver()
    frozen = tmp_path / "train_data_step17.jsonl"
    rows = (
        {
            "idx": 0,
            "token_ids": [11, 12, 0],
            "input_lengths": 2,
            "token_loss_mask": [0.0, 1.0, 0.0],
            "sample_loss_mask": 1.0,
            "advantages": [0.0, 0.5, 0.0],
            "generation_logprobs": [0.0, -0.25, 0.0],
            "prev_logprobs": [0.0, -0.5, 0.0],
            "rewards": 1.0,
        },
        {
            "idx": 1,
            "token_ids": [21, 22, 23],
            "input_lengths": 3,
            "token_loss_mask": [0.0, 1.0, 1.0],
            "sample_loss_mask": 1.0,
            "advantages": [0.0, -0.5, 0.25],
            "generation_logprobs": [0.0, -0.75, -1.0],
            "prev_logprobs": [0.0, -0.5, -1.25],
            "rewards": -1.0,
        },
    )
    frozen.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecar = _write_r3_parity_sidecar(
        frozen,
        token_ids=np.asarray([[11, 12, 0], [21, 22, 23]], dtype=np.int64),
        input_lengths=np.asarray([2, 3], dtype=np.int64),
        routed_experts=np.asarray(
            [
                [[[0, 1]], [[2, 3]], [[0, 1]]],
                [[[1, 2]], [[2, 3]], [[3, 4]]],
            ],
            dtype=np.int16,
        ),
    )

    loaded = driver.load_frozen_batch(frozen)

    assert loaded.source_path == frozen.resolve()
    assert loaded.sidecar_path == sidecar.resolve()
    assert loaded.source_sha256 == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert loaded.row_count == 2
    assert loaded.batch["input_ids"].tolist() == [[11, 12, 0], [21, 22, 23]]
    assert loaded.batch["routed_experts"].tolist() == [
        [[[0, 1]], [[2, 3]], [[0, 1]]],
        [[[1, 2]], [[2, 3]], [[3, 4]]],
    ]
    assert loaded.batch["sample_mask"].tolist() == [1.0, 1.0]
    assert loaded.batch["rewards"].tolist() == [1.0, -1.0]


def test_r3_router_graph_parity_driver_rejects_missing_or_unbound_sidecar(
    tmp_path: Path,
) -> None:
    driver = _load_r3_router_graph_parity_driver()
    frozen = tmp_path / "train_data_step1.jsonl"
    frozen.write_text(
        json.dumps(
            {
                "idx": 0,
                "token_ids": [1, 2],
                "input_lengths": 2,
                "token_loss_mask": [0.0, 1.0],
                "sample_loss_mask": 1.0,
                "advantages": [0.0, 1.0],
                "generation_logprobs": [0.0, -1.0],
                "prev_logprobs": [0.0, -1.0],
                "rewards": 1.0,
            }
        )
        + "\n"
    )

    with pytest.raises(FileNotFoundError, match="sidecar"):
        driver.load_frozen_batch(frozen)

    sidecar = _write_r3_parity_sidecar(
        frozen,
        token_ids=np.asarray([[1, 2]], dtype=np.int64),
        input_lengths=np.asarray([2], dtype=np.int64),
        routed_experts=np.asarray([[[[0, 1]], [[1, 2]]]], dtype=np.int16),
    )
    frozen.write_text(frozen.read_text().replace("-1.0", "-2.0"))
    with pytest.raises(ValueError, match="JSON SHA256"):
        driver.load_frozen_batch(frozen)

    frozen.unlink()
    sidecar.unlink()


def test_r3_router_graph_parity_driver_requires_typed_runtime_attestation(
    tmp_path: Path,
) -> None:
    driver = _load_r3_router_graph_parity_driver()
    attestation = tmp_path / "runtime.json"
    payload = {
        "runtime_feature_set": "dropless_hybridep_nano16_r3_router_graph_v1",
        "mcore_capabilities": {
            "router_replay_cuda_graph_input": "r3_router_cuda_graph_input_v1"
        },
    }
    attestation.write_text(json.dumps(payload))

    loaded, digest = driver.load_runtime_attestation(attestation)

    assert loaded == payload
    assert digest == hashlib.sha256(attestation.read_bytes()).hexdigest()
    attestation.write_text(json.dumps({"runtime_feature_set": "unbound"}))
    with pytest.raises(ValueError, match="runtime feature"):
        driver.load_runtime_attestation(attestation)


def _r3_router_graph_parity_rank_result(rank: int, arm: str) -> dict[str, object]:
    def tensor_evidence(*, sha256: str, values: list[float]) -> dict[str, object]:
        return {
            "sha256": sha256,
            "shape": [2],
            "dtype": "torch.float32",
            "numel": 2,
            "l2_norm": 2.2360679775,
            "max_abs": 2.0,
            "mean": 1.5,
            "values": values,
        }

    return {
        "rank": rank,
        "arm": arm,
        "token_digest": "c" * 64,
        "route_digest": "d" * 64,
        "mask_digest": "e" * 64,
        "reward_digest": "f" * 64,
        "loss": 1.25,
        "selected_output": tensor_evidence(sha256="a" * 64, values=[1.0, 2.0]),
        "selected_output_gradient": tensor_evidence(
            sha256="a" * 64, values=[1.0, 2.0]
        ),
        "selected_input_gradient": tensor_evidence(
            sha256="a" * 64, values=[1.0, 2.0]
        ),
        "parameter_gradients": {
            "decoder.weight": tensor_evidence(
                sha256="a" * 64, values=[1.0, 2.0]
            )
        },
        "simulated_parameter_deltas": {
            "decoder.weight": tensor_evidence(
                sha256="b" * 64, values=[-0.1, -0.2]
            )
        },
        "metrics": {
            "token_mult_prob_error": 1.0,
            "policy_kl": 0.125,
            "generation_kl": 0.25,
        },
        "graph_metrics": {
            "requested_graph_calls": 1 if arm == "graph" else 0,
            "graph_calls": 1 if arm == "graph" else 0,
            "cache_hits": 1 if arm == "graph" else 0,
            "captures": 1 if arm == "graph" else 0,
            "recaptures": 0,
            "fallback_count": 0,
            "unsafe_route_events": 0,
        },
    }


def test_r3_router_graph_parity_driver_checks_every_rank_and_parameter() -> None:
    driver = _load_r3_router_graph_parity_driver()
    eager = [_r3_router_graph_parity_rank_result(rank, "eager") for rank in range(16)]
    graph = [_r3_router_graph_parity_rank_result(rank, "graph") for rank in range(16)]

    comparison = driver.validate_parity(eager, graph, rtol=0.05, atol=0.05)

    assert comparison["status"] == "passed"
    assert comparison["world_size"] == 16
    assert comparison["compared_parameter_gradients"] == 16
    graph[7]["route_digest"] = "0" * 64
    with pytest.raises(ValueError, match="route_digest.*rank 7"):
        driver.validate_parity(eager, graph, rtol=0.05, atol=0.05)
    graph[7]["route_digest"] = "d" * 64
    graph[9]["parameter_gradients"]["decoder.weight"]["values"][1] = 4.0
    with pytest.raises(ValueError, match="decoder.weight.*rank 9"):
        driver.validate_parity(eager, graph, rtol=0.05, atol=0.05)


def test_r3_router_graph_parity_driver_rejects_incomplete_rank_evidence() -> None:
    driver = _load_r3_router_graph_parity_driver()
    eager = [_r3_router_graph_parity_rank_result(rank, "eager") for rank in range(16)]
    graph = [_r3_router_graph_parity_rank_result(rank, "graph") for rank in range(16)]

    eager[0].pop("selected_input_gradient")
    graph[0].pop("selected_input_gradient")
    with pytest.raises(ValueError, match="selected_input_gradient.*rank 0"):
        driver.validate_parity(eager, graph)

    eager[0]["selected_input_gradient"] = _r3_router_graph_parity_rank_result(
        0, "eager"
    )["selected_input_gradient"]
    graph[0]["selected_input_gradient"] = _r3_router_graph_parity_rank_result(
        0, "graph"
    )["selected_input_gradient"]
    eager.append(eager[0])
    graph.append(graph[0])
    with pytest.raises(ValueError, match="exactly one.*all 16 ranks"):
        driver.validate_parity(eager, graph)


def test_r3_router_graph_parity_driver_writes_one_immutable_artifact(
    tmp_path: Path,
) -> None:
    driver = _load_r3_router_graph_parity_driver()
    artifact = tmp_path / "parity.json"

    driver.write_immutable_json(artifact, {"status": "passed", "world_size": 16})

    assert json.loads(artifact.read_text()) == {
        "status": "passed",
        "world_size": 16,
    }
    assert artifact.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="already exists"):
        driver.write_immutable_json(artifact, {"status": "replaced"})


def _r3_router_graph_parity_launcher_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, str]]:
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "runtime_feature_set": (
                    "dropless_hybridep_nano16_r3_router_graph_v1"
                ),
                "mcore_capabilities": {
                    "router_replay_cuda_graph_input": (
                        "r3_router_cuda_graph_input_v1"
                    )
                },
            }
        )
    )
    container = tmp_path / "runtime.sqsh"
    container.write_bytes(b"immutable-runtime")
    runtime_stage = tmp_path / "staged-runtimes" / ("a" * 64)
    runtime_python = runtime_stage / "environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\nexit 0\n")
    runtime_python.chmod(0o755)
    vllm_python = runtime_stage / "vllm-environment" / "bin" / "python"
    vllm_python.parent.mkdir(parents=True)
    vllm_python.write_text("#!/bin/sh\nexit 0\n")
    vllm_python.chmod(0o755)
    uv = runtime_stage / "uv" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    fake_bin = tmp_path / "profile-bin"
    fake_bin.mkdir()
    verifier_calls = tmp_path / "runtime-verifier-calls.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == *verify_runtime_attestation.py ]]; then\n'
        '  printf "%s\\n" "$*" >>"${FAKE_VERIFIER_CALLS:?}"\n'
        "  exit 0\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    fake_python.chmod(0o755)
    frozen = tmp_path / "train_data_step17.jsonl"
    frozen.write_text('{"fixture":"validated by driver"}\n')
    artifacts = tmp_path / "artifacts"
    profile = tmp_path / "oci-hsg.env"
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=oci-hsg",
                "ACCOUNT=unit-account",
                "PARTITION=batch",
                f"CONTAINER={container}",
                f"CONTAINER_SHA256={hashlib.sha256(container.read_bytes()).hexdigest()}",
                f"MOUNTS={tmp_path}:{tmp_path}",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=gpu:4",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=00:30:00",
                f"RUNTIME_ATTESTATION={runtime}",
                "RUNTIME_PREFLIGHT_JOB_ID=123",
                f"UV_EXECUTABLE={uv}",
                f"EXPECTED_TE_SHA={'e' * 40}",
                f"EXPECTED_TE_VERSION_BASE_SHA={'f' * 40}",
                f"EXPECTED_NEMORL_SHA={_git(REPO_ROOT, 'rev-parse', 'HEAD')}",
                f"EXPECTED_BRIDGE_SHA={'b' * 40}",
                f"EXPECTED_MCORE_SHA={'c' * 40}",
                f"RUN_LOG_ROOT={tmp_path / 'logs'}",
                "",
            )
        )
    )
    environment = {
        "PROJECT_ROOT": str(REPO_ROOT),
        "PROFILE_FILE": str(profile),
        "FROZEN_BATCH": str(frozen),
        "ARTIFACT_DIR": str(artifacts),
        "CONFIG": str(
            REPO_ROOT
            / "examples/configs/recipes/llm/"
            "grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml"
        ),
        "HF_HOME": str(hf_home),
        "RUNTIME_PYTHON": str(runtime_python),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_VERIFIER_CALLS": str(verifier_calls),
    }
    return profile, environment


def test_r3_router_graph_parity_launcher_test_modes_render_identical_payload(
    tmp_path: Path,
) -> None:
    _, environment = _r3_router_graph_parity_launcher_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation = tmp_path / "sbatch.txt"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        'printf "ENV_SBATCH_GPUS=%s\\n" "${SBATCH_GPUS-unset}" >"${FAKE_INVOCATION:?}"\n'
        'printf "ENV_NRL_ROUTER_REPLAY_VALIDATE=%s\\n" "${NRL_ROUTER_REPLAY_VALIDATE-unset}" >>"${FAKE_INVOCATION:?}"\n'
        'printf "%s\\n" "$@" >>"${FAKE_INVOCATION:?}"\n'
        "printf 'sbatch: Job 123 would be submitted\\n'\n"
    )
    sbatch.chmod(0o755)
    ambient = {
        "SBATCH_GPUS": "99",
        "NRL_ROUTER_REPLAY_VALIDATE": "0",
        "NRL_R3_PARITY_ARM": "ambient-invalid",
    }

    test_only = _run_script(
        "diagnostics/submit_r3_router_graph_parity.sh",
        **environment,
        **ambient,
        TEST_ONLY="1",
        SBATCH_TEST_ONLY="0",
    )
    scheduler_environment = {
        **environment,
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "FAKE_INVOCATION": str(invocation),
    }
    scheduler_only = _run_script(
        "diagnostics/submit_r3_router_graph_parity.sh",
        **scheduler_environment,
        **ambient,
        TEST_ONLY="0",
        SBATCH_TEST_ONLY="1",
    )

    assert test_only.returncode == 0, test_only.stderr
    assert scheduler_only.returncode == 0, scheduler_only.stderr
    test_payload = next(
        line for line in test_only.stdout.splitlines() if line.startswith("PAYLOAD: ")
    )
    scheduler_payload = next(
        line
        for line in scheduler_only.stdout.splitlines()
        if line.startswith("PAYLOAD: ")
    )
    assert test_payload == scheduler_payload
    payload = shlex.split(test_payload.removeprefix("PAYLOAD: "))
    assert "--nodes=4" in payload
    assert "--gres=gpu:4" in payload
    assert "--export=NONE" in payload
    assert not any(argument.startswith("--dependency") for argument in payload)
    rendered = " ".join(payload)
    for exact in (
        "dropless_hybridep_nano16_r3_router_graph_v1",
        "r3_router_cuda_graph_input_v1",
        "NRL_ROUTER_REPLAY_VALIDATE=1",
        "++policy.router_replay.enabled=true",
        "++policy.generation.vllm_kwargs.moe_backend=triton",
        "policy.precision=bfloat16",
        "policy.megatron_cfg.moe_token_dispatcher_type=flex",
        "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep",
        "++policy.megatron_cfg.thd_max_packed_sequences=16",
        "++policy.megatron_cfg.cuda_graph_modules=[moe_router]",
        "cluster.num_nodes=4",
        "cluster.gpus_per_node=4",
    ):
        assert exact in rendered
    assert "TEST_ONLY: no submission performed" in test_only.stdout
    invocation_lines = invocation.read_text().splitlines()
    assert "ENV_SBATCH_GPUS=unset" in invocation_lines
    assert "ENV_NRL_ROUTER_REPLAY_VALIDATE=unset" in invocation_lines
    assert "--test-only" in invocation_lines
    assert not Path(environment["ARTIFACT_DIR"]).exists()
    verifier_calls = Path(environment["FAKE_VERIFIER_CALLS"]).read_text().splitlines()
    assert len(verifier_calls) == 2
    assert all("verify_runtime_attestation.py" in call for call in verifier_calls)


def test_r3_router_graph_parity_launcher_delegates_to_multinode_ray_bootstrap() -> None:
    source = (
        EXPERIMENT_DIR / "diagnostics/submit_r3_router_graph_parity.sh"
    ).read_text()

    assert "scripts/run_nemorl_scope.sub" in source
    assert "ray.sub" in (
        EXPERIMENT_DIR / "scripts/run_nemorl_scope.sub"
    ).read_text()
    assert "--ntasks=1" not in source
    assert "srun" not in source
    assert "RUNTIME_PYTHON=${runtime_stage_root}/environment/bin/python" not in source
    assert "canonical_runtime_python=${runtime_stage_root}/environment/bin/python" in source


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("source", "source SHA mismatch"),
        ("profile", "exactly 4 GPUs per node"),
        ("attestation", "runtime feature"),
    ),
)
def test_r3_router_graph_parity_launcher_fails_closed_before_payload(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    profile, environment = _r3_router_graph_parity_launcher_fixture(tmp_path)
    if mutation == "source":
        profile.write_text(
            profile.read_text().replace(
                f"EXPECTED_NEMORL_SHA={_git(REPO_ROOT, 'rev-parse', 'HEAD')}",
                f"EXPECTED_NEMORL_SHA={'0' * 40}",
            )
        )
    elif mutation == "profile":
        profile.write_text(
            profile.read_text().replace(
                "SBATCH_GPUS_PER_NODE=4", "SBATCH_GPUS_PER_NODE=8"
            )
        )
    else:
        Path(profile.parent / "runtime.json").write_text(
            json.dumps(
                {
                    "runtime_feature_set": "dropless_hybridep_nano16",
                    "mcore_capabilities": {
                        "router_replay_cuda_graph_input": (
                            "r3_router_cuda_graph_input_v1"
                        )
                    },
                }
            )
        )

    result = _run_script(
        "diagnostics/submit_r3_router_graph_parity.sh",
        **environment,
        TEST_ONLY="1",
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "PAYLOAD:" not in result.stdout
    assert "SBATCH:" not in result.stdout


def test_r3_router_graph_mcore_row_renders_typed_distributed_commands() -> None:
    runner = _load_experiment_module("scripts/run_mcore_training")
    rows = runner.load_matrix(
        EXPERIMENT_DIR / "mcore_test_matrix.json", candidate_kind="mcore"
    )

    row = rows["dropless_hybridep_nano16_r3_router_graph"]
    commands = runner.pytest_commands(row, python_executable=Path("/runtime/python"))

    assert row.world_size == 16
    assert row.allocations == ((4, 4),)
    assert commands == (
        (
            "/runtime/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py::"
            "test_dropless_hybridep_nano16_r3_router_graph",
        ),
        (
            "/runtime/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit_tests/transformer/test_partial_moe_cuda_graph_distributed.py::"
            "test_dropless_hybridep_nano16_r3_router_graph_rejects_invalid_route",
        ),
    )
    rendered = " ".join(shlex.join(command) for command in commands)
    assert "MCORE_TEST_DISABLE_NANO_SHARED_EXPERT" not in rendered
    assert "MCORE_TEST_" not in rendered


def _create_staged_runtime(
    tmp_path: Path,
    runtime_python_source: str = "#!/bin/bash\nexit 0\n",
) -> tuple[Path, Path]:
    stage = tmp_path / "staged-runtimes" / ("a" * 64)
    uv = stage / "uv" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/bash\nexit 0\n")
    uv.chmod(0o755)
    runtime_python = stage / "environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text(runtime_python_source)
    runtime_python.chmod(0o755)
    return uv, runtime_python


def _extract_shell_function(relative_path: str, function_name: str) -> str:
    source = (EXPERIMENT_DIR / relative_path).read_text()
    match = re.search(
        rf"(?ms)^{re.escape(function_name)}\(\) \{{\n.*?^\}}\n",
        source,
    )
    assert match is not None, f"missing shell function: {function_name}"
    return match.group(0)


def _campaign_leaf_harness(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    root = tmp_path / "repo"
    experiment = root / "experiments" / "cuda_graph" / EXPERIMENT_DIR.name
    shutil.copytree(
        EXPERIMENT_DIR, experiment, ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "docker").mkdir(parents=True)
    (root / "tools").mkdir()
    shutil.copy2(REPO_ROOT / ".python-version", root / ".python-version")
    shutil.copy2(REPO_ROOT / "docker" / "Dockerfile", root / "docker" / "Dockerfile")
    shutil.copy2(
        REPO_ROOT / "tools" / "check_r3_trace.py", root / "tools" / "check_r3_trace.py"
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    runtime_stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    staged_uv = runtime_stage_root / "uv" / "uv"
    staged_uv.parent.mkdir(parents=True)
    staged_uv.write_text(f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n")
    staged_uv.chmod(0o755)
    managed_python = (
        tmp_path
        / "uv-python-installations"
        / f"cpython-{PYTHON_VERSION}-fixture"
        / "bin"
        / "python3.13"
    )
    managed_python.parent.mkdir(parents=True)
    managed_python.write_text("#!/bin/sh\nexit 0\n")
    managed_python.chmod(0o755)
    runtime_python = runtime_stage_root / "environment" / "bin" / "python"
    vllm_runtime_python = runtime_stage_root / "vllm-environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    vllm_runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(managed_python)
    vllm_runtime_python.symlink_to(managed_python)
    provenance = {
        "nemo_rl_commit": "1" * 40,
        "bridge_commit": "2" * 40,
        "mcore_commit": "3" * 40,
        "container_sha256": "4" * 64,
        "runtime_attestation_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
    }
    profile = experiment / "profiles" / "oci-hsg.env"
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=unit",
                "ACCOUNT=unit",
                "PARTITION=batch",
                "CONTAINER=/tmp/container.sqsh",
                f"CONTAINER_SHA256={provenance['container_sha256']}",
                f"MOUNTS={tmp_path}:{tmp_path}",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=gpu:4",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=01:00:00",
                f"RUNTIME_ATTESTATION={runtime}",
                "RUNTIME_PREFLIGHT_JOB_ID=1",
                f"UV_EXECUTABLE={staged_uv}",
                f"EXPECTED_TE_SHA={'e' * 40}",
                f"EXPECTED_TE_VERSION_BASE_SHA={'f' * 40}",
                f"EXPECTED_NEMORL_SHA={provenance['nemo_rl_commit']}",
                f"EXPECTED_BRIDGE_SHA={provenance['bridge_commit']}",
                f"EXPECTED_MCORE_SHA={provenance['mcore_commit']}",
                "",
            )
        )
    )
    return root, experiment, profile, provenance


def _write_campaign_gate(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_campaign_leaf(
    root: Path, experiment: Path, leaf: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CLUSTER": "oci-hsg",
            "MODE": "nemorl",
            "TEST_ONLY": "1",
            "RUN_TAG": "unit",
            **environment,
        }
    )
    return subprocess.run(
        ["bash", str(experiment / "conditions" / leaf)],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_copied_experiment_script(
    root: Path, experiment: Path, relative_path: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(experiment / relative_path)],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _validate_campaign_gate(
    root: Path,
    experiment: Path,
    kind: str,
    gate_file: Path,
    gate_sha256: str,
    model: str,
    profile: Path,
    arm: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        "python3",
        str(experiment / "validate_campaign_gate.py"),
        kind,
        "--gate-file",
        str(gate_file),
        "--gate-sha256",
        gate_sha256,
        "--model",
        model,
        "--profile-file",
        str(profile),
        "--profile-dir",
        str(experiment / "profiles"),
        "--cluster",
        "oci-hsg",
    ]
    if arm is not None:
        command.extend(("--arm", arm))
    return subprocess.run(
        command, cwd=root, check=False, capture_output=True, text=True
    )


def test_direct_qwen235_r3_leaf_rejects_self_attested_evidence(
    tmp_path: Path,
) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    r3 = {
        "gate_type": "qwen235_r3_routes",
        "status": "passed",
        "model": "qwen3_235b",
        "slurm_job_id": 1,
        "provenance": provenance,
        "diagnostic": {
            "model": "Qwen/Qwen3-235B-A22B",
            "num_prompts": 128,
            "max_tokens": 256,
            "max_model_len": 8192,
            "prompt_repeat": 128,
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enforce_eager": False,
            "moe_backend": "triton",
            "num_outputs": 128,
            "num_failures": 0,
        },
    }
    r3_file = tmp_path / "r3.json"
    r3_sha = _write_campaign_gate(r3_file, r3)
    result = _run_campaign_leaf(
        root,
        experiment,
        "qwen_C_baseline_r3on.sh",
        MODEL="qwen3_235b",
        PROFILE_FILE=str(profile),
        R3_PREFLIGHT_FILE=str(r3_file),
        R3_PREFLIGHT_SHA256=r3_sha,
    )
    assert result.returncode == 2
    assert "content-bound Slurm diagnostic producer" in result.stderr
    assert "SBATCH:" not in result.stdout
    missing = _run_campaign_leaf(
        root,
        experiment,
        "qwen_C_baseline_r3on.sh",
        MODEL="qwen3_235b",
        PROFILE_FILE=str(profile),
    )
    assert missing.returncode == 2
    assert "SBATCH:" not in missing.stdout


def test_direct_qwen30_performance_requires_and_accepts_promotion_without_digest(
    tmp_path: Path,
) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    promotion = {
        "gate_type": "smoke_promotion",
        "status": "passed",
        "model": "qwen3_30ba3b",
        "phase": "smoke",
        "steps": 5,
        "provenance": provenance,
        "arms": {
            "A": {
                "job_id": 1,
                "status": "passed",
                "completed_steps": 5,
                "metrics_finite": True,
                "correctness_passed": True,
                "undeclared_fallbacks": 0,
                "router_replay": "off",
                "graph_coverage_status": "not_applicable",
                "r3_trace_status": "not_applicable",
            }
        },
    }
    gate = tmp_path / "promotion.json"
    digest = _write_campaign_gate(gate, promotion)
    result = _run_campaign_leaf(
        root,
        experiment,
        "qwen_A_baseline_r3off.sh",
        MODEL="qwen3_30ba3b",
        STEPS="20",
        PROFILE_FILE=str(profile),
        SMOKE_PROMOTION_FILE=str(gate),
        SMOKE_PROMOTION_SHA256=digest,
    )
    assert result.returncode == 0, result.stderr
    assert "SBATCH:" in result.stdout
    missing = _run_campaign_leaf(
        root,
        experiment,
        "qwen_A_baseline_r3off.sh",
        MODEL="qwen3_30ba3b",
        STEPS="20",
        PROFILE_FILE=str(profile),
    )
    assert missing.returncode == 2
    assert "SBATCH:" not in missing.stdout
    mismatched = _run_campaign_leaf(
        root,
        experiment,
        "qwen_A_baseline_r3off.sh",
        MODEL="qwen3_30ba3b",
        PROFILE_FILE=str(profile),
        VALIDATED_PROFILE_SHA256="0" * 64,
    )
    assert mismatched.returncode == 2
    assert "SBATCH:" not in mismatched.stdout


def test_direct_qwen235_performance_rejects_self_attested_r3_gate(
    tmp_path: Path,
) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    r3 = {
        "gate_type": "qwen235_r3_routes",
        "status": "passed",
        "model": "qwen3_235b",
        "slurm_job_id": 1,
        "provenance": provenance,
        "diagnostic": {
            "model": "Qwen/Qwen3-235B-A22B",
            "num_prompts": 128,
            "max_tokens": 256,
            "max_model_len": 8192,
            "prompt_repeat": 128,
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enforce_eager": False,
            "moe_backend": "triton",
            "num_outputs": 128,
            "num_failures": 0,
        },
    }
    promotion = {
        "gate_type": "smoke_promotion",
        "status": "passed",
        "model": "qwen3_235b",
        "phase": "smoke",
        "steps": 5,
        "provenance": provenance,
        "arms": {
            "C": {
                "job_id": 1,
                "status": "passed",
                "completed_steps": 5,
                "metrics_finite": True,
                "correctness_passed": True,
                "undeclared_fallbacks": 0,
                "router_replay": "on",
                "graph_coverage_status": "not_applicable",
                "r3_trace_status": "passed",
            }
        },
    }
    r3_file = tmp_path / "r3.json"
    promotion_file = tmp_path / "promotion.json"
    r3_sha = _write_campaign_gate(r3_file, r3)
    promotion_sha = _write_campaign_gate(promotion_file, promotion)
    assert (
        _validate_campaign_gate(
            root,
            experiment,
            "promotion",
            promotion_file,
            promotion_sha,
            "qwen3_235b",
            profile,
            arm="C",
        ).returncode
        == 0
    )

    result = _run_campaign_leaf(
        root,
        experiment,
        "qwen_C_baseline_r3on.sh",
        MODEL="qwen3_235b",
        STEPS="20",
        PROFILE_FILE=str(profile),
        R3_PREFLIGHT_FILE=str(r3_file),
        R3_PREFLIGHT_SHA256=r3_sha,
        SMOKE_PROMOTION_FILE=str(promotion_file),
        SMOKE_PROMOTION_SHA256=promotion_sha,
    )

    assert result.returncode == 2
    assert "content-bound Slurm diagnostic producer" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_direct_qwen30_smoke_accepts_explicit_profile_file(tmp_path: Path) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)

    result = _run_campaign_leaf(
        root,
        experiment,
        "qwen_A_baseline_r3off.sh",
        MODEL="qwen3_30ba3b",
        PROFILE_FILE=str(profile),
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH:" in result.stdout


def _create_bridge_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    mcore = tmp_path / "mcore-source"
    mcore.mkdir()
    _git(mcore, "init", "-q")
    _git(mcore, "config", "user.email", "test@example.com")
    _git(mcore, "config", "user.name", "Test")
    (mcore / "README.md").write_text("fixture\n")
    _git(mcore, "add", "README.md")
    _git(mcore, "commit", "-qm", "fixture mcore")
    mcore_sha = _git(mcore, "rev-parse", "HEAD")

    bridge = tmp_path / "bridge-source"
    bridge.mkdir()
    _git(bridge, "init", "-q")
    _git(bridge, "config", "user.email", "test@example.com")
    _git(bridge, "config", "user.name", "Test")
    (bridge / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    (bridge / "uv.lock").write_text("committed-lock\n")
    for name in ("nano", "super", "ultra"):
        recipe_test = (
            bridge
            / "tests"
            / "unit_tests"
            / "recipes"
            / "nemotronh"
            / f"test_nemotron_3_{name}.py"
        )
        recipe_test.parent.mkdir(parents=True, exist_ok=True)
        recipe_test.write_text(f"def test_{name}():\n    assert True\n")
    _git(bridge, "add", ".")
    _git(bridge, "commit", "-qm", "fixture bridge")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(bridge),
            "submodule",
            "add",
            "-q",
            str(mcore),
            "3rdparty/Megatron-LM",
        ],
        check=True,
    )
    _git(bridge, "commit", "-qam", "pin fixture mcore")
    return bridge, _git(bridge, "rev-parse", "HEAD"), mcore_sha


def test_oci_bridge_bootstrap_test_only_renders_reproducible_batch_submission(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_bridge_bootstrap.sub",
        TEST_ONLY="1",
        BRIDGE_REPOSITORY="git@github.com:seonjinn/Megatron-Bridge.git",
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--account=coreai_dlalgo_nemorl" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert f"EXPECTED_BRIDGE_SHA={BRIDGE_SHA}" in result.stdout
    assert f"EXPECTED_MCORE_SHA={MCORE_SHA}" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_oci_bridge_bootstrap_rejects_credential_bearing_or_ambiguous_remote(
    tmp_path: Path,
) -> None:
    invalid_repositories = (
        "https://user:placeholder@github.com/org/Megatron-Bridge.git",
        "https://github.com/org/Megatron-Bridge.git?token=placeholder",
        "https://github.com/org/Megatron-Bridge.git#fragment",
        "http://github.com/org/Megatron-Bridge.git",
        "ssh://git@github.com/org/Megatron-Bridge.git",
        "git@github.com:org/Megatron Bridge.git",
        "git@github.com:org/Megatron-Bridge.git\nsecond-line",
    )
    for repository in invalid_repositories:
        result = _run_script(
            "scripts/validate_oci_bridge_bootstrap.sub",
            TEST_ONLY="1",
            BRIDGE_REPOSITORY=repository,
            EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
            EXPECTED_MCORE_SHA=MCORE_SHA,
            ARTIFACT_DIR=str(tmp_path / "artifacts"),
            CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
            CONTAINER_SHA256=CONTAINER_SHA256,
        )

        assert result.returncode == 2
        assert "credential-free public HTTPS or git@host:path remote" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert repository not in result.stdout
        assert repository not in result.stderr


def test_oci_bridge_bootstrap_has_no_singleton_dependency() -> None:
    result = _run_script(
        "scripts/validate_oci_bridge_bootstrap.sub",
        TEST_ONLY="1",
        BRIDGE_REPOSITORY="git@github.com:seonjinn/Megatron-Bridge.git",
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR="/lustre/example/bridge-bootstrap",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
    )

    assert result.returncode == 0, result.stderr
    assert "dependency" not in result.stdout.lower()


def test_oci_bridge_bootstrap_uses_persistent_payload_when_wrapper_is_spooled(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_bridge_bootstrap.sub"
    persistent_payload = (
        EXPERIMENT_DIR / "scripts" / "bridge_bootstrap_payload.sh"
    ).resolve()
    spool_dir = tmp_path / "slurm-spool" / "job314"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    digest = hashlib.sha256(container.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.txt"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('#!/bin/bash\nprintf \'%s\\n\' "$*" >"${SRUN_LOG}"\n')
    fake_srun.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "314",
            "BRIDGE_BOOTSTRAP_PAYLOAD": str(persistent_payload),
            "BRIDGE_REPOSITORY": "git@github.com:seonjinn/Megatron-Bridge.git",
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "CONTAINER": str(container),
            "CONTAINER_SHA256": digest,
        }
    )

    result = subprocess.run(
        ["bash", str(spooled_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(persistent_payload) in srun_log.read_text()
    assert str(spool_dir / "bridge_bootstrap_payload.sh") not in srun_log.read_text()


def test_bridge_bootstrap_payload_relocks_and_runs_three_recipe_files(
    tmp_path: Path,
) -> None:
    bridge, bridge_sha, mcore_sha = _create_bridge_fixture(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_call_log = tmp_path / "uv-calls.txt"
    python_call_log = tmp_path / "python-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s|%s\\n' "${FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD:-}" "$*" >>"${UV_CALL_LOG}"
if [[ "$1" == "lock" ]]; then
  printf 'resolved-lock\\n' >uv.lock
  exit 0
fi
exit 99
"""
    )
    fake_uv.chmod(0o755)
    fake_python = fake_bin / "container-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s|%s\\n' "${PYTHONPATH:-}" "$*" >>"${PYTHON_CALL_LOG}"
if [[ "$1" == "--version" ]]; then
  printf 'Python 3.12.9\\n'
  exit 0
fi
if [[ "$1" == "-c" ]]; then
  exit 0
fi
for argument in "$@"; do
  case "${argument}" in
    --junitxml=*)
      junit=${argument#--junitxml=}
      mkdir -p "$(dirname "${junit}")"
      printf '<testsuite tests="3" failures="0"/>\\n' >"${junit}"
      ;;
  esac
done
printf '3 passed\\n'
"""
    )
    fake_python.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    work_parent = tmp_path / "work"
    work_parent.mkdir()
    sentinel = work_parent / "caller-owned.txt"
    sentinel.write_text("preserve\n")

    result = _run_script(
        "scripts/bridge_bootstrap_payload.sh",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        GIT_ALLOW_PROTOCOL="file",
        UV_CALL_LOG=str(uv_call_log),
        PYTHON_CALL_LOG=str(python_call_log),
        BRIDGE_REPOSITORY=str(bridge),
        EXPECTED_BRIDGE_SHA=bridge_sha,
        EXPECTED_MCORE_SHA=mcore_sha,
        ARTIFACT_DIR=str(artifacts),
        CONTAINER="/lustre/example/nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        SLURM_JOB_ID="314",
        WORK_ROOT=str(work_parent),
        LOCK_PYTHON="/usr/bin/python3.12",
        CONTAINER_PYTHON=str(fake_python),
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text() == "preserve\n"
    assert list(work_parent.glob("bridge-bootstrap-314.*")) == []
    result_dir = artifacts / f"bridge-{bridge_sha}-314"
    assert (result_dir / "uv.lock").read_text() == "resolved-lock\n"
    assert (result_dir / "status.txt").read_text() == "passed\n"
    assert (result_dir / "recipe-tests.junit.xml").is_file()
    calls = uv_call_log.read_text().splitlines()
    assert calls == [
        "TRUE|lock --no-build-isolation-package fast-hadamard-transform "
        "--python /usr/bin/python3.12 --no-python-downloads"
    ]
    python_calls = python_call_log.read_text().splitlines()
    assert python_calls[0].endswith("|--version")
    assert "|-c import megatron.bridge" in python_calls[1]
    assert "transformer_engine" in python_calls[1]
    assert "|-m pytest -q" in python_calls[2]
    assert "test_nemotron_3_nano.py" in python_calls[2]
    assert "test_nemotron_3_super.py" in python_calls[2]
    assert "test_nemotron_3_ultra.py" in python_calls[2]
    assert "/Megatron-Bridge/src:" in python_calls[1]
    assert "/Megatron-Bridge/3rdparty/Megatron-LM" in python_calls[1]
    provenance = (result_dir / "provenance.env").read_text()
    assert f"bridge_sha={bridge_sha}" in provenance
    assert f"mcore_sha={mcore_sha}" in provenance
    assert "uv_lock_sha256=" in provenance
    assert "lock_python=/usr/bin/python3.12" in provenance
    assert f"container_python={fake_python}" in provenance
    assert "container_python_version=Python 3.12.9" in provenance


def test_bridge_bootstrap_payload_rejects_credential_bearing_remote(
    tmp_path: Path,
) -> None:
    repository = "https://user:placeholder@github.com/org/Megatron-Bridge.git"
    result = _run_script(
        "scripts/bridge_bootstrap_payload.sh",
        BRIDGE_REPOSITORY=repository,
        EXPECTED_BRIDGE_SHA=BRIDGE_SHA,
        EXPECTED_MCORE_SHA=MCORE_SHA,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        CONTAINER="/lustre/example/nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        WORK_ROOT=str(tmp_path / "work"),
    )

    assert result.returncode == 2
    assert (
        "BRIDGE_REPOSITORY is not an approved credential-free source" in result.stderr
    )
    assert repository not in result.stderr
    assert not (tmp_path / "artifacts").exists()


def test_stage_enroot_image_test_only_renders_immutable_batch_submission(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidia/nemo:nightly",
        SOURCE_DIGEST=digest,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_20260731",
        CONTAINER_DIR=str(tmp_path / "containers"),
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--cpus-per-task" not in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout
    assert f"SOURCE_DIGEST={digest}" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "containers").exists()


def test_stage_enroot_image_uses_no_gpu_for_ptyche(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidia/nemo:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_ptyche",
        CONTAINER_DIR=str(tmp_path / "containers"),
        ACCOUNT="coreai_dlalgo_llm",
    )

    assert result.returncode == 0, result.stderr
    assert "--account=coreai_dlalgo_llm" in result.stdout
    assert "--cpus-per-task" not in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout


def test_stage_enroot_image_allows_cpu_datamover_without_gpu(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidian/nemo-rl:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_oci",
        CONTAINER_DIR=str(tmp_path / "containers"),
        PARTITION="cpu_datamover",
    )

    assert result.returncode == 0, result.stderr
    assert "--partition=cpu_datamover" in result.stdout
    assert "--cpus-per-task=32" in result.stdout
    assert "--time=04:00:00" in result.stdout
    assert "STAGE_CPUS_PER_TASK\\,STAGE_TIME_LIMIT" in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout


def test_stage_enroot_image_rejects_invalid_cpu_staging_width(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidian/nemo-rl:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_oci",
        CONTAINER_DIR=str(tmp_path / "containers"),
        PARTITION="cpu_datamover",
        STAGE_CPUS_PER_TASK="0",
    )

    assert result.returncode == 2
    assert "STAGE_CPUS_PER_TASK must be an integer from 1 through 96" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_stage_enroot_image_rejects_unapproved_partition(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidian/nemo-rl:nightly",
        SOURCE_DIGEST="sha256:" + "a" * 64,
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_oci",
        CONTAINER_DIR=str(tmp_path / "containers"),
        PARTITION="interactive",
    )

    assert result.returncode == 2
    assert "PARTITION must be one of: batch, cpu, cpu_datamover" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_stage_enroot_image_rejects_unpinned_source_before_submission(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/stage_enroot_image.sbatch",
        TEST_ONLY="1",
        SOURCE_IMAGE="nvcr.io/nvidia/nemo:nightly",
        SOURCE_COMMIT="b" * 40,
        OUTPUT_PREFIX="nemo_rl_nightly_20260731",
        CONTAINER_DIR=str(tmp_path / "containers"),
    )

    assert result.returncode == 2
    assert "SOURCE_DIGEST" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_stage_enroot_image_rejects_credential_bearing_or_ambiguous_references(
    tmp_path: Path,
) -> None:
    invalid_images = (
        "user:placeholder@nvcr.io/nvidia/nemo-rl:nightly",
        "nvcr.io/nvidia/nemo-rl:nightly?token=placeholder",
        "https://nvcr.io/nvidia/nemo-rl:nightly",
        "nvcr.io/nvidia/nemo rl:nightly",
        "nvcr.io/nvidia/nemo-rl:nightly\nsecond-line",
    )
    for source_image in invalid_images:
        result = _run_script(
            "scripts/stage_enroot_image.sbatch",
            TEST_ONLY="1",
            SOURCE_IMAGE=source_image,
            SOURCE_DIGEST="sha256:" + "a" * 64,
            SOURCE_COMMIT="b" * 40,
            OUTPUT_PREFIX="nemo_rl_nightly_20260731",
            CONTAINER_DIR=str(tmp_path / "containers"),
        )

        assert result.returncode == 2
        assert "credential-free registry/repository reference" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert source_image not in result.stdout
        assert source_image not in result.stderr


def test_stage_enroot_image_requires_full_source_commit_before_submission(
    tmp_path: Path,
) -> None:
    for source_commit in ("", "abc123"):
        result = _run_script(
            "scripts/stage_enroot_image.sbatch",
            TEST_ONLY="1",
            SOURCE_IMAGE="nvcr.io/nvidia/nemo-rl:nightly",
            SOURCE_DIGEST="sha256:" + "a" * 64,
            SOURCE_COMMIT=source_commit,
            OUTPUT_PREFIX="nemo_rl_nightly_20260731",
            CONTAINER_DIR=str(tmp_path / "containers"),
        )

        assert result.returncode == 2
        assert "SOURCE_COMMIT must be a full 40-character commit" in result.stderr
        assert "SBATCH:" not in result.stdout
        assert "SOURCE_IMAGE=" not in result.stdout


def test_scope_matrix_contains_all_32_te_rows_and_baseline() -> None:
    module = _load_experiment_module("scope_matrix")

    rows = module.load_scope_matrix()

    assert len(rows) == 33
    assert rows[0].scope == ()
    assert rows[0].cuda_graph_enabled is False
    assert {row.scope for row in rows[1:]} == VALID_TE_SCOPES
    assert rows[-1].scope == (
        "attn",
        "mlp",
        "mamba",
        "moe_router",
        "moe_preprocess",
    )


def test_scope_matrix_list_command_prints_every_persistent_row() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "scope_matrix.py"),
            "list",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 33
    assert lines[0].startswith("00\tbaseline_no_cg\tbaseline\t")
    assert lines[-1].startswith(
        "32\tattn_mlp_mamba_moe_router_preprocess\t"
        "attn,mlp,mamba,moe_router,moe_preprocess\t"
    )


def test_scope_launcher_does_not_require_host_uv(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 91\n")
    fake_uv.chmod(0o755)

    result = _run_script(
        "scopes/17_attn.sh",
        TEST_ONLY="1",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert "COMMAND:" in result.stdout
    assert "NVTE_WITH_NCCL_EP: 0" in result.stdout


def test_scope_classifier_reports_pre_submission_outcomes() -> None:
    module = _load_experiment_module("scope_matrix")
    rows = module.load_scope_matrix()
    by_name = {row.name: row for row in rows}

    assert module.classify_scope(by_name["attn"], model="nano").status == "runnable"
    assert (
        module.classify_scope(by_name["mamba"], model="qwen3_30ba3b").status
        == "model-incompatible"
    )
    assert (
        module.classify_scope(by_name["mlp"], model="qwen3_30ba3b").status
        == "model-incompatible"
    )
    assert (
        module.classify_scope(by_name["moe"], model="nano").status == "capacity-blocked"
    )
    nano_preprocess = module.classify_scope(
        by_name["moe_router_preprocess"], model="nano"
    )
    assert nano_preprocess.status == "capacity-blocked"
    assert "HybridEP moe_preprocess" in nano_preprocess.reason
    assert (
        module.classify_scope(by_name["attn"], model="ultra").status
        == "dependency-blocked"
    )
    ultra_with_external_paths = module.classify_scope(
        by_name["attn"],
        model="ultra",
        external_dependencies_ready=True,
    )
    assert ultra_with_external_paths.status == "dependency-blocked"
    assert "validated launcher adapter" in ultra_with_external_paths.reason
    assert (
        module.classify_scope(by_name["attn"], model="nano", mode="mcore").status
        == "dependency-blocked"
    )
    assert (
        module.classify_scope(
            by_name["attn"],
            model="nano",
            submitted_job_id="12345",
        ).status
        == "submitted"
    )


def test_rendered_nemorl_command_uses_only_current_graph_fields() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn",),
        steps=20,
        run_name="nano-attn-test",
    )

    assert "checkpointing.enabled=false" in command
    assert "policy.megatron_cfg.cuda_graph_warmup_steps=3" in command
    assert "policy.megatron_cfg.cuda_graph_modules=[attn]" in command
    assert "policy.megatron_cfg.thd_max_packed_sequences=16" in command
    assert "cluster.num_nodes=6" in command
    assert "policy.generation.colocated.enabled=false" in command
    assert "policy.generation.colocated.resources.num_nodes=2" in command
    assert "policy.generation.colocated.resources.gpus_per_node=4" in command
    assert "policy.offload_optimizer_for_logprob=false" in command
    assert "logger.wandb.project=sna-cg-study" in shlex.split(command)
    assert "NRL_FORCE_REBUILD_VENVS=true" in command
    assert "NEMO_RL_PY_EXECUTABLES_SYSTEM=1" not in command
    assert "++policy.router_replay.enabled=false" in command
    assert "cuda_graph_scope" not in command
    assert "cuda_graph_max_packed_seqs" not in command
    assert "cuda_graph_max_cached_schedules" not in command


def test_rendered_nemorl_command_can_use_shared_runtime_python_without_wandb() -> None:
    module = _load_experiment_module("scope_matrix")
    runtime_python = "/lustre/runtime/environment/bin/python"

    arguments = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=("moe_router",),
            steps=20,
            run_name="nano-router-shared-python",
            driver_python=runtime_python,
            wandb_enabled=False,
        )
    )

    assert arguments[:4] == [
        "env",
        "NRL_FORCE_REBUILD_VENVS=true",
        "UV_NO_EDITABLE=1",
        runtime_python,
    ]
    assert "uv" not in arguments
    assert arguments.count("logger.wandb_enabled=false") == 1
    assert "logger.wandb_enabled=true" not in arguments


def test_rendered_nemorl_command_adds_experimental_struct_keys_with_hydra_plus_plus() -> (
    None
):
    module = _load_experiment_module("scope_matrix")

    baseline_arguments = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=(),
            steps=20,
            run_name="nano-baseline-struct-test",
            cuda_graph_enabled=False,
        )
    )
    graph_arguments = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=("attn", "moe_router"),
            steps=20,
            run_name="nano-graph-struct-test",
        )
    )

    assert "++policy.megatron_cfg.cuda_graph_impl=none" in baseline_arguments
    assert not any(
        argument.startswith("++policy.megatron_cfg.cuda_graph_modules=")
        for argument in baseline_arguments
    )
    assert (
        "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" in graph_arguments
    )
    assert "++policy.megatron_cfg.cuda_graph_impl=transformer_engine" in graph_arguments
    assert (
        "++policy.megatron_cfg.cuda_graph_modules=[attn,moe_router]" in graph_arguments
    )
    assert "++policy.megatron_cfg.cuda_graph_warmup_steps=3" in graph_arguments
    assert "++policy.megatron_cfg.thd_max_packed_sequences=16" in graph_arguments


@pytest.mark.parametrize("model", ("nano", "qwen3_30ba3b"))
@pytest.mark.parametrize("cuda_graph_enabled", (False, True))
def test_inherited_fp64_campaign_recipes_pin_fp32_router_math(
    model: str, cuda_graph_enabled: bool
) -> None:
    """Baseline and graph rows must override the inherited fp64 recipe default."""
    module = _load_experiment_module("scope_matrix")

    arguments = shlex.split(
        module.render_scope_command(
            model=model,
            scope=("moe_router",) if cuda_graph_enabled else (),
            steps=20,
            run_name=f"{model}-router-fp32",
            cuda_graph_enabled=cuda_graph_enabled,
        )
    )

    assert arguments.count("++policy.megatron_cfg.moe_router_dtype=fp32") == 1


@pytest.mark.parametrize(
    "override",
    (
        "policy.megatron_cfg.moe_router_dtype=fp64",
        "++policy.megatron_cfg.moe_router_dtype=fp32",
        "~policy.megatron_cfg.moe_router_dtype",
    ),
)
def test_campaign_router_dtype_override_cannot_be_replaced_or_deleted(
    override: str,
) -> None:
    """Callers must not weaken or duplicate the campaign's router-math pin."""
    module = _load_experiment_module("scope_matrix")

    with pytest.raises(ValueError, match="protected campaign override"):
        module.render_scope_command(
            model="nano",
            scope=("moe_router",),
            steps=20,
            run_name="nano-protected-router-dtype",
            extra_overrides=(override,),
        )


def test_rendered_nano_command_pins_the_claimed_hybridep_dispatcher() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn", "moe_router"),
        steps=20,
        run_name="nano-hybridep-test",
    )

    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" in command
    assert "policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" in command
    assert "policy.megatron_cfg.moe_token_dispatcher_type=hybridep" not in command


@pytest.mark.parametrize(
    ("modules", "expected_padding"),
    (
        ("attn,mamba", "true"),
        ("attn,mamba,moe_router,moe_preprocess", "false"),
    ),
)
def test_direct_oci_launcher_submits_hybridep_with_scope_safe_padding(
    tmp_path: Path, modules: str, expected_padding: str
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/bin/bash
{
  printf 'ARGS='
  printf '%q ' "$@"
  printf '\nCOMMAND=%s\n' "${COMMAND:-}"
  for argument in "$@"; do
    printf 'ARG=%s\n' "${argument}"
  done
} >> "${FAKE_SBATCH_LOG}"
if [[ " $* " != *" --test-only "* ]]; then
  printf '12345\n'
fi
"""
    )
    fake_sbatch.chmod(0o755)

    exclusion = "nvl72027-T07,nvl72114-T[01-07,09-18]"
    result = _run_script(
        "scripts/submit_oci_nano_direct.sh",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        FAKE_SBATCH_LOG=str(sbatch_log),
        SOURCE_ROOT=str(REPO_ROOT),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        CONTAINER=str(tmp_path / "hybridep.sqsh"),
        MOE_TOKEN_DISPATCHER_TYPE="flex",
        MOE_FLEX_DISPATCHER_BACKEND="hybridep",
        HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="8",
        CUDA_GRAPH_MODULES=modules,
        TIME_LIMIT="04:00:00",
        EXCLUDE=exclusion,
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    scope_name = modules.replace(",", "-")
    assert f"RUN_NAME=nano-{scope_name}-5step-hybridep-unit" in result.stdout
    submissions = sbatch_log.read_text()
    assert submissions.count("ARGS=") == 2
    assert submissions.count(f"ARG=--exclude={exclusion}") == 2
    assert submissions.count("ARG=--time=04:00:00") == 2
    assert submissions.count("ARG=--nodes=6") == 2
    assert "cluster.num_nodes=6" in submissions
    assert "policy.generation.colocated.enabled=false" in submissions
    assert "policy.generation.colocated.resources.num_nodes=2" in submissions
    assert "policy.generation.colocated.resources.gpus_per_node=4" in submissions
    assert "policy.offload_optimizer_for_logprob=false" in submissions
    assert "++policy.megatron_cfg.hybridep_use_mnnvl=true" in submissions
    assert "USE_MNNVL=1" not in submissions
    assert "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN" not in submissions
    assert "++policy.megatron_cfg.hybridep_num_ranks_per_nvlink_domain=8" in submissions
    assert "policy.megatron_cfg.moe_token_dispatcher_type=flex" in submissions
    assert "++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep" in submissions
    assert (
        "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs="
        f"{expected_padding}"
    ) in submissions
    unexpected_padding = "false" if expected_padding == "true" else "true"
    assert (
        "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs="
        f"{unexpected_padding}"
    ) not in submissions
    assert "policy.megatron_cfg.moe_token_dispatcher_type=alltoall" not in submissions


def test_direct_oci_launcher_rejects_hybridep_domain_larger_than_nano_ep(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/bin/bash\nexit 99\n")
    fake_sbatch.chmod(0o755)

    result = _run_script(
        "scripts/submit_oci_nano_direct.sh",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SOURCE_ROOT=str(REPO_ROOT),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        MOE_TOKEN_DISPATCHER_TYPE="flex",
        MOE_FLEX_DISPATCHER_BACKEND="hybridep",
        HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="16",
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "must divide Nano expert_model_parallel_size=8" in result.stderr


def test_direct_oci_launcher_rejects_preprocess_without_router(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/bin/bash\nexit 99\n")
    fake_sbatch.chmod(0o755)

    result = _run_script(
        "scripts/submit_oci_nano_direct.sh",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        SOURCE_ROOT=str(REPO_ROOT),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        CUDA_GRAPH_MODULES="moe_preprocess",
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "moe_preprocess requires moe_router" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    (("POLICY_NUM_NODES", "0"), ("GENERATION_NUM_NODES", "1.5")),
)
def test_direct_oci_launcher_rejects_invalid_role_node_counts(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    result = _run_script(
        "scripts/submit_oci_nano_direct.sh",
        SOURCE_ROOT=str(REPO_ROOT),
        EXPERIMENT_ROOT=str(tmp_path / "runs"),
        **{name: value},
    )

    assert result.returncode == 2
    assert f"{name} must be a positive integer" in result.stderr
    assert "SUBMITTED_JOB_ID" not in result.stdout


def test_baseline_and_mamba_render_use_the_same_fused_attention_backend() -> None:
    module = _load_experiment_module("scope_matrix")

    commands = (
        module.render_scope_command(
            model="nano",
            scope=(),
            steps=20,
            run_name="nano-baseline-fused",
            cuda_graph_enabled=False,
        ),
        module.render_scope_command(
            model="nano",
            scope=("mamba",),
            steps=20,
            run_name="nano-mamba-fused",
        ),
    )

    for command in commands:
        assert "++policy.megatron_cfg.attention_backend=fused" in shlex.split(command)


def test_rendered_command_shell_quotes_log_paths_without_changing_arguments() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="nano",
        scope=("attn",),
        steps=20,
        run_name="nano-safe-path",
        log_dir="/lustre/experiment path/attn;literal",
    )

    arguments = shlex.split(command)
    assert "logger.log_dir=/lustre/experiment path/attn;literal" in arguments
    assert arguments.count("uv") == 1


def test_ultra_command_is_fail_closed_until_launcher_adapter_is_validated() -> None:
    module = _load_experiment_module("scope_matrix")

    with pytest.raises(ValueError, match="validated launcher adapter"):
        module.render_scope_command(
            model="ultra",
            scope=("attn",),
            steps=20,
            run_name="ultra-attn-test",
        )


def test_scope_and_variant_leaves_are_persistent_and_exact() -> None:
    module = _load_experiment_module("scope_matrix")
    rows = module.load_scope_matrix()
    scopes = sorted((EXPERIMENT_DIR / "scopes").glob("*.sh"))
    variants = sorted((EXPERIMENT_DIR / "variants").glob("*.sh"))

    assert [path.name for path in scopes] == [
        f"{row.index:02d}_{row.name}.sh" for row in rows
    ]
    assert len(variants) == 17
    for launcher in [*scopes, *variants]:
        text = launcher.read_text()
        assert "WARMUP_STEPS=3" in text
        assert "THD_MAX_PACKED_SEQUENCES=16" in text
        assert "CHECKPOINTING_ENABLED=false" in text
        assert "WANDB_PROJECT=sna-cg-study" in text
        assert 'bash "$(dirname "${BASH_SOURCE[0]}")/../run_scope.sh"' in text


def test_model_selectors_cover_nemotron_and_qwen_recipes() -> None:
    expected = {
        "nano.env": "examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-pack-cp.yaml",
        "super.env": "examples/configs/recipes/llm/grpo-nemotron3-super-120BA12B-8n4g-megatron.yaml",
        "ultra.env": "examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml",
        "qwen3_30ba3b.env": "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml",
        "qwen3_235b.env": "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml",
    }

    assert {
        path.name: next(
            line.removeprefix("NEMORL_RECIPE=")
            for line in path.read_text().splitlines()
            if line.startswith("NEMORL_RECIPE=")
        )
        for path in sorted((EXPERIMENT_DIR / "models").glob("*.env"))
    } == expected


def test_qwen3_235b_selector_enables_router_graphs_but_blocks_preprocess() -> None:
    module = _load_experiment_module("scope_matrix")

    spec = module.load_model_spec("qwen3_235b")

    assert spec.nemorl_recipe.endswith("grpo-qwen3-235b-16n4g.yaml")
    assert (
        spec.nemorl_cluster_num_nodes,
        spec.nemorl_generation_num_nodes,
        spec.gpus_per_node,
    ) == (
        18,
        2,
        4,
    )
    assert spec.dispatcher == "hybridep"
    assert spec.nemorl_tensorboard_enabled is False
    assert spec.moe_preprocess_graph_ready is False
    assert (
        module.classify_scope(
            module.find_scope_row("moe_router"), model="qwen3_235b"
        ).status
        == "runnable"
    )
    assert (
        module.classify_scope(
            module.find_scope_row("moe_router,moe_preprocess"), model="qwen3_235b"
        ).status
        == "capacity-blocked"
    )


@pytest.mark.parametrize(
    ("model", "allocation", "actor", "policy", "generation", "nemo_gym"),
    (
        ("nano", 6, 6, 4, 2, 0),
        ("super", 8, 8, 4, 4, 0),
        ("qwen3_30ba3b", 5, 5, 4, 1, 0),
        ("qwen3_235b", 18, 18, 16, 2, 0),
        ("ultra", 256, 236, 64, 172, 20),
    ),
)
def test_model_selectors_account_for_actor_and_external_nodes_separately(
    model: str,
    allocation: int,
    actor: int,
    policy: int,
    generation: int,
    nemo_gym: int,
) -> None:
    module = _load_experiment_module("scope_matrix")

    spec = module.load_model_spec(model)

    assert spec.nemorl_allocation_num_nodes == allocation
    assert spec.nemorl_cluster_num_nodes == actor
    assert spec.policy_num_nodes == policy
    assert spec.nemorl_generation_num_nodes == generation
    assert spec.nemorl_gym_num_nodes == nemo_gym
    assert spec.mcore_num_nodes == policy


def test_selector_tensorboard_policy_is_rendered_verbatim() -> None:
    module = _load_experiment_module("scope_matrix")

    qwen235 = shlex.split(
        module.render_scope_command(
            model="qwen3_235b",
            scope=(),
            steps=5,
            run_name="qwen235-tb",
            cuda_graph_enabled=False,
        )
    )
    nano = shlex.split(
        module.render_scope_command(
            model="nano",
            scope=(),
            steps=5,
            run_name="nano-tb",
            cuda_graph_enabled=False,
        )
    )

    assert "logger.tensorboard_enabled=false" in qwen235
    assert "logger.tensorboard_enabled=true" in nano


@pytest.mark.parametrize(
    ("environment", "error"),
    (
        ({"TEST_ONLY": "2"}, "TEST_ONLY must be 0 or 1"),
        ({"SBATCH_TEST_ONLY": "yes"}, "SBATCH_TEST_ONLY must be 0 or 1"),
        ({"TEST_ONLY": "1", "SBATCH_TEST_ONLY": "1"}, "mutually exclusive"),
    ),
)
def test_dry_run_flags_fail_before_any_rendered_output(
    environment: dict[str, str],
    error: str,
) -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        **environment,
    )

    assert result.returncode == 2
    assert error in result.stderr
    assert "STATUS:" not in result.stdout
    assert "COMMAND:" not in result.stdout
    assert "SBATCH:" not in result.stdout


@pytest.mark.parametrize(
    ("driver", "checker_exit", "expected_status", "expected_exit"),
    (
        ("exit 0", "0", "passed", 0),
        ("exit 17", "0", "not_run_driver_failed", 17),
        ("exit 0", "19", "failed", 19),
    ),
)
def test_r3_wrapper_records_atomic_terminal_result(
    tmp_path: Path,
    driver: str,
    checker_exit: str,
    expected_status: str,
    expected_exit: int,
) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "check_r3_trace.py").write_text("# fixture\n")
    driver_file = tmp_path / "driver.sh"
    driver_file.write_text(driver)
    fake_uv, _ = _create_staged_runtime(
        tmp_path,
        "#!/bin/bash\n"
        'printf invoked >"${CHECKER_MARKER:?}"\n'
        "[[ \"$1 $2 $3 $4\" == '-I -B -S -' ]] || exit 70\n"
        "[[ \"$*\" == *'--require-forward-verify'* ]] || exit 71\n"
        "[[ \"$*\" == *'--require-cp-identity'* ]] || exit 72\n"
        f"exit {checker_exit}\n",
    )
    marker = tmp_path / "checker.marker"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    wrapper = EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"

    result = subprocess.run(
        [
            "bash",
            str(wrapper),
            sys.executable,
            str(log_dir),
            str(root),
            str(fake_uv),
            str(driver_file),
            hashlib.sha256(driver_file.read_bytes()).hexdigest(),
            hashlib.sha256((tools / "check_r3_trace.py").read_bytes()).hexdigest(),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_SLURM_JOB_ID": "44",
            "NRL_SLURM_RESTART_COUNT": "3",
            "CHECKER_MARKER": str(marker),
        },
    )

    attempt = log_dir / "r3-validation-job-44-restart-3"
    record = json.loads((attempt / "r3-validation.json").read_text())
    assert result.returncode == expected_exit
    assert record["status"] == expected_status
    assert record["trace_dir"] == str(attempt / "trace-job-44-restart-3")
    assert "--require-forward-verify" in record["checker_command"]
    assert "--require-cp-identity" in record["checker_command"]
    assert record["checker_source_path"] == str(tools / "check_r3_trace.py")
    assert record["checker_expected_sha256"] == record["checker_actual_sha256"]
    assert (attempt / "r3-validation.env").is_file()
    r3_environment = dict(
        line.split("=", 1)
        for line in (attempt / "r3-validation.env").read_text().splitlines()
    )
    assert (
        base64.b64decode(r3_environment["trace_dir_base64"]).decode()
        == record["trace_dir"]
    )
    assert base64.b64decode(r3_environment["driver_command_base64"]).decode() == driver
    assert r3_environment["checker_sha256"] == record["checker_actual_sha256"]
    assert marker.exists() is (expected_status != "not_run_driver_failed")
    if expected_status == "not_run_driver_failed":
        assert record["checker_exit_code"] is None


def test_router_replay_renders_wrapper_but_r3off_keeps_driver() -> None:
    replay = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="qwen3_30ba3b",
        MODE="nemorl",
        ROUTER_REPLAY="on",
        TEST_ONLY="1",
        RUN_TAG="r3-wrapper",
    )
    direct = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="qwen3_30ba3b",
        MODE="nemorl",
        ROUTER_REPLAY="off",
        TEST_ONLY="1",
        RUN_TAG="r3-direct",
    )

    assert replay.returncode == 0, replay.stderr
    assert "run_r3_validated_command.sh" in replay.stdout
    assert "r3-driver-command.sh" in replay.stdout
    assert direct.returncode == 0, direct.stderr
    assert "run_r3_validated_command.sh" not in direct.stdout
    assert "examples/run_grpo.py" in direct.stdout


def test_r3_wrapper_rejects_replaced_driver_bytes_before_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "check_r3_trace.py").write_text("# fixture\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    driver = tmp_path / "driver.sh"
    driver.write_text("exit 0")
    expected = hashlib.sha256(driver.read_bytes()).hexdigest()
    driver.write_text("exit 88")
    staged_uv, _ = _create_staged_runtime(tmp_path)
    wrapper = EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"

    result = subprocess.run(
        [
            "bash",
            str(wrapper),
            sys.executable,
            str(log_dir),
            str(root),
            str(staged_uv),
            str(driver),
            expected,
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NRL_SLURM_JOB_ID": "1"},
    )

    assert result.returncode == 2
    assert "digest mismatch" in result.stderr
    assert not list(log_dir.rglob("r3-validation.json"))


def test_r3_wrapper_binds_checker_bytes_and_exact_driver_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    checker = tools / "check_r3_trace.py"
    checker.write_bytes(b"original checker bytes\n")
    captured_checker = tmp_path / "checker.stdin"
    driver_environment = tmp_path / "driver.env"
    injected = tmp_path / "bash-env-injected"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(
        "if [[ ${NRL_R3_TRACE:-0} == 1 ]]; then "
        f"printf injected >{shlex.quote(str(injected))}; fi\n"
    )
    driver = tmp_path / "driver.sh"
    driver_bytes = (
        'printf \'%s\\n\' "${NRL_R3_TRACE_DIR}" "${NRL_R3_TRACE_STEPS}" '
        '"${NRL_R3_TRACE_SAMPLES}" "${NRL_R3_TRACE_MICROBATCHES}" '
        '"${NRL_R3_TRACE}" "${NRL_R3_TRACE_VERIFY_FORWARD}" '
        '"${NRL_ROUTER_REPLAY_VALIDATE}" "${BASH_ENV-unset}" "${ENV-unset}" '
        '>"${DRIVER_ENV_CAPTURE}"\n'
        f"printf mutated >{shlex.quote(str(checker))}"
    ).encode()
    driver.write_bytes(driver_bytes)
    fake_uv, _ = _create_staged_runtime(
        tmp_path,
        "#!/bin/bash\n"
        "[[ \"$1 $2 $3 $4\" == '-I -B -S -' ]] || exit 70\n"
        'cat >"${CHECKER_CAPTURE:?}"\n',
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"),
            sys.executable,
            str(log_dir),
            str(root),
            str(fake_uv),
            str(driver),
            hashlib.sha256(driver_bytes).hexdigest(),
            hashlib.sha256(checker.read_bytes()).hexdigest(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_SLURM_JOB_ID": "52",
            "DRIVER_ENV_CAPTURE": str(driver_environment),
            "CHECKER_CAPTURE": str(captured_checker),
            "BASH_ENV": str(bash_env),
            "ENV": str(bash_env),
        },
    )

    assert result.returncode == 0, result.stderr
    assert captured_checker.read_bytes() == b"original checker bytes\n"
    assert driver_environment.read_text().splitlines()[1:] == [
        "5",
        "2",
        "2",
        "1",
        "1",
        "1",
        "unset",
        "unset",
    ]
    assert not injected.exists()
    record = json.loads(
        (log_dir / "r3-validation-job-52-restart-0" / "r3-validation.json").read_text()
    )
    assert record["driver_command"] == driver_bytes.decode()


def test_r3_wrapper_runs_checker_with_attested_runtime_python(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    checker = tools / "check_r3_trace.py"
    checker.write_bytes(b"original checker bytes\n")
    driver = tmp_path / "driver.sh"
    driver.write_text("exit 0")
    stage = tmp_path / "staged-runtimes" / ("a" * 64)
    fake_uv = stage / "uv" / "uv"
    fake_uv.parent.mkdir(parents=True)
    uv_marker = tmp_path / "uv.marker"
    fake_uv.write_text(
        f"#!/bin/bash\nprintf invoked >{shlex.quote(str(uv_marker))}\nexit 99\n"
    )
    fake_uv.chmod(0o755)
    runtime_python = stage / "environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    checker_capture = tmp_path / "checker.stdin"
    argv_capture = tmp_path / "checker.argv"
    runtime_python.write_text(
        "#!/bin/bash\n"
        'printf \'%s\\n\' "$@" >"${CHECKER_ARGV_CAPTURE:?}"\n'
        'cat >"${CHECKER_CAPTURE:?}"\n'
    )
    runtime_python.chmod(0o755)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"),
            sys.executable,
            str(log_dir),
            str(root),
            str(fake_uv),
            str(driver),
            hashlib.sha256(driver.read_bytes()).hexdigest(),
            hashlib.sha256(checker.read_bytes()).hexdigest(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_SLURM_JOB_ID": "55",
            "CHECKER_CAPTURE": str(checker_capture),
            "CHECKER_ARGV_CAPTURE": str(argv_capture),
        },
    )

    assert result.returncode == 0, result.stderr
    assert not uv_marker.exists()
    assert checker_capture.read_bytes() == checker.read_bytes()
    assert argv_capture.read_text().splitlines()[:4] == ["-I", "-B", "-S", "-"]
    record = json.loads(
        (log_dir / "r3-validation-job-55-restart-0" / "r3-validation.json").read_text()
    )
    assert record["checker_command"][0] == str(runtime_python)


def test_r3_wrapper_isolates_checker_and_keeps_stage_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    checker = tools / "check_r3_trace.py"
    checker.write_text(
        "try:\n"
        "    import r3_pythonpath_injected\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    r3_pythonpath_injected.mark()\n"
    )
    driver = tmp_path / "driver.sh"
    driver.write_text("exit 0")
    staged_uv, runtime_python = _create_staged_runtime(tmp_path)
    runtime_python.unlink()
    runtime_python.symlink_to(sys.executable)
    injected = tmp_path / "pythonpath-injected"
    malicious = tmp_path / "malicious"
    malicious.mkdir()
    (malicious / "r3_pythonpath_injected.py").write_text(
        "from pathlib import Path\n"
        f"def mark():\n    Path({str(injected)!r}).write_text('injected')\n"
    )
    stage_root = staged_uv.parents[1]
    before = sorted(str(path.relative_to(stage_root)) for path in stage_root.rglob("*"))
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"),
            sys.executable,
            str(log_dir),
            str(root),
            str(staged_uv),
            str(driver),
            hashlib.sha256(driver.read_bytes()).hexdigest(),
            hashlib.sha256(checker.read_bytes()).hexdigest(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "NRL_SLURM_JOB_ID": "56",
            "PYTHONPATH": str(malicious),
            "PYTHONDONTWRITEBYTECODE": "0",
        },
    )

    after = sorted(str(path.relative_to(stage_root)) for path in stage_root.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert not injected.exists()
    assert after == before


def test_r3_wrapper_rejects_checker_digest_before_driver(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (tools / "check_r3_trace.py").write_text("original\n")
    driver_marker = tmp_path / "driver-ran"
    driver = tmp_path / "driver.sh"
    driver.write_text(f"printf ran >{shlex.quote(str(driver_marker))}")
    staged_uv, _ = _create_staged_runtime(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"),
            sys.executable,
            str(log_dir),
            str(root),
            str(staged_uv),
            str(driver),
            hashlib.sha256(driver.read_bytes()).hexdigest(),
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NRL_SLURM_JOB_ID": "53"},
    )

    assert result.returncode == 2
    assert "checker digest mismatch" in result.stderr
    assert not driver_marker.exists()
    assert not list(log_dir.rglob("r3-validation.json"))


def test_r3_wrapper_normalizes_signal_exit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    checker = tools / "check_r3_trace.py"
    checker.write_text("# fixture\n")
    driver = tmp_path / "driver.sh"
    driver.write_text("kill -TERM $$")
    staged_uv, _ = _create_staged_runtime(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_r3_validated_command.sh"),
            sys.executable,
            str(log_dir),
            str(root),
            str(staged_uv),
            str(driver),
            hashlib.sha256(driver.read_bytes()).hexdigest(),
            hashlib.sha256(checker.read_bytes()).hexdigest(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NRL_SLURM_JOB_ID": "54"},
    )

    assert result.returncode == 143
    record = json.loads(
        (log_dir / "r3-validation-job-54-restart-0" / "r3-validation.json").read_text()
    )
    assert record["status"] == "not_run_driver_failed"
    assert record["driver_exit_code"] == 143
    assert record["driver_raw_return_code"] == -15


@pytest.mark.parametrize("router_replay", ("off", "on"))
def test_fake_sbatch_submission_writes_strict_complete_metadata(
    tmp_path: Path,
    router_replay: str,
) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    container = tmp_path / "container.sqsh"
    container.write_bytes(b"container")
    container_sha = hashlib.sha256(container.read_bytes()).hexdigest()
    profile.write_text(
        profile.read_text()
        .replace("CONTAINER=/tmp/container.sqsh", f"CONTAINER={container}")
        .replace(
            f"CONTAINER_SHA256={provenance['container_sha256']}",
            f"CONTAINER_SHA256={container_sha}",
        )
    )
    verifier = experiment / "scripts" / "verify_source_provenance.sh"
    verifier.write_text("#!/bin/bash\nexit 0\n")
    verifier.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "sbatch"
    fake.write_text("#!/bin/bash\nprintf '321\\n'\n")
    fake.chmod(0o755)
    logs = tmp_path / "logs"

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/00_baseline_no_cg.sh",
        CLUSTER="oci-hsg",
        MODEL="qwen3_30ba3b" if router_replay == "on" else "nano",
        MODE="nemorl",
        STEPS="5",
        TEST_ONLY="0",
        SBATCH_TEST_ONLY="0",
        RUN_TAG=f"metadata-{router_replay}",
        ROUTER_REPLAY=router_replay,
        PROFILE_FILE=str(profile),
        LOG_ROOT_OVERRIDE=str(logs),
        NEMORL_WANDB_ENABLED="true" if router_replay == "on" else "false",
        PATH=f"{fake_bin}:{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    run_dir = next(logs.iterdir())
    env = dict(
        line.split("=", 1)
        for line in (run_dir / "run-metadata.env").read_text().splitlines()
    )
    assert all(
        re.fullmatch(r"""[a-z][a-z0-9_]*=[^'"`$\\;|&<>]*""", line)
        for line in (run_dir / "run-metadata.env").read_text().splitlines()
    )
    decoded = {
        key.removesuffix("_base64"): base64.b64decode(value).decode()
        for key, value in env.items()
        if key.endswith("_base64")
    }
    record = json.loads((run_dir / "run-metadata.json").read_text())
    assert env["job_id"] == "321" and record["job_id"] == 321
    assert isinstance(record["sbatch_argv"], list)
    assert decoded["rendered_driver_command"] == record["rendered_driver_command"]
    assert decoded["effective_command"] == record["command"]
    assert decoded["output_pattern"] == record["output_pattern"]
    assert decoded["resolved_output_path"] == record["resolved_output_path"]
    assert decoded["run_log_dir"] == record["run_log_dir"]
    assert record["tensorboard_enabled"] is True
    assert record["wandb_enabled"] is (router_replay == "on")
    assert record["container_path"] == str(container)
    assert decoded["container_path"] == record["container_path"]
    assert record["runtime_preflight_job_id"] == 1
    assert isinstance(record["runtime_preflight_job_id"], int)
    assert (
        record["runtime_attestation_sha256"] == provenance["runtime_attestation_sha256"]
    )
    assert decoded["runtime_attestation"] == record["runtime_attestation"]
    assert decoded["megatron_checkpoint_dir"] == record["megatron_checkpoint_dir"]
    assert record["megatron_checkpoint_dir"] == str(
        Path(record["runtime_attestation"]).parent / "megatron-checkpoints"
    )
    assert decoded["managed_python_install_dir"] == record["managed_python_install_dir"]
    assert decoded["uv_executable"] == record["uv_executable"]
    assert decoded["runtime_python"] == record["runtime_python"]
    assert decoded["vllm_runtime_python"] == record["vllm_runtime_python"]
    assert decoded["hf_home"] == record["hf_home"] == ""
    expected_runtime_contract = (
        ("dropless_alltoall_qwen30_16", "deep-ep,fast-hadamard-transform")
        if router_replay == "on"
        else ("dropless_hybridep_nano16", "fast-hadamard-transform")
    )
    assert record["runtime_feature_set"] == expected_runtime_contract[0]
    assert record["runtime_excluded_packages"] == expected_runtime_contract[1]
    assert decoded["r3_record_python"] == record["r3_record_python"]
    assert env["scope_name"] == record["scope_name"] == "baseline_no_cg"
    expected_topology = (
        {
            "num_nodes": 5,
            "gpus_per_node": 4,
            "nemorl_allocation_num_nodes": 5,
            "nemorl_cluster_num_nodes": 5,
            "policy_num_nodes": 4,
            "nemorl_generation_num_nodes": 1,
            "nemorl_gym_num_nodes": 0,
            "mcore_num_nodes": 4,
        }
        if router_replay == "on"
        else {
            "num_nodes": 6,
            "gpus_per_node": 4,
            "nemorl_allocation_num_nodes": 6,
            "nemorl_cluster_num_nodes": 6,
            "policy_num_nodes": 4,
            "nemorl_generation_num_nodes": 2,
            "nemorl_gym_num_nodes": 0,
            "mcore_num_nodes": 4,
        }
    )
    assert record["topology"] == expected_topology
    if router_replay == "on":
        assert decoded["r3_validation_record_pattern"]
        assert decoded["r3_validation_record_initial_path"].endswith(
            "/r3-validation-job-321-restart-0/r3-validation.json"
        )
        assert env["r3_driver_command_sha256"] == record["r3_driver_command_sha256"]
        assert env["r3_checker_sha256"] == record["r3_checker_sha256"]
        assert decoded["r3_checker_path"] == record["r3_checker_path"]
        assert env["r3_driver_command_sha256"] in record["command"]
        assert env["r3_checker_sha256"] in record["command"]
    else:
        assert decoded["r3_validation_record_pattern"] == ""
        assert decoded["r3_validation_record_initial_path"] == ""


def test_fake_sbatch_scheduler_test_only_publishes_no_metadata(tmp_path: Path) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    container = tmp_path / "container.sqsh"
    container.write_bytes(b"container")
    profile.write_text(
        profile.read_text()
        .replace("CONTAINER=/tmp/container.sqsh", f"CONTAINER={container}")
        .replace(
            f"CONTAINER_SHA256={provenance['container_sha256']}",
            f"CONTAINER_SHA256={hashlib.sha256(container.read_bytes()).hexdigest()}",
        )
    )
    verifier = experiment / "scripts" / "verify_source_provenance.sh"
    verifier.write_text("#!/bin/bash\nexit 0\n")
    verifier.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation = tmp_path / "sbatch.argv"
    fake = fake_bin / "sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        'printf \'ENV_SBATCH_GPUS=%s\\n\' "${SBATCH_GPUS-unset}" >"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_GPUS_PER_NODE=%s\\n\' "${SBATCH_GPUS_PER_NODE-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_GRES=%s\\n\' "${SBATCH_GRES-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_TRES_PER_TASK=%s\\n\' "${SBATCH_TRES_PER_TASK-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_TEST_ONLY=%s\\n\' "${SBATCH_TEST_ONLY-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_EXCLUSIVE=%s\\n\' "${SBATCH_EXCLUSIVE-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'ENV_SBATCH_MEM=%s\\n\' "${SBATCH_MEM-unset}" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        'printf \'%s\\n\' "$@" >>"${FAKE_SBATCH_INVOCATION:?}"\n'
        "printf 'sbatch: Job 321 would be submitted\\n'\n"
    )
    fake.chmod(0o755)
    logs = tmp_path / "logs"

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/00_baseline_no_cg.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="5",
        TEST_ONLY="0",
        SBATCH_TEST_ONLY="1",
        RUN_TAG="scheduler-test-only",
        PROFILE_FILE=str(profile),
        LOG_ROOT_OVERRIDE=str(logs),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        FAKE_SBATCH_INVOCATION=str(invocation),
        SBATCH_GPUS="7",
        SBATCH_GPUS_PER_NODE="7",
        SBATCH_GRES="gpu:7",
        SBATCH_TRES_PER_TASK="gres/gpu=1",
        SBATCH_EXCLUSIVE="1",
        SBATCH_MEM="0",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "SBATCH_TEST_ONLY_OUTPUT: sbatch: Job 321 would be submitted" in result.stdout
    )
    invocation_lines = invocation.read_text().splitlines()
    assert "ENV_SBATCH_GPUS=unset" in invocation_lines
    assert "ENV_SBATCH_GPUS_PER_NODE=unset" in invocation_lines
    assert "ENV_SBATCH_GRES=unset" in invocation_lines
    assert "ENV_SBATCH_TRES_PER_TASK=unset" in invocation_lines
    assert "ENV_SBATCH_TEST_ONLY=unset" in invocation_lines
    assert "ENV_SBATCH_EXCLUSIVE=unset" in invocation_lines
    assert "ENV_SBATCH_MEM=unset" in invocation_lines
    assert "--test-only" in invocation_lines
    assert not logs.exists()


@pytest.mark.parametrize("sbatch_output", ("", "0", "1.5", "warning\n321"))
def test_fake_sbatch_rejects_malformed_real_job_ids(
    tmp_path: Path, sbatch_output: str
) -> None:
    root, experiment, profile, provenance = _campaign_leaf_harness(tmp_path)
    container = tmp_path / "container.sqsh"
    container.write_bytes(b"container")
    profile.write_text(
        profile.read_text()
        .replace("CONTAINER=/tmp/container.sqsh", f"CONTAINER={container}")
        .replace(
            f"CONTAINER_SHA256={provenance['container_sha256']}",
            f"CONTAINER_SHA256={hashlib.sha256(container.read_bytes()).hexdigest()}",
        )
    )
    verifier = experiment / "scripts" / "verify_source_provenance.sh"
    verifier.write_text("#!/bin/bash\nexit 0\n")
    verifier.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "sbatch"
    fake.write_text("#!/bin/bash\nprintf '%s' \"${FAKE_SBATCH_OUTPUT-}\"\n")
    fake.chmod(0o755)
    logs = tmp_path / "logs"
    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/00_baseline_no_cg.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="5",
        TEST_ONLY="0",
        RUN_TAG="malformed",
        PROFILE_FILE=str(profile),
        LOG_ROOT_OVERRIDE=str(logs),
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        FAKE_SBATCH_OUTPUT=sbatch_output,
    )
    assert result.returncode == 2
    assert "invalid job ID" in result.stderr
    assert not list(logs.rglob("run-metadata.*")) if logs.exists() else True


def test_router_replay_rendering_is_explicit_and_gates_router_graphs() -> None:
    module = _load_experiment_module("scope_matrix")

    arguments = shlex.split(
        module.render_scope_command(
            model="qwen3_30ba3b",
            scope=(),
            steps=5,
            run_name="qwen30-baseline-r3on",
            cuda_graph_enabled=False,
            router_replay_enabled=True,
        )
    )

    assert "++policy.router_replay.enabled=true" in arguments
    assert "loss_fn.force_on_policy_ratio=false" in arguments
    assert "NRL_ROUTER_REPLAY_VALIDATE=1" in arguments
    assert "NRL_R3_TRACE_VERIFY_FORWARD=1" in arguments
    for scope in (("moe_router",), ("attn", "mamba", "moe_router")):
        graph_arguments = shlex.split(
            module.render_scope_command(
                model="nano",
                scope=scope,
                steps=5,
                run_name="nano-r3-router-graph",
                router_replay_enabled=True,
            )
        )
        assert "++policy.router_replay.enabled=true" in graph_arguments
        assert (
            f"++policy.megatron_cfg.cuda_graph_modules=[{','.join(scope)}]"
            in graph_arguments
        )
    for model, scope in (
        ("nano", ("moe",)),
        ("nano", ()),
        ("nano", ("attn", "moe_router")),
        ("nano", ("moe_router", "moe_preprocess")),
        ("qwen3_30ba3b", ("moe_router",)),
    ):
        with pytest.raises(ValueError, match="Router Replay.*router CUDA Graph"):
            module.render_scope_command(
                model=model,
                scope=scope,
                steps=5,
                run_name="unsafe-router-graph",
                router_replay_enabled=True,
            )


@pytest.mark.parametrize(
    ("router_replay_enabled", "override"),
    (
        (False, "++policy.router_replay.enabled=true"),
        (True, "+policy.router_replay.enabled=false"),
        (False, "~policy.router_replay.enabled"),
        (False, "~policy.router_replay.enabled=true"),
        (True, "++policy.generation.vllm_cfg.enable_prefix_caching=true"),
        (True, "~policy.generation.vllm_cfg.enable_prefix_caching"),
        (True, "~policy.generation.vllm_cfg.enable_prefix_caching=true"),
        (True, "++policy.generation.vllm_kwargs.enable_chunked_prefill=true"),
        (True, "~policy.generation.vllm_kwargs.enable_chunked_prefill"),
        (True, "~policy.generation.vllm_kwargs.enable_chunked_prefill=true"),
        (True, "loss_fn.force_on_policy_ratio=true"),
        (True, "~loss_fn.force_on_policy_ratio"),
    ),
)
def test_router_replay_rejects_normalized_protected_overrides(
    router_replay_enabled: bool, override: str
) -> None:
    module = _load_experiment_module("scope_matrix")

    with pytest.raises(ValueError, match="protected Router Replay override"):
        module.render_scope_command(
            model="qwen3_30ba3b",
            scope=("attn",),
            steps=5,
            run_name="protected-override",
            router_replay_enabled=router_replay_enabled,
            extra_overrides=(override,),
        )


def test_router_replay_off_allows_unprotected_vllm_override() -> None:
    module = _load_experiment_module("scope_matrix")

    command = module.render_scope_command(
        model="qwen3_30ba3b",
        scope=("attn",),
        steps=5,
        run_name="r3off-vllm-override",
        extra_overrides=("++policy.generation.vllm_cfg.enable_prefix_caching=true",),
    )

    assert "++policy.generation.vllm_cfg.enable_prefix_caching=true" in shlex.split(
        command
    )


def test_nano_router_replay_can_force_modular_triton_moe_backend() -> None:
    result = _run_script(
        "scopes/05_mamba.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        ROUTER_REPLAY="on",
        TEST_ONLY="1",
        VLLM_MOE_BACKEND="triton",
    )

    assert result.returncode == 0, result.stderr
    assert "++policy.generation.vllm_kwargs.moe_backend=triton" in result.stdout


def test_vllm_moe_backend_rejects_unknown_value() -> None:
    result = _run_script(
        "scopes/05_mamba.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        TEST_ONLY="1",
        VLLM_MOE_BACKEND="unknown",
    )

    assert result.returncode == 2
    assert "VLLM_MOE_BACKEND must be auto or triton" in result.stderr


def test_router_replay_shell_validation_rejects_invalid_and_unsafe_graphs() -> None:
    invalid_value = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        ROUTER_REPLAY="invalid",
    )
    unsafe_router_graphs = [
        _run_script(
            scope,
            CLUSTER="oci-hsg",
            MODEL=model,
            MODE="nemorl",
            ROUTER_REPLAY="on",
            TEST_ONLY="1",
        )
        for model, scope in (
            ("nano", "scopes/01_whole_layer.sh"),
            ("nano", "scopes/02_moe.sh"),
            ("nano", "scopes/04_moe_router_preprocess.sh"),
            ("nano", "scopes/19_attn_moe_router.sh"),
            ("qwen3_30ba3b", "scopes/03_moe_router.sh"),
        )
    ]

    for result in (invalid_value, *unsafe_router_graphs):
        assert result.returncode == 2
        assert "SBATCH:" not in result.stdout
    assert "ROUTER_REPLAY must be off or on" in invalid_value.stderr
    for result in unsafe_router_graphs:
        assert "Router Replay cannot be combined with router CUDA Graph scopes" in (
            result.stderr
        )


@pytest.mark.parametrize(
    "scope",
    ("scopes/03_moe_router.sh", "scopes/23_attn_mamba_moe_router.sh"),
)
def test_router_replay_router_graph_test_only_requires_exact_runtime_feature(
    tmp_path: Path,
    scope: str,
) -> None:
    """A supported leaf requests only the version-bound runtime attestation."""
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    (tmp_path / "runtime.json").write_text(
        json.dumps(
            {
                "runtime_feature_set": ("dropless_hybridep_nano16_r3_router_graph_v1"),
                "mcore_capabilities": {
                    "router_replay_cuda_graph_input": ("r3_router_cuda_graph_input_v1")
                },
            }
        )
    )
    result = _run_copied_experiment_script(
        root,
        experiment,
        scope,
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        ROUTER_REPLAY="on",
        TEST_ONLY="1",
        RUN_TAG="r3-router-graph-v1",
        PROFILE_FILE=str(profile),
    )

    assert result.returncode == 0, result.stderr
    runtime_attestation_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("RUNTIME_ATTESTATION: ")
    )
    runtime_attestation_command = shlex.split(
        runtime_attestation_line.removeprefix("RUNTIME_ATTESTATION: ")
    )[0]
    assert (
        "--runtime-feature-set dropless_hybridep_nano16_r3_router_graph_v1"
        in runtime_attestation_command
    )
    assert "SBATCH:" in result.stdout


@pytest.mark.parametrize(
    ("runtime_feature_set", "mcore_capability"),
    (
        ("dropless_hybridep_nano16", "r3_router_cuda_graph_input_v1"),
        ("dropless_hybridep_nano16_r3_router_graph_v1", None),
        (
            "dropless_hybridep_nano16_r3_router_graph_v1",
            "r3_router_cuda_graph_input_v0",
        ),
    ),
)
def test_router_replay_router_graph_test_only_rejects_unbound_attestation(
    tmp_path: Path,
    runtime_feature_set: str,
    mcore_capability: str | None,
) -> None:
    """Legacy, missing, and wrong-version evidence fail before SBATCH rendering."""
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    payload: dict[str, object] = {"runtime_feature_set": runtime_feature_set}
    if mcore_capability is not None:
        payload["mcore_capabilities"] = {
            "router_replay_cuda_graph_input": mcore_capability
        }
    (tmp_path / "runtime.json").write_text(json.dumps(payload))

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/03_moe_router.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        ROUTER_REPLAY="on",
        TEST_ONLY="1",
        RUN_TAG="r3-router-graph-unbound",
        PROFILE_FILE=str(profile),
    )

    assert result.returncode == 2
    assert "must bind dropless_hybridep_nano16_r3_router_graph_v1" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_nano_test_only_launcher_renders_batch_job_without_singleton() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_GROUP="unit-performance-group",
        REPEAT_INDEX="2",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: runnable" in result.stdout
    assert "verify_runtime_attestation.py" in result.stdout
    assert "validate_te_runtime.py" not in result.stdout
    assert "policy.megatron_cfg.cuda_graph_modules=\\[attn\\]" in result.stdout
    assert "policy.megatron_cfg.thd_max_packed_sequences=16" in result.stdout
    assert "policy.megatron_cfg.cuda_graph_warmup_steps=3" in result.stdout
    assert "checkpointing.enabled=false" in result.stdout
    assert "logger.wandb.project=sna-cg-study" in result.stdout
    assert "--partition=batch" in result.stdout
    assert f"--chdir={REPO_ROOT}" in result.stdout
    assert "RUN_GROUP: unit-performance-group" in result.stdout
    assert "REPEAT_INDEX: 2" in result.stdout
    assert "run_nemorl_scope.sub" in result.stdout
    assert "dependency" not in result.stdout.lower()
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_nano_router_launcher_forwards_overlap_param_gather_diagnostic() -> None:
    result = _run_script(
        "scopes/03_moe_router.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="overlap-param-gather-off",
        OVERLAP_PARAM_GATHER="false",
    )

    assert result.returncode == 0, result.stderr
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("COMMAND: ")
    )
    command = shlex.split(command_line.removeprefix("COMMAND: "))[0]
    arguments = shlex.split(command)
    assert (
        "policy.megatron_cfg.distributed_data_parallel_config."
        "overlap_param_gather=false"
    ) in arguments


def test_nano_attention_launcher_forwards_hybridep_uneven_input_padding() -> None:
    result = _run_script(
        "variants/attn_hybridep_pad_uneven.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="hybridep-uneven-padding",
    )

    assert result.returncode == 0, result.stderr
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("COMMAND: ")
    )
    command = shlex.split(command_line.removeprefix("COMMAND: "))[0]
    arguments = shlex.split(command)
    assert (
        arguments.count(
            "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=true"
        )
        == 1
    )


def test_nano_attention_cache3_variant_is_persistent_and_exact() -> None:
    result = _run_script(
        "variants/attn_hybridep_pad_uneven_cache3.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="hybridep-uneven-padding-cache3",
    )

    assert result.returncode == 0, result.stderr
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("COMMAND: ")
    )
    command = shlex.split(command_line.removeprefix("COMMAND: "))[0]
    arguments = shlex.split(command)
    assert (
        arguments.count(
            "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=true"
        )
        == 1
    )
    assert (
        arguments.count("++policy.megatron_cfg.cuda_graph_max_cached_schedules=3") == 1
    )


@pytest.mark.parametrize(
    ("launcher", "scope", "cache_capacity"),
    (
        ("variants/baseline_hybridep_pad_uneven.sh", "baseline", None),
        ("variants/mamba_hybridep_pad_uneven_cache4.sh", "mamba", "4"),
        (
            "variants/moe_router_hybridep_pad_uneven_cache4.sh",
            "moe_router",
            "4",
        ),
        (
            "variants/attn_mamba_hybridep_pad_uneven_cache4.sh",
            "attn,mamba",
            "4",
        ),
        (
            "variants/attn_mamba_moe_router_hybridep_pad_uneven_cache4.sh",
            "attn,mamba,moe_router",
            "4",
        ),
    ),
)
def test_nano_hybridep_padding_variants_are_persistent_and_exact(
    launcher: str, scope: str, cache_capacity: str | None
) -> None:
    result = _run_script(
        launcher,
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="hybridep-phase-padding",
    )

    assert result.returncode == 0, result.stderr
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("COMMAND: ")
    )
    command = shlex.split(command_line.removeprefix("COMMAND: "))[0]
    arguments = shlex.split(command)
    assert f"++policy.megatron_cfg.cuda_graph_modules=[{scope}]" in arguments or (
        scope == "baseline"
        and "++policy.megatron_cfg.cuda_graph_impl=none" in arguments
    )
    assert (
        arguments.count(
            "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=true"
        )
        == 1
    )
    cache_arguments = [
        argument
        for argument in arguments
        if argument.startswith("++policy.megatron_cfg.cuda_graph_max_cached_schedules=")
    ]
    if cache_capacity is None:
        assert cache_arguments == []
    else:
        assert cache_arguments == [
            f"++policy.megatron_cfg.cuda_graph_max_cached_schedules={cache_capacity}"
        ]


def test_nano_preprocess_launcher_rejects_static_uneven_padding() -> None:
    result = _run_script(
        "scopes/04_moe_router_preprocess.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="hybridep-preprocess-static-padding",
        HYBRIDEP_PAD_UNEVEN_DISPATCH_INPUTS="true",
    )

    assert result.returncode == 2
    assert "moe_preprocess capture must start with uneven-input padding disabled" in (
        result.stderr
    )
    assert "SBATCH:" not in result.stdout


def test_nano_launcher_rejects_invalid_hybridep_uneven_input_padding() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        TEST_ONLY="1",
        RUN_TAG="hybridep-invalid-padding",
        HYBRIDEP_PAD_UNEVEN_DISPATCH_INPUTS="invalid",
    )

    assert result.returncode == 2
    assert "HYBRIDEP_PAD_UNEVEN_DISPATCH_INPUTS must be true or false" in result.stderr
    assert "SBATCH:" not in result.stdout


@pytest.mark.parametrize("capacity", ("0", "true", "3.0"))
def test_nano_launcher_rejects_invalid_graph_cache_capacity(capacity: str) -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        TEST_ONLY="1",
        RUN_TAG="invalid-graph-cache-capacity",
        CUDA_GRAPH_MAX_CACHED_SCHEDULES=capacity,
    )

    assert result.returncode == 2
    assert "CUDA_GRAPH_MAX_CACHED_SCHEDULES must be a positive integer" in result.stderr
    assert "SBATCH:" not in result.stdout


def test_baseline_launcher_rejects_irrelevant_graph_cache_capacity() -> None:
    result = _run_script(
        "scopes/00_baseline_no_cg.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        TEST_ONLY="1",
        RUN_TAG="irrelevant-graph-cache-capacity",
        CUDA_GRAPH_MAX_CACHED_SCHEDULES="3",
    )

    assert result.returncode == 2
    assert (
        "CUDA_GRAPH_MAX_CACHED_SCHEDULES requires NeMo-RL Transformer Engine "
        "training graphs" in result.stderr
    )
    assert "SBATCH:" not in result.stdout


@pytest.mark.parametrize(
    ("relative_path", "model", "mode"),
    (
        ("variants/attn_hybridep_pad_uneven.sh", "super", "nemorl"),
        ("variants/attn_hybridep_pad_uneven.sh", "nano", "mcore"),
    ),
)
def test_hybridep_uneven_input_padding_rejects_unvalidated_contracts(
    relative_path: str, model: str, mode: str
) -> None:
    result = _run_script(
        relative_path,
        CLUSTER="oci-hsg",
        MODEL=model,
        MODE=mode,
        TEST_ONLY="1",
        RUN_TAG="hybridep-unsafe-padding",
        HYBRIDEP_PAD_UNEVEN_DISPATCH_INPUTS="true",
    )

    assert result.returncode == 2
    assert (
        "HYBRIDEP_PAD_UNEVEN_DISPATCH_INPUTS requires "
        "MODEL=nano MODE=nemorl DISPATCHER=hybridep"
    ) in result.stderr
    assert "SBATCH:" not in result.stdout


def test_nano_ptyche_launcher_omits_unsupported_gpu_tres_options(
    tmp_path: Path,
) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    profile.write_text(
        profile.read_text()
        .replace("PROFILE_ID=unit", "PROFILE_ID=ptyche")
        .replace("SBATCH_GRES=gpu:4", "SBATCH_GRES=none")
        .replace("SBATCH_SEGMENT_SIZE=", "SBATCH_SEGMENT_SIZE=2")
    )

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        PROFILE_FILE=str(profile),
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_GROUP="unit-ptyche",
        REPEAT_INDEX="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout
    assert "--segment=2" in result.stdout
    assert (
        "--job-name=unit-cuda-graph.attn-nano-nemorl-ptyche-20step-r3off-unit"
        in result.stdout
    )


def test_leaf_job_verifies_one_exact_runtime_preflight_artifact_without_dependency(
    tmp_path: Path,
) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    attestation = "/lustre/example/runtime/oci-container-runtime-733.json"
    staged_uv = f"/lustre/example/runtime/staged-runtimes/{'a' * 64}/uv/uv"
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=oci-hsg-runtime-attested",
                "ACCOUNT=coreai_dlalgo_nemorl",
                "PARTITION=batch",
                "CONTAINER=/lustre/example/nemo_rl_immutable.sqsh",
                f"CONTAINER_SHA256={CONTAINER_SHA256}",
                "MOUNTS=/lustre:/lustre",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=gpu:4",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=04:00:00",
                f"RUNTIME_ATTESTATION={attestation}",
                "RUNTIME_PREFLIGHT_JOB_ID=733",
                f"UV_EXECUTABLE={staged_uv}",
                f"EXPECTED_TE_SHA={TE_SHA}",
                f"EXPECTED_TE_VERSION_BASE_SHA={TE_SHA}",
                f"EXPECTED_NEMORL_SHA={NEMORL_SHA}",
                f"EXPECTED_BRIDGE_SHA={BRIDGE_SHA}",
                f"EXPECTED_MCORE_SHA={MCORE_SHA}",
                "",
            )
        )
    )

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        PROFILE_FILE=str(profile),
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "--dependency" not in result.stdout
    assert "verify_runtime_attestation.py" in result.stdout
    assert attestation in result.stdout
    assert "validate_te_runtime.py" not in result.stdout
    assert f"MANAGED_PYTHON_VERSION: {PYTHON_VERSION}" in result.stdout
    assert (
        "MANAGED_PYTHON_INSTALL_DIR: "
        "/lustre/example/runtime/uv-python-installations" in result.stdout
    )
    runtime_attestation_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("RUNTIME_ATTESTATION: ")
    )
    runtime_attestation_command = shlex.split(
        runtime_attestation_line.removeprefix("RUNTIME_ATTESTATION: ")
    )[0]
    assert f"--expected-python-version {PYTHON_VERSION}" in runtime_attestation_command
    assert (
        "--expected-python-install-dir "
        "/lustre/example/runtime/uv-python-installations" in runtime_attestation_command
    )
    assert f"PINNED_UV_VERSION: {UV_VERSION}" in result.stdout
    assert f"UV_EXECUTABLE: {staged_uv}" in result.stdout
    assert f"--expected-uv-version {UV_VERSION}" in runtime_attestation_command
    assert f"--expected-uv-executable {staged_uv}" in runtime_attestation_command
    assert "--expected-runtime-attestation-job-id 733" in runtime_attestation_command


def test_leaf_uses_attested_shared_python_and_can_disable_wandb(
    tmp_path: Path,
) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    runtime_stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    runtime_python = runtime_stage_root / "environment" / "bin" / "python"

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        PROFILE_FILE=str(profile),
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="shared-python",
        NEMORL_WANDB_ENABLED="false",
    )

    assert result.returncode == 0, result.stderr
    assert f"RUNTIME_PYTHON: {runtime_python}" in result.stdout
    command_line = next(
        line for line in result.stdout.splitlines() if line.startswith("COMMAND: ")
    )
    command = shlex.split(command_line.removeprefix("COMMAND: "))[0]
    arguments = shlex.split(command)
    assert arguments[:3] == [
        "env",
        "NRL_FORCE_REBUILD_VENVS=true",
        "UV_NO_EDITABLE=1",
    ]
    assert arguments[3] == str(runtime_python)
    assert "logger.wandb_enabled=false" in arguments
    assert "logger.wandb_enabled=true" not in arguments


def test_leaf_runtime_attestation_uses_the_nightly_container_python() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    runtime_attestation_line = next(
        line
        for line in result.stdout.splitlines()
        if line.startswith("RUNTIME_ATTESTATION: ")
    )
    runtime_attestation_command = shlex.split(
        runtime_attestation_line.removeprefix("RUNTIME_ATTESTATION: ")
    )[0]
    assert runtime_attestation_command.startswith("/opt/nemo_rl_venv/bin/python ")
    assert "/usr/bin/python3" not in runtime_attestation_command
    assert (
        "--runtime-feature-set dropless_hybridep_nano16" in runtime_attestation_command
    )
    assert "--excluded-packages fast-hadamard-transform" in runtime_attestation_command
    assert "--torch-cuda-arch-list 10.0a" in runtime_attestation_command
    assert "--nvte-cuda-archs 100a" in runtime_attestation_command


def test_leaf_job_rejects_relative_profile_uv_executable(tmp_path: Path) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    profile.write_text(
        re.sub(
            r"^UV_EXECUTABLE=.*$",
            "UV_EXECUTABLE=relative/uv",
            profile.read_text(),
            flags=re.MULTILINE,
        )
    )

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        PROFILE_FILE=str(profile),
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "UV_EXECUTABLE must be an absolute path" in result.stderr


def test_leaf_job_rejects_unmounted_profile_uv_executable(tmp_path: Path) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    profile.write_text(
        re.sub(
            r"^UV_EXECUTABLE=.*$",
            "UV_EXECUTABLE=/outside/runtime/uv/uv",
            profile.read_text().replace(
                f"MOUNTS={tmp_path}:{tmp_path}", "MOUNTS=/lustre:/lustre"
            ),
            flags=re.MULTILINE,
        ).replace(
            f"RUNTIME_ATTESTATION={tmp_path}/runtime.json",
            "RUNTIME_ATTESTATION=/lustre/runtime/runtime.json",
        )
    )

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        PROFILE_FILE=str(profile),
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "pinned uv executable is not container-mounted" in result.stderr


def test_leaf_job_rejects_unmounted_managed_python_installation(
    tmp_path: Path,
) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    profile.write_text(
        "\n".join(
            (
                "PROFILE_ID=oci-hsg-runtime-unmounted",
                "ACCOUNT=coreai_dlalgo_nemorl",
                "PARTITION=batch",
                "CONTAINER=/lustre/example/nemo_rl_immutable.sqsh",
                f"CONTAINER_SHA256={CONTAINER_SHA256}",
                "MOUNTS=/lustre:/lustre",
                "SBATCH_GPUS_PER_NODE=4",
                "SBATCH_GRES=gpu:4",
                "SBATCH_SEGMENT_SIZE=",
                "TIME_LIMIT=04:00:00",
                "RUNTIME_ATTESTATION=/shared/runtime/oci-container-runtime-733.json",
                "RUNTIME_PREFLIGHT_JOB_ID=733",
                f"EXPECTED_TE_SHA={TE_SHA}",
                f"EXPECTED_TE_VERSION_BASE_SHA={TE_SHA}",
                f"EXPECTED_NEMORL_SHA={NEMORL_SHA}",
                f"EXPECTED_BRIDGE_SHA={BRIDGE_SHA}",
                f"EXPECTED_MCORE_SHA={MCORE_SHA}",
                "",
            )
        )
    )

    result = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        PROFILE_FILE=str(profile),
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "managed Python install directory is not container-mounted" in result.stderr


def test_source_provenance_verifier_rejects_queued_worktree_mutation(
    tmp_path: Path,
) -> None:
    repositories_and_commits = [
        _create_clean_git_repository(tmp_path, name)
        for name in ("nemo-rl", "bridge", "mcore")
    ]
    verifier = (EXPERIMENT_DIR / "scripts" / "verify_source_provenance.sh").resolve()
    arguments = [str(verifier)]
    for repository, commit in repositories_and_commits:
        arguments.extend((str(repository.resolve()), commit))

    clean = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr
    assert "SOURCE_PROVENANCE_VERIFIED=true" in clean.stdout

    (repositories_and_commits[1][0] / "tracked.txt").write_text("mutated\n")
    dirty = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dirty.returncode != 0
    assert "Megatron-Bridge source worktree has unstaged changes" in dirty.stderr


def test_scope_job_wrappers_revalidate_source_before_container_execution() -> None:
    for name in ("run_nemorl_scope.sub", "run_mcore_scope.sub"):
        source = (EXPERIMENT_DIR / "scripts" / name).read_text()
        verification_offset = source.index('"${SOURCE_PROVENANCE_VERIFIER}"')
        attestation_offset = source.index('"${RUNTIME_ATTESTATION_COMMAND}"')

        assert verification_offset < attestation_offset
        assert "EXPECTED_NEMORL_SHA" in source
        assert "EXPECTED_BRIDGE_SHA" in source
        assert "EXPECTED_MCORE_SHA" in source
        assert 'sha256sum "${CONTAINER}"' not in source


@pytest.mark.parametrize(
    ("wrapper_name", "expected_srun_count"),
    (("run_nemorl_scope.sub", 1), ("run_mcore_scope.sub", 2)),
)
def test_scope_job_wrappers_never_mount_host_home(
    wrapper_name: str, expected_srun_count: int
) -> None:
    source = (EXPERIMENT_DIR / "scripts" / wrapper_name).read_text()

    assert source.count("--no-container-mount-home") == expected_srun_count


def test_nemorl_job_wrapper_requires_managed_python_contract(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "ray.sub").write_text("#!/bin/bash\nexit 0\n")
    environment = os.environ.copy()
    environment.update(
        {
            "COMMAND": "true",
            "CONTAINER": "/lustre/example/nightly.sqsh",
            "CONTAINER_SHA256": CONTAINER_SHA256,
            "MOUNTS": "/lustre:/lustre",
            "RUNTIME_ATTESTATION_COMMAND": "true",
            "REPO_ROOT": str(repo_root),
            "EXPECTED_NEMORL_SHA": NEMORL_SHA,
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "SOURCE_PROVENANCE_VERIFIER": "/usr/bin/true",
        }
    )
    for variable in (
        "PINNED_UV_VERSION",
        "UV_EXECUTABLE",
        "UV_PYTHON",
        "UV_PYTHON_INSTALL_DIR",
        "UV_MANAGED_PYTHON",
        "UV_PYTHON_DOWNLOADS",
    ):
        environment.pop(variable, None)

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_nemorl_scope.sub"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PINNED_UV_VERSION" in result.stderr


def test_nemorl_job_wrapper_uses_shared_attested_python_and_isolates_worker_venvs(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".python-version").write_text(f"{PYTHON_VERSION}\n")
    (repo_root / "docker").mkdir()
    (repo_root / "docker" / "Dockerfile").write_text(f"ARG UV_VERSION={UV_VERSION}\n")
    environment_log = tmp_path / "managed-python.env"
    (repo_root / "ray.sub").write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"${UV_PROJECT_ENVIRONMENT:-}\" "
        '"${UV_PYTHON:-}" "${UV_PYTHON_INSTALL_DIR:-}" '
        '"${UV_MANAGED_PYTHON:-}" "${UV_PYTHON_DOWNLOADS:-}" '
        '"${UV_NO_EDITABLE:-}" '
        '"${UV_PROJECT:-}" '
        '"${CONTAINER_ENV_VARS:-}" '
        '"${CONTAINER_PATH_PREFIX:-}" '
        '"${RUNTIME_PYTHON:-}" '
        '"${NEMO_RL_VENV_DIR:-}" '
        '"${NRL_MEGATRON_CHECKPOINT_DIR:-}" '
        '"${NEMO_RL_MCORE_PY_EXECUTABLE:-}" '
        '"${NEMO_RL_VLLM_PY_EXECUTABLE:-}" '
        '"${PATH:-}" '
        '"${NRL_SLURM_JOB_ID:-}" '
        '"${NRL_SLURM_RESTART_COUNT:-}" '
        '>"${ENVIRONMENT_LOG}"\n'
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text("#!/bin/bash\nexit 0\n")
    fake_srun.chmod(0o755)
    python_install_dir = tmp_path / "uv-python-installations"
    runtime_stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    uv_executable = runtime_stage_root / "uv" / "uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_text(f"#!/bin/sh\nprintf 'uv {UV_VERSION} (fixture)\\n'\n")
    uv_executable.chmod(0o755)
    runtime_python = runtime_stage_root / "environment" / "bin" / "python"
    vllm_runtime_python = runtime_stage_root / "vllm-environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    vllm_runtime_python.parent.mkdir(parents=True)
    managed_python = python_install_dir / "cpython-fixture" / "bin" / "python3.13"
    managed_python.parent.mkdir(parents=True)
    managed_python.write_text("#!/bin/sh\nexit 0\n")
    managed_python.chmod(0o755)
    runtime_python.symlink_to(managed_python)
    vllm_runtime_python.symlink_to(managed_python)
    base_log_dir = tmp_path / "logs"
    megatron_checkpoint_dir = tmp_path / "shared" / "megatron-checkpoints"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "733",
            "COMMAND": "true",
            "CONTAINER": "/lustre/example/nightly.sqsh",
            "CONTAINER_SHA256": CONTAINER_SHA256,
            "MOUNTS": "/lustre:/lustre",
            "RUNTIME_ATTESTATION_COMMAND": "true",
            "REPO_ROOT": str(repo_root),
            "EXPECTED_NEMORL_SHA": NEMORL_SHA,
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "SOURCE_PROVENANCE_VERIFIER": "/usr/bin/true",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_executable),
            "RUNTIME_PYTHON": str(runtime_python),
            "NEMO_RL_MCORE_PY_EXECUTABLE": str(runtime_python),
            "NEMO_RL_VLLM_PY_EXECUTABLE": str(vllm_runtime_python),
            "UV_PYTHON": PYTHON_VERSION,
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "UV_MANAGED_PYTHON": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "NVTE_WITH_NCCL_EP": "0",
            "BASE_LOG_DIR": str(base_log_dir),
            "NRL_MEGATRON_CHECKPOINT_DIR": str(megatron_checkpoint_dir),
            "ENVIRONMENT_LOG": str(environment_log),
        }
    )
    environment.pop("UV_PROJECT_ENVIRONMENT", None)

    result = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_nemorl_scope.sub"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    environment_lines = environment_log.read_text().splitlines()
    assert environment_lines[:14] == [
        str(runtime_stage_root / "environment"),
        PYTHON_VERSION,
        str(python_install_dir),
        "1",
        "never",
        "1",
        "/tmp/nemo-rl-worker-projects/job-733-restart-0/source",
        CONTAINER_ENV_VARS,
        str(uv_executable.parent),
        str(runtime_python),
        "/tmp/nemo-rl-worker-venvs/job-733-restart-0",
        str(megatron_checkpoint_dir),
        str(runtime_python),
        str(vllm_runtime_python),
    ]
    assert megatron_checkpoint_dir.is_dir()
    assert environment_lines[14].split(":")[0] == str(fake_bin)
    assert environment_lines[15:] == ["733", "0"]

    runtime_python.unlink()
    outside_python = tmp_path / "outside-managed-python" / "bin" / "python3.13"
    outside_python.parent.mkdir(parents=True)
    outside_python.write_text("#!/bin/sh\nexit 0\n")
    outside_python.chmod(0o755)
    runtime_python.symlink_to(outside_python)

    rejected = subprocess.run(
        [
            "bash",
            str(EXPERIMENT_DIR / "scripts" / "run_nemorl_scope.sub"),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 2
    assert "RUNTIME_PYTHON must resolve inside UV_PYTHON_INSTALL_DIR" in rejected.stderr


def test_readonly_actor_venv_probe_uses_non_editable_tier_installs() -> None:
    source = (EXPERIMENT_DIR / "scripts" / "probe_readonly_actor_venvs.sub").read_text()

    assert "export UV_NO_EDITABLE=1" in source
    assert "NVTE_WITH_NCCL_EP=0" in source
    assert "export NVTE_CUDA_ARCHS=100a" in source
    assert "export TORCH_CUDA_ARCH_LIST=10.0a" in source
    assert "export CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:?}" in source
    assert (
        "READONLY_ACTOR_VENV_PROBE_PAYLOAD,NVTE_WITH_NCCL_EP,NVTE_CUDA_ARCHS,"
        "TORCH_CUDA_ARCH_LIST,CMAKE_BUILD_PARALLEL_LEVEL" in source
    )
    assert "probe_tier vllm vllm" in source
    assert "probe_tier mcore megatron.core" in source
    assert '--container-image="${CONTAINER}"' in source
    assert 'bash "${PROBE_SCRIPT_PATH}"' in source
    assert source.count('find "${project_root}" -xdev') == 2
    assert "export UV_PROJECT=${build_project_root}" in source
    assert "--exclude='*.egg-info'" in source
    assert "diff -qr --no-dereference" in source
    assert '"${uv_executable}" sync --locked --directory "${project_root}"' in source
    assert '"${uv_executable}" run --locked --extra "${tier}"' in source
    assert "path.is_relative_to(root)" in source


def test_nemorl_wrapper_builds_actor_venvs_from_private_exact_source_copy() -> None:
    source = (EXPERIMENT_DIR / "scripts" / "run_nemorl_scope.sub").read_text()

    assert "export UV_PROJECT=/tmp/nemo-rl-worker-projects/job-" in source
    assert "export NVTE_CUDA_ARCHS=100a" in source
    assert "export TORCH_CUDA_ARCH_LIST=10.0a" in source
    assert "export CMAKE_BUILD_PARALLEL_LEVEL=32" in source
    assert 'tar --exclude="./.git" --exclude="*/.git"' in source
    assert "diff -qr --no-dereference --exclude=.git" in source
    assert 'chmod -R u+w -- "${UV_PROJECT}"' in source
    assert '"${source_project_root}" "${UV_PROJECT}"' in source


@pytest.mark.parametrize(
    ("wrapper_name", "extra_environment"),
    (("run_nemorl_scope.sub", {}),),
)
def test_scope_job_wrapper_rejects_mutated_uv_before_executing_it(
    tmp_path: Path,
    wrapper_name: str,
    extra_environment: dict[str, str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".python-version").write_text(f"{PYTHON_VERSION}\n")
    (repo_root / "docker").mkdir()
    (repo_root / "docker" / "Dockerfile").write_text(f"ARG UV_VERSION={UV_VERSION}\n")
    (repo_root / "ray.sub").write_text("#!/bin/bash\nexit 0\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        "while (( $# > 0 )); do\n"
        '  if [[ "$1" == /bin/bash ]]; then\n'
        '    exec "$@"\n'
        "  fi\n"
        "  shift\n"
        "done\n"
        "exit 97\n"
    )
    fake_srun.chmod(0o755)

    execution_marker = tmp_path / "mutated-uv-executed"
    python_install_dir = tmp_path / "uv-python-installations"
    runtime_stage_root = tmp_path / "staged-runtimes" / ("a" * 64)
    uv_executable = runtime_stage_root / "uv" / "uv"
    uv_executable.parent.mkdir(parents=True)
    uv_executable.write_text(
        "#!/bin/sh\n"
        'printf executed >"${UV_EXECUTION_MARKER}"\n'
        f"printf 'uv {UV_VERSION} (mutated fixture)\\n'\n"
    )
    uv_executable.chmod(0o755)
    runtime_python = runtime_stage_root / "environment" / "bin" / "python"
    vllm_runtime_python = runtime_stage_root / "vllm-environment" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    vllm_runtime_python.parent.mkdir(parents=True)
    managed_python = python_install_dir / "cpython-fixture" / "bin" / "python3.13"
    managed_python.parent.mkdir(parents=True)
    managed_python.write_text("#!/bin/sh\nexit 0\n")
    managed_python.chmod(0o755)
    runtime_python.symlink_to(managed_python)
    vllm_runtime_python.symlink_to(managed_python)
    host_execution_marker = tmp_path / "unattested-path-command-executed"
    sibling_srun = uv_executable.parent / "srun"
    sibling_srun.write_text(
        '#!/bin/sh\nprintf executed >"${HOST_EXECUTION_MARKER}"\nexit 72\n'
    )
    sibling_srun.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "812",
            "COMMAND": "true",
            "CONTAINER": "/lustre/example/nightly.sqsh",
            "CONTAINER_SHA256": CONTAINER_SHA256,
            "MOUNTS": "/lustre:/lustre",
            "RUNTIME_ATTESTATION_COMMAND": (
                'echo "uv executable SHA256 mismatch" >&2; exit 73'
            ),
            "REPO_ROOT": str(repo_root),
            "EXPECTED_NEMORL_SHA": NEMORL_SHA,
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "SOURCE_PROVENANCE_VERIFIER": "/usr/bin/true",
            "PINNED_UV_VERSION": UV_VERSION,
            "UV_EXECUTABLE": str(uv_executable),
            "RUNTIME_PYTHON": str(runtime_python),
            "NEMO_RL_MCORE_PY_EXECUTABLE": str(runtime_python),
            "NEMO_RL_VLLM_PY_EXECUTABLE": str(vllm_runtime_python),
            "UV_EXECUTION_MARKER": str(execution_marker),
            "HOST_EXECUTION_MARKER": str(host_execution_marker),
            "UV_PYTHON": PYTHON_VERSION,
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
            "UV_MANAGED_PYTHON": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "NVTE_WITH_NCCL_EP": "0",
            "BASE_LOG_DIR": str(tmp_path / "logs"),
            "NRL_MEGATRON_CHECKPOINT_DIR": str(
                tmp_path / "shared" / "megatron-checkpoints"
            ),
            **extra_environment,
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "scripts" / wrapper_name)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 73
    assert "uv executable SHA256 mismatch" in result.stderr
    assert not execution_marker.exists()
    assert not host_execution_marker.exists()


def test_source_snapshot_copies_exact_recursive_gitlinks_and_writes_manifest(
    tmp_path: Path,
) -> None:
    mcore, mcore_sha = _create_clean_git_repository(tmp_path, "mcore-source")
    bridge, _ = _create_clean_git_repository(tmp_path, "bridge-source")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(bridge),
            "submodule",
            "add",
            "-q",
            str(mcore),
            "3rdparty/Megatron-LM",
        ],
        check=True,
    )
    _git(bridge, "commit", "-qm", "pin mcore")
    bridge_sha = _git(bridge, "rev-parse", "HEAD")
    nested_mcore = bridge / "3rdparty" / "Megatron-LM"

    outer, _ = _create_clean_git_repository(tmp_path, "nemo-rl-source")
    experiment_scripts = (
        outer
        / "experiments"
        / "cuda_graph"
        / "nemotron_thd_te_graph_20260731"
        / "scripts"
    )
    experiment_scripts.mkdir(parents=True)
    for name in ("create_source_snapshot.sh", "verify_source_provenance.sh"):
        shutil.copy2(EXPERIMENT_DIR / "scripts" / name, experiment_scripts / name)
    (outer / "uv.lock").write_text("fixture-lock\n")
    _git(outer, "add", "experiments", "uv.lock")
    _git(outer, "commit", "-qm", "add snapshot tools")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(outer),
            "submodule",
            "add",
            "-q",
            str(bridge),
            "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge",
        ],
        check=True,
    )
    _git(outer, "commit", "-qm", "pin bridge")
    outer_sha = _git(outer, "rev-parse", "HEAD")
    nested_bridge = outer / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge"
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(outer),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )

    snapshot_store = tmp_path / "snapshots"
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "scripts" / "create_source_snapshot.sh")],
        env=os.environ
        | {
            "SOURCE_ROOT": str(outer.resolve()),
            "SNAPSHOT_STORE": str(snapshot_store.resolve()),
            "EXPECTED_NEMORL_SHA": outer_sha,
            "EXPECTED_BRIDGE_SHA": bridge_sha,
            "EXPECTED_MCORE_SHA": mcore_sha,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    snapshot = snapshot_store / f"{outer_sha[:12]}-{bridge_sha[:12]}-{mcore_sha[:12]}"
    manifest = snapshot / ".source-manifest.env"
    assert _git(snapshot, "rev-parse", "HEAD") == outer_sha
    assert (
        _git(
            snapshot / "3rdparty" / "Megatron-Bridge-workspace" / "Megatron-Bridge",
            "rev-parse",
            "HEAD",
        )
        == bridge_sha
    )
    assert (
        _git(
            snapshot
            / "3rdparty"
            / "Megatron-Bridge-workspace"
            / "Megatron-Bridge"
            / "3rdparty"
            / "Megatron-LM",
            "rev-parse",
            "HEAD",
        )
        == mcore_sha
    )
    submodule_status = subprocess.run(
        ["git", "-C", str(snapshot), "submodule", "status", "--recursive"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(submodule_status) == 2
    assert all(line.startswith(" ") for line in submodule_status)
    manifest_text = manifest.read_text()
    assert f"nemo_rl_commit={outer_sha}" in manifest_text
    assert f"bridge_commit={bridge_sha}" in manifest_text
    assert f"mcore_commit={mcore_sha}" in manifest_text
    assert "uv_lock_sha256=" in manifest_text


def test_ray_submission_has_no_global_singleton_dependency() -> None:
    ray_submission = (REPO_ROOT / "ray.sub").read_text()

    assert "#SBATCH --dependency=singleton" not in ray_submission


def test_ray_and_nemorl_sruns_override_image_uv_environment() -> None:
    ray_submission = (REPO_ROOT / "ray.sub").read_text()
    assert 'CONTAINER_ENV_VARS="${CONTAINER_ENV_VARS:-}"' in ray_submission
    assert "--container-env=$CONTAINER_ENV_VARS" in ray_submission
    assert "invalid CONTAINER_ENV_VARS" in ray_submission
    assert ray_submission.count(r'export PATH="\${CONTAINER_PATH_PREFIX}:\$PATH"') == 2

    nemorl_wrapper = (EXPERIMENT_DIR / "scripts" / "run_nemorl_scope.sub").read_text()
    assert (
        ': "${NVTE_WITH_NCCL_EP:?run_scope.sh must export NVTE_WITH_NCCL_EP}"'
        in nemorl_wrapper
    )
    assert f"CONTAINER_ENV_VARS={CONTAINER_ENV_VARS}" in nemorl_wrapper
    assert "export NRL_SLURM_JOB_ID=${SLURM_JOB_ID:?}" in nemorl_wrapper
    assert "export NRL_SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}" in nemorl_wrapper
    assert "NEMO_RL_MCORE_PY_EXECUTABLE" in nemorl_wrapper
    assert "NEMO_RL_VLLM_PY_EXECUTABLE" in nemorl_wrapper
    assert "export CONTAINER_ENV_VARS" in nemorl_wrapper


def test_runtime_stage_builds_and_attests_split_actor_environments() -> None:
    source = (
        EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    ).read_text()

    assert "--locked --extra mcore --group test" in source
    assert "--locked --extra vllm --no-python-downloads" in source
    assert (
        "probe_vllm_actor_runtime"
        in (EXPERIMENT_DIR / "validate_container_runtime.py").read_text()
    )


def test_cluster_profiles_render_cluster_specific_gres_and_segment_contracts(
    tmp_path: Path,
) -> None:
    root, experiment, profile, _ = _campaign_leaf_harness(tmp_path)
    profile.write_text(
        profile.read_text()
        .replace("PROFILE_ID=unit", "PROFILE_ID=ptyche")
        .replace("SBATCH_GRES=gpu:4", "SBATCH_GRES=none")
    )
    ptyche = _run_copied_experiment_script(
        root,
        experiment,
        "scopes/17_attn.sh",
        CLUSTER="ptyche",
        PROFILE_FILE=str(profile),
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )
    lyris = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="lyris",
        MODEL="nano",
        MODE="nemorl",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert ptyche.returncode == 0, ptyche.stderr
    assert "--gpus-per-node" not in ptyche.stdout
    assert "--gres=" not in ptyche.stdout
    assert "--segment=6" in ptyche.stdout
    assert lyris.returncode == 0, lyris.stderr
    assert "--gres=" not in lyris.stdout
    assert "--segment=" not in lyris.stdout


@pytest.mark.parametrize(
    ("num_nodes", "expected"),
    ((2, 2), (4, 4), (5, 5), (6, 6), (8, 8), (16, 16), (18, 18), (64, 16), (256, 16)),
)
def test_ptyche_segment_is_derived_for_every_reachable_allocation(
    num_nodes: int, expected: int
) -> None:
    module = _load_experiment_module("slurm_segment")

    assert module.resolve_segment_size("ptyche", num_nodes, "") == expected


def test_segment_resolution_validates_explicit_profile_value() -> None:
    module = _load_experiment_module("slurm_segment")

    assert module.resolve_segment_size("ptyche", 6, "2") == 2
    assert module.resolve_segment_size("oci-hsg", 6, "") is None
    with pytest.raises(ValueError, match="divide"):
        module.resolve_segment_size("ptyche", 6, "4")
    with pytest.raises(ValueError, match="at most 18"):
        module.resolve_segment_size("ptyche", 36, "36")


def test_ptyche_profile_requires_immutable_container_and_auto_segment() -> None:
    values = dict(
        line.split("=", 1)
        for line in (EXPERIMENT_DIR / "profiles" / "ptyche.env.example")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    )

    assert values["CONTAINER"] == "__REQUIRED_IMMUTABLE_CONTAINER__"
    assert values["SBATCH_SEGMENT_SIZE"] == ""


def test_mcore_launcher_is_dependency_blocked_without_standalone_driver() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="mcore",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: dependency-blocked" in result.stdout
    assert "MCORE_DRIVER" in result.stdout
    assert "SBATCH:" not in result.stdout


def test_mcore_launcher_rejects_successful_noop_as_a_driver() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="mcore",
        MCORE_DRIVER="true",
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: dependency-blocked" in result.stdout
    assert "committed standalone driver" in result.stdout
    assert "COMMAND:" not in result.stdout
    assert "SBATCH:" not in result.stdout


def test_mcore_launcher_allocates_policy_nodes_not_nemorl_total() -> None:
    result = _run_script(
        "scopes/17_attn.sh",
        CLUSTER="oci-hsg",
        MODEL="nano",
        MODE="mcore",
        MCORE_DRIVER=str(EXPERIMENT_DIR / "scripts" / "run_mcore_training.py"),
        STEPS="20",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS: runnable" in result.stdout
    assert "--nodes=4" in result.stdout
    assert "--nodes=6" not in result.stdout


def test_submitters_pin_smoke_performance_and_accuracy_steps() -> None:
    cases = (
        ("submit_smoke_matrix.sh", "5"),
        ("submit_performance_matrix.sh", "20"),
        ("submit_accuracy_soak.sh", "100"),
    )
    for relative_path, steps in cases:
        result = _run_script(
            relative_path,
            CLUSTER="oci-hsg",
            MODEL="nano",
            MODE="nemorl",
            TEST_ONLY="1",
            RUN_TAG="unit",
        )
        assert result.returncode == 0, result.stderr
        assert f"STEPS: {steps}" in result.stdout


@pytest.mark.parametrize(
    "relative_path",
    ("submit_performance_matrix.sh", "submit_accuracy_soak.sh"),
)
@pytest.mark.parametrize("model", ("qwen3_30ba3b", "qwen3_235b"))
def test_legacy_generic_submitters_route_qwen_to_campaign_submitter(
    relative_path: str,
    model: str,
) -> None:
    result = _run_script(
        relative_path,
        CLUSTER="oci-hsg",
        MODEL=model,
        MODE="nemorl",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 2
    assert "submit_qwen_router_validation.sh" in result.stderr
    for launch_marker in ("STATUS:", "COMMAND:", "SBATCH:"):
        assert launch_marker not in result.stdout


@pytest.mark.parametrize(
    "relative_path",
    ("submit_performance_matrix.sh", "submit_accuracy_soak.sh"),
)
@pytest.mark.parametrize("model", ("nano", "super", "ultra"))
def test_legacy_generic_submitters_preserve_nemotron_selectors(
    relative_path: str,
    model: str,
) -> None:
    result = _run_script(
        relative_path,
        CLUSTER="oci-hsg",
        MODEL=model,
        MODE="nemorl",
        TEST_ONLY="1",
        RUN_TAG="unit",
    )

    assert result.returncode == 0, result.stderr
    assert "MATRIX_ROW:" in result.stdout
    assert "submit_qwen_router_validation.sh" not in result.stderr


def test_oci_container_runtime_smoke_renders_four_gpu_batch_job(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 0, result.stderr
    assert "SBATCH: sbatch --parsable" in result.stdout
    assert "--partition=batch" in result.stdout
    assert "--account=coreai_dlalgo_nemorl" in result.stdout
    assert "--gres=gpu:4" in result.stdout
    assert "singleton" not in result.stdout.lower()
    assert "TEST_ONLY: no submission performed" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_container_runtime_smoke_omits_unsupported_gpu_tres_for_ptyche(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        SBATCH_GPUS_PER_NODE="4",
        SBATCH_GRES="none",
    )

    assert result.returncode == 0, result.stderr
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout


def test_oci_runtime_staging_renders_cpu_only_job(tmp_path: Path) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 0, result.stderr
    assert "--gres=gpu" not in result.stdout
    assert "--gpus" not in result.stdout
    assert "--partition=cpu" in result.stdout
    assert "--cpus-per-task=32" in result.stdout
    assert "RUNTIME_STAGE_CPUS_PER_TASK=32" in result.stdout
    assert "RUNTIME_PHASE=stage" in result.stdout
    assert "TEST_ONLY: no submission performed" in result.stdout


def test_oci_runtime_staging_accepts_cpu_datamover(tmp_path: Path) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        STAGE_PARTITION="cpu_datamover",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 0, result.stderr
    assert "--partition=cpu_datamover" in result.stdout
    assert "--cpus-per-task=32" in result.stdout
    assert "--gres=gpu" not in result.stdout
    assert "--gpus" not in result.stdout


def test_runtime_staging_accepts_ptyche_batch_without_gpu_request(
    tmp_path: Path,
) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        STAGE_PARTITION="batch",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        SBATCH_GPUS_PER_NODE="4",
        SBATCH_GRES="none",
    )

    assert result.returncode == 0, result.stderr
    assert "--partition=batch" in result.stdout
    assert "--gpus-per-node" not in result.stdout
    assert "--gres=" not in result.stdout


def test_oci_runtime_staging_binds_explicit_single_cpu_request(tmp_path: Path) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CPUS_PER_TASK="1",
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 0, result.stderr
    assert "--cpus-per-task=1" in result.stdout
    assert "RUNTIME_STAGE_CPUS_PER_TASK=1" in result.stdout


@pytest.mark.parametrize("cpus_per_task", ("0", "97", "1.5", "many"))
def test_oci_runtime_staging_rejects_invalid_cpu_request(
    tmp_path: Path, cpus_per_task: str
) -> None:
    result = _run_script(
        "scripts/validate_oci_container_runtime.sub",
        TEST_ONLY="1",
        RUNTIME_PHASE="stage",
        RUNTIME_STAGE_CPUS_PER_TASK=cpus_per_task,
        RUNTIME_STAGE_CAPABILITY="mcore-test-v1",
        CONTAINER="/lustre/example/nemo_rl_nightly.sqsh",
        CONTAINER_SHA256=CONTAINER_SHA256,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
    )

    assert result.returncode == 2
    assert (
        "RUNTIME_STAGE_CPUS_PER_TASK must be an integer from 1 through 96"
        in result.stderr
    )


def test_locked_uv_sync_retries_once_with_the_same_job_local_cache(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
attempt=0
if [[ -f "${ATTEMPT_FILE}" ]]; then
  attempt=$(<"${ATTEMPT_FILE}")
fi
attempt=$((attempt + 1))
printf '%s\n' "${attempt}" >"${ATTEMPT_FILE}"
printf '%s\n' "$*" >>"${COMMAND_LOG}"
if ((attempt == 1)); then
  mkdir -p "${UV_CACHE_DIR}"
  touch "${UV_CACHE_DIR}/first-attempt"
  exit 17
fi
[[ -f "${UV_CACHE_DIR}/first-attempt" ]] || exit 91
"""
    )
    fake_uv.chmod(0o755)
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$1" >>"${SLEEP_LOG}"
"""
    )
    fake_sleep.chmod(0o755)
    attempt_file = tmp_path / "attempt"
    command_log = tmp_path / "commands"
    sleep_log = tmp_path / "sleeps"
    uv_cache = tmp_path / "uv-cache"
    shell_function = _extract_shell_function(
        "scripts/validate_oci_container_runtime.sub", "run_locked_uv_sync"
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            shell_function,
            'sync_command=("${FAKE_UV}" sync --locked --extra mcore)',
            "run_locked_uv_sync",
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_UV": str(fake_uv),
            "ATTEMPT_FILE": str(attempt_file),
            "COMMAND_LOG": str(command_log),
            "SLEEP_LOG": str(sleep_log),
            "UV_CACHE_DIR": str(uv_cache),
        }
    )

    result = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert attempt_file.read_text().splitlines() == ["2"]
    assert command_log.read_text().splitlines() == [
        "sync --locked --extra mcore",
        "sync --locked --extra mcore",
    ]
    assert sleep_log.read_text().splitlines() == ["5"]


def test_locked_uv_sync_returns_the_third_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
attempt=0
if [[ -f "${ATTEMPT_FILE}" ]]; then
  attempt=$(<"${ATTEMPT_FILE}")
fi
attempt=$((attempt + 1))
printf '%s\n' "${attempt}" >"${ATTEMPT_FILE}"
case "${attempt}" in
  1) exit 17 ;;
  2) exit 18 ;;
  3) exit 19 ;;
  *) exit 90 ;;
esac
"""
    )
    fake_uv.chmod(0o755)
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$1" >>"${SLEEP_LOG}"
"""
    )
    fake_sleep.chmod(0o755)
    attempt_file = tmp_path / "attempt"
    sleep_log = tmp_path / "sleeps"
    shell_function = _extract_shell_function(
        "scripts/validate_oci_container_runtime.sub", "run_locked_uv_sync"
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            shell_function,
            'sync_command=("${FAKE_UV}" sync --locked)',
            "run_locked_uv_sync",
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_UV": str(fake_uv),
            "ATTEMPT_FILE": str(attempt_file),
            "SLEEP_LOG": str(sleep_log),
        }
    )

    result = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 19
    assert attempt_file.read_text().splitlines() == ["3"]
    assert sleep_log.read_text().splitlines() == ["5", "10"]


def test_oci_container_runtime_smoke_uses_persistent_probe_when_spooled(
    tmp_path: Path,
) -> None:
    source_wrapper = EXPERIMENT_DIR / "scripts" / "validate_oci_container_runtime.sub"
    persistent_probe = (EXPERIMENT_DIR / "validate_container_runtime.py").resolve()
    spool_dir = tmp_path / "slurm-spool" / "job315"
    spool_dir.mkdir(parents=True)
    spooled_wrapper = spool_dir / "slurm_script"
    spooled_wrapper.write_text(source_wrapper.read_text())
    spooled_wrapper.chmod(0o755)
    container = tmp_path / "nightly.sqsh"
    container.write_bytes(b"container")
    digest = hashlib.sha256(container.read_bytes()).hexdigest()
    artifacts = tmp_path / "artifacts"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_log = tmp_path / "srun.txt"
    provenance_verifier = tmp_path / "verify_source_provenance.sh"
    provenance_verifier.write_text("#!/bin/bash\nset -euo pipefail\n")
    provenance_verifier.chmod(0o755)
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >"${SRUN_LOG}"
output=
while (($#)); do
  if [[ "$1" == "--output" ]]; then
    shift
    output=$1
  fi
  shift
done
printf '{"status":"passed"}\n' >"${output}"
"""
    )
    fake_srun.chmod(0o755)
    runtime_stage_root = artifacts / "staged-runtimes" / ("a" * 64)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SRUN_LOG": str(srun_log),
            "SLURM_JOB_ID": "315",
            "CONTAINER_RUNTIME_VALIDATOR": str(persistent_probe),
            "CONTAINER": str(container),
            "CONTAINER_SHA256": digest,
            "ARTIFACT_DIR": str(artifacts),
            "CONTAINER_PYTHON": "/fixture/python",
            "EXPECTED_NEMORL_SHA": NEMORL_SHA,
            "EXPECTED_BRIDGE_SHA": BRIDGE_SHA,
            "EXPECTED_MCORE_SHA": MCORE_SHA,
            "EXPECTED_TE_SHA": TE_SHA,
            "EXPECTED_TE_VERSION_BASE_SHA": TE_SHA,
            "SOURCE_PROVENANCE_VERIFIER": str(provenance_verifier),
            "RUNTIME_STAGE_ROOT": str(runtime_stage_root),
            "RUNTIME_STAGE_MARKER": str(
                artifacts / "stage-markers" / f"{runtime_stage_root.name}.env"
            ),
            "RUNTIME_STAGE_MARKER_SHA256": "b" * 64,
            "RUNTIME_STAGE_JOB_ID": "315",
            "RUNTIME_STAGE_CAPABILITY": "mcore-test-v1",
        }
    )

    result = subprocess.run(
        ["bash", str(spooled_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(persistent_probe) in srun_log.read_text()
    assert (
        str(spool_dir / "../validate_container_runtime.py") not in srun_log.read_text()
    )
    assert (artifacts / "oci-container-runtime-315.json").is_file()


def test_container_runtime_probe_requires_four_visible_gpus_and_packages(
    tmp_path: Path,
) -> None:
    module = _load_experiment_module("validate_container_runtime")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"GPU-{index}"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            del index
            return 10, 0

    modules = {
        name: SimpleNamespace(__file__=str(tmp_path / f"{name}.py"))
        for name in module.REQUIRED_MODULE_DISTRIBUTIONS
    }
    modules["torch"] = SimpleNamespace(
        __file__=str(tmp_path / "torch.py"),
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
    )
    modules["megatron.core.extensions.transformer_engine"] = SimpleNamespace(
        __file__=str(tmp_path / "megatron_transformer_engine.py"),
        TEColumnParallelGroupedLinear=object,
        TERowParallelGroupedLinear=object,
    )

    result = module.probe_runtime(
        expected_device_count=4,
        importer=lambda name: modules[name],
        version_getter=lambda distribution: f"fixture-{distribution}",
    )

    assert result["cuda_available"] is True
    assert result["device_count"] == 4
    assert [device["name"] for device in result["devices"]] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    assert result["transformer_engine_grouped_linear_symbols"] == [
        "TEColumnParallelGroupedLinear",
        "TERowParallelGroupedLinear",
    ]
    assert result["all_eval_callables_supported"] == "not_tested"
    assert result["mcore_eval_reuse_graph_io"] == "not_implemented"
    assert result["raw_te_eval_reuse_graph_io"] == "not_tested"
    assert result["topology"] == {
        "num_nodes": 1,
        "gpus_per_node": 4,
        "world_size": 4,
    }
    assert set(result["packages"]) == {
        "torch",
        "transformer_engine.pytorch",
        "megatron.core",
        "megatron.core.extensions.transformer_engine",
        "megatron.bridge",
        "mamba_ssm",
        "causal_conv1d",
        "cupy",
    }


def test_container_runtime_probe_requires_te_grouped_linear_backend(
    tmp_path: Path,
) -> None:
    module = _load_experiment_module("validate_container_runtime")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"GPU-{index}"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            del index
            return 10, 0

    modules = {
        name: SimpleNamespace(__file__=str(tmp_path / f"{name}.py"))
        for name in module.REQUIRED_MODULE_DISTRIBUTIONS
    }
    modules["torch"] = SimpleNamespace(
        __file__=str(tmp_path / "torch.py"),
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
    )
    modules["megatron.core.extensions.transformer_engine"] = SimpleNamespace(
        __file__=str(tmp_path / "megatron_transformer_engine.py"),
        TEColumnParallelGroupedLinear=None,
        TERowParallelGroupedLinear=object,
    )

    with pytest.raises(RuntimeError, match="TE grouped-linear backend is unavailable"):
        module.probe_runtime(
            expected_device_count=4,
            importer=lambda name: modules[name],
            version_getter=lambda distribution: f"fixture-{distribution}",
        )


def test_container_runtime_probe_rejects_wrong_gpu_count(tmp_path: Path) -> None:
    module = _load_experiment_module("validate_container_runtime")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 3

    torch_module = SimpleNamespace(
        __file__=str(tmp_path / "torch.py"),
        cuda=FakeCuda(),
    )

    try:
        module.probe_runtime(
            expected_device_count=4,
            importer=lambda name: torch_module if name == "torch" else None,
            version_getter=lambda distribution: distribution,
        )
    except RuntimeError as error:
        assert "expected exactly 4 visible CUDA devices, got 3" in str(error)
    else:
        raise AssertionError("three visible GPUs must fail the OCI runtime smoke")


def test_readme_documents_container_runtime_gate_and_artifact() -> None:
    readme = (EXPERIMENT_DIR / "README.md").read_text()
    normalized_readme = " ".join(readme.split())

    assert "scripts/validate_oci_container_runtime.sub" in readme
    assert "exactly four visible devices" in readme
    assert "machine-readable success or failure artifact" in normalized_readme
    assert "MCore TE grouped-linear symbols" in normalized_readme
    for package in (
        "PyTorch",
        "Transformer Engine",
        "Megatron Core",
        "Megatron Bridge",
        "Mamba SSM",
        "causal-conv1d",
        "CuPy",
    ):
        assert package in normalized_readme


def _complete_result_record(*, model: str, step: int) -> dict[str, object]:
    return {
        "model": model,
        "dispatcher": "alltoall",
        "scope": "attn,moe_router",
        "status": "passed",
        "mode": "nemorl",
        "cluster": "oci-hsg",
        "profile": "oci-hsg-gb200",
        "phase": "performance",
        "steps": 20,
        "step": step,
        "job_id": f"job-{model}",
        "nemo_rl_commit": "1" * 40,
        "bridge_commit": "2" * 40,
        "mcore_commit": "3" * 40,
        "te_commit": "4" * 40,
        "te_version": "2.16.0.dev0",
        "container_sha256": "5" * 64,
        "metrics": {
            "timing/train/total_step_time": float(step),
            "timing/train/generation": float(step + 1),
            "timing/train/policy_training": float(step + 2),
            "timing/train/policy_and_reference_logprobs": float(step + 3),
            "performance/tokens_per_sec_per_gpu": float(1000 + step),
            "performance/generation_tokens_per_sec_per_gpu": float(2000 + step),
            "performance/policy_training_tokens_per_sec_per_gpu": float(3000 + step),
            "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": float(
                4000 + step
            ),
            "cuda_graph/graph_calls": 75,
            "cuda_graph/eligible_calls": 100,
            "cuda_graph/coverage": 0.75,
            "cuda_graph/logical_tokens": 80,
            "cuda_graph/padded_tokens": 96,
            "cuda_graph/capacity_tokens": 128,
            "cuda_graph/capacity_utilization": 0.625,
            "cuda_graph/padding_utilization": 0.833333,
            "train/reward": 0.8,
            "train/loss": 0.2,
            "train/gen_kl_error": 0.01,
            "train/token_mult_prob_error": 0.02,
            "train/grad_norm": 1.5,
            "correctness/router_topk_parity": True,
            "correctness/expert_count_parity": True,
            "correctness/nan_inf_status": "clear",
        },
    }


def test_collector_normalizes_full_current_result_schema(tmp_path: Path) -> None:
    collector = _load_experiment_module("collect_results")
    row = collector.normalize_record(_complete_result_record(model="nano", step=6))

    assert set(collector.REQUIRED_REPORT_FIELDS) <= set(collector.CSV_FIELDS)
    assert row["model"] == "nano"
    assert row["profile"] == "oci-hsg-gb200"
    assert row["e2e_step_time"] == 6.0
    assert row["logprob_tokens_per_sec_per_gpu"] == 4006.0
    assert row["graph_calls"] == 75
    assert row["eligible_calls"] == 100
    assert row["graph_coverage"] == 0.75
    assert row["logical_tokens"] == 80
    assert row["capacity_utilization"] == 0.625
    assert row["reward"] == 0.8
    assert row["policy_loss"] == 0.2
    assert row["gen_kl_error"] == 0.01
    assert row["token_mult_prob_error"] == 0.02
    assert row["router_topk_parity"] is True
    assert row["expert_count_parity"] is True
    assert row["nan_inf_status"] == "clear"
    assert row["te_version"] == "2.16.0.dev0"

    output_json = tmp_path / "results.json"
    output_csv = tmp_path / "results.csv"
    collector.write_results([row], output_json=output_json, output_csv=output_csv)
    payload = json.loads(output_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["fields"] == list(collector.CSV_FIELDS)
    assert payload["rows"] == [row]
    assert output_csv.read_text().splitlines()[0].split(",") == list(
        collector.CSV_FIELDS
    )


def test_collector_preserves_failures_without_metric_rows() -> None:
    collector = _load_experiment_module("collect_results")

    row = collector.normalize_record(
        {
            "model": "ultra",
            "dispatcher": "flex",
            "scope": "attn",
            "status": "failed",
            "failure": "CUDA out of memory",
            "exit_code": 1,
            "mode": "nemorl",
            "cluster": "oci-hsg",
            "profile": "oci-hsg-gb200",
            "phase": "smoke",
            "steps": 5,
            "job_id": "failed-job",
        }
    )

    assert row["status"] == "failed"
    assert row["failure"] == "CUDA out of memory"
    assert row["exit_code"] == 1
    assert row["e2e_step_time"] == ""


def test_steady_state_rows_exclude_three_warmups_and_capture_window() -> None:
    collector = _load_experiment_module("collect_results")
    rows = [
        collector.normalize_record(_complete_result_record(model="nano", step=step))
        for step in range(1, 21)
    ]

    steady = collector.steady_state_rows(rows)

    assert [row["step"] for row in steady] == list(range(6, 21))


def test_report_is_multi_model_escaped_and_separates_coverage_definitions() -> None:
    collector = _load_experiment_module("collect_results")
    renderer = _load_experiment_module("render_report")
    rows = [
        collector.normalize_record(_complete_result_record(model="nano", step=6)),
        collector.normalize_record(_complete_result_record(model="super", step=6)),
        collector.normalize_record(
            {
                "model": "<script>alert(1)</script>",
                "dispatcher": "flex",
                "scope": "attn",
                "status": "failed",
                "failure": "OOM <node>",
                "mode": "nemorl",
                "cluster": "oci-hsg",
                "profile": "oci-hsg-gb200",
                "phase": "smoke",
                "steps": 5,
                "job_id": "failed-job",
            }
        ),
    ]
    nsys = {
        "nano-attn": {
            "nsys_profile_count": 4,
            "nsys_profiles_with_cuda_graph_launches": 4,
            "nsys_cuda_graph_launch_share_of_cuda_api_calls_pct": 12.5,
        }
    }

    report = renderer.render_html(rows, nsys_coverage=nsys)

    assert "nano" in report
    assert "super" in report
    assert report.count("<h2>") == 4
    assert "Runtime coverage and correctness" in report
    assert "75.00%" in report
    assert "Nsight CUDA API launch share" in report
    assert "12.50%" in report
    assert "Failure" in report
    assert "OOM &lt;node&gt;" in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<script>alert(1)</script>" not in report
    assert "https://" not in report
