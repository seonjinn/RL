from __future__ import annotations

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread

import pytest
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(name: str) -> dict[str, object]:
    with (EXPERIMENT_ROOT / name).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def _load_preflight():
    path = EXPERIMENT_ROOT / "preflight.py"
    spec = importlib.util.spec_from_file_location("dflash2_static_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_launcher():
    path = EXPERIMENT_ROOT / "launch.py"
    spec = importlib.util.spec_from_file_location("dflash2_static_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_image_contract_module():
    path = EXPERIMENT_ROOT / "image_contract.py"
    spec = importlib.util.spec_from_file_location("dflash2_image_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_image_contract_pins_a_post_merge_multiarch_image() -> None:
    contract = json.loads((EXPERIMENT_ROOT / "image_contract.json").read_text())

    assert contract == {
        "schema_version": 1,
        "source_image": (
            "docker.io/vllm/vllm-openai:"
            "nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9"
        ),
        "source_commit": "f94666b60d4c58ec0807d22c837cfae322a1dde9",
        "source_index_digest": (
            "sha256:f50b406f696712019a673e317a0db6e029c430cf81ec7bdea2ebd7111e55aef7"
        ),
        "source_arm64_digest": (
            "sha256:4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6"
        ),
        "dflash2_merge_ancestor": "b389ac29465b33f9e9c534df221ea3c129e9793f",
        "required_platforms": ["linux/arm64", "linux/amd64"],
    }


def test_enroot_uri_uses_the_docker_hub_registry_api_host() -> None:
    image_contract = _load_image_contract_module()

    assert image_contract.enroot_image_uri(
        "docker.io/vllm/vllm-openai:nightly-f946"
    ) == ("docker://registry-1.docker.io#vllm/vllm-openai:nightly-f946")
    assert (
        image_contract.enroot_image_uri("nvcr.io/nvidian/nemo-rl:nightly")
        == "docker://nvcr.io#nvidian/nemo-rl:nightly"
    )


def test_registry_index_validation_requires_exact_digest_and_arm64_manifest() -> None:
    image_contract = _load_image_contract_module()
    contract = image_contract.load_contract(EXPERIMENT_ROOT / "image_contract.json")
    registry_index = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
        "manifests": [
            {
                "digest": contract["source_arm64_digest"],
                "platform": {"os": "linux", "architecture": "arm64"},
            },
            {
                "digest": "sha256:" + "5" * 64,
                "platform": {"os": "linux", "architecture": "amd64"},
            },
        ],
    }

    image_contract.validate_registry_index(
        contract,
        registry_digest=contract["source_index_digest"],
        registry_index=registry_index,
    )

    with pytest.raises(ValueError, match="index digest"):
        image_contract.validate_registry_index(
            contract,
            registry_digest="sha256:" + "0" * 64,
            registry_index=registry_index,
        )

    registry_index["manifests"][0]["digest"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="ARM64 digest"):
        image_contract.validate_registry_index(
            contract,
            registry_digest=contract["source_index_digest"],
            registry_index=registry_index,
        )


def test_image_metadata_rejects_downgrade_or_extra_contract_fields(
    tmp_path: Path,
) -> None:
    image_contract = _load_image_contract_module()
    contract = image_contract.load_contract(EXPERIMENT_ROOT / "image_contract.json")
    metadata = {
        "source_image": contract["source_image"],
        "source_commit": contract["source_commit"],
        "source_index_digest": contract["source_index_digest"],
        "source_arm64_digest": contract["source_arm64_digest"],
        "dflash2_merge_ancestor": contract["dflash2_merge_ancestor"],
        "platform": "linux/arm64",
        "sha256": "a" * 64,
    }
    metadata_path = tmp_path / "image.sqsh.metadata.txt"
    metadata_path.write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items())
    )
    image_contract.validate_metadata(contract, metadata_path)

    metadata["source_commit"] = "d626108b1841888ec90aced33367149a6bbc7e4b"
    metadata_path.write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items())
    )
    with pytest.raises(ValueError, match="source_commit"):
        image_contract.validate_metadata(contract, metadata_path)

    metadata["source_commit"] = contract["source_commit"]
    metadata["unexpected"] = "downgrade"
    metadata_path.write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items())
    )
    with pytest.raises(ValueError, match="metadata fields"):
        image_contract.validate_metadata(contract, metadata_path)


def test_arm64_image_config_requires_exact_official_source_labels() -> None:
    image_contract = _load_image_contract_module()
    contract = image_contract.load_contract(EXPERIMENT_ROOT / "image_contract.json")
    labels = {
        "ai.vllm.build.commit": contract["source_commit"],
        "org.opencontainers.image.revision": contract["source_commit"],
        "org.opencontainers.image.source": "https://github.com/vllm-project/vllm",
    }
    config = {
        "architecture": "arm64",
        "os": "linux",
        "config": {"Labels": labels},
    }

    image_contract.validate_image_config(contract, config)
    labels["ai.vllm.build.commit"] = "d626108b1841888ec90aced33367149a6bbc7e4b"
    with pytest.raises(ValueError, match="build.commit"):
        image_contract.validate_image_config(contract, config)


def test_runtime_environment_requires_the_exact_staged_image_contract() -> None:
    preflight = _load_preflight()
    environment = {
        "NRL_VLLM_SOURCE_COMMIT": "f94666b60d4c58ec0807d22c837cfae322a1dde9",
        "NRL_VLLM_SOURCE_INDEX_DIGEST": (
            "sha256:f50b406f696712019a673e317a0db6e029c430cf81ec7bdea2ebd7111e55aef7"
        ),
        "NRL_VLLM_SOURCE_ARM64_DIGEST": (
            "sha256:4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6"
        ),
        "NRL_DFLASH2_MERGE_ANCESTOR": ("b389ac29465b33f9e9c534df221ea3c129e9793f"),
    }

    assert preflight.validate_image_contract_environment(environment) == environment

    environment["NRL_VLLM_SOURCE_COMMIT"] = "d626108b1841888ec90aced33367149a6bbc7e4b"
    with pytest.raises(RuntimeError, match="image contract"):
        preflight.validate_image_contract_environment(environment)


def test_baseline_and_dflash2_hold_target_and_workload_constant() -> None:
    baseline = _load_yaml("baseline.yaml")
    dflash2 = _load_yaml("dflash2.yaml")

    baseline_without_identity = {
        key: value
        for key, value in baseline.items()
        if key not in {"name", "speculative_config"}
    }
    dflash2_without_identity = {
        key: value
        for key, value in dflash2.items()
        if key not in {"name", "speculative_config"}
    }

    assert baseline_without_identity == dflash2_without_identity
    assert baseline["model"] == "Qwen/Qwen3.8-27B"
    assert baseline["model_revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert baseline["workload"]["requests"] == 20
    assert "speculative_config" not in baseline


def test_dflash2_arm_uses_the_published_static_vllm_contract() -> None:
    config = _load_yaml("dflash2.yaml")

    assert config["speculative_config"] == {
        "method": "dflash",
        "model": "incoai/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
        "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    }
    assert config["mode"] == "static_rollout"
    assert config["draft_refit"] is False
    assert config["online_draft_training"] is False


def test_preflight_rejects_the_current_nemo_rl_vllm_pin() -> None:
    preflight = _load_preflight()

    with pytest.raises(RuntimeError, match="DFlash2-capable vLLM"):
        preflight.validate_runtime(
            vllm_version="0.25.1",
            has_dflash2_capability=False,
            uses_v2_runner=False,
        )


def test_preflight_accepts_a_capable_v2_runtime() -> None:
    preflight = _load_preflight()

    preflight.validate_runtime(
        vllm_version="0.26.0.dev0",
        has_dflash2_capability=True,
        uses_v2_runner=True,
    )


def test_preflight_rejects_dflash2_when_v2_is_not_explicitly_enabled() -> None:
    preflight = _load_preflight()

    with pytest.raises(RuntimeError, match="V2 model runner"):
        preflight.validate_runtime(
            vllm_version="0.27.0.dev0+gb389ac294",
            has_dflash2_capability=True,
            uses_v2_runner=False,
        )


def test_launcher_builds_identical_non_spec_server_commands() -> None:
    launcher = _load_launcher()
    baseline = launcher.load_manifest(EXPERIMENT_ROOT / "baseline.yaml")
    dflash2 = launcher.load_manifest(EXPERIMENT_ROOT / "dflash2.yaml")

    baseline_command = launcher.build_server_command(baseline, port=8123)
    dflash2_command = launcher.build_server_command(dflash2, port=8123)

    assert baseline_command == [
        "vllm",
        "serve",
        "Qwen/Qwen3.8-27B",
        "--revision",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "--served-model-name",
        "qwen3.8-27b-static",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.9",
    ]
    assert dflash2_command[:-2] == baseline_command
    assert dflash2_command[-2] == "--speculative-config"
    assert json.loads(dflash2_command[-1]) == {
        "method": "dflash",
        "model": "incoai/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
        "revision": "dedf8df68adfb1afeaf7b7480c0a0243108177b4",
    }


def test_launcher_manifest_metadata_is_json_serializable() -> None:
    launcher = _load_launcher()
    manifest = launcher.load_manifest(EXPERIMENT_ROOT / "dflash2.yaml")

    encoded = json.dumps(launcher.manifest_metadata(manifest), sort_keys=True)

    assert '"arm": "dflash2"' in encoded
    assert '"source": "' in encoded
    assert '"sha256": "' in encoded


def test_preflight_failure_writes_a_machine_readable_failure_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    manifest = launcher.load_manifest(EXPERIMENT_ROOT / "dflash2.yaml")

    def fail_preflight(_experiment_root: Path) -> dict[str, object]:
        raise RuntimeError("DFlash2 runtime unavailable")

    monkeypatch.setattr(launcher, "_runtime_fingerprint", fail_preflight)
    output_dir = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        launcher.run_experiment(
            manifest,
            output_dir=output_dir,
            port=8123,
            request_count=20,
        )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["requested_requests"] == 20
    assert summary["completed_requests"] == 0
    assert summary["error"] == {
        "type": "RuntimeError",
        "message": "DFlash2 runtime unavailable",
    }


@pytest.mark.parametrize("request_count", [0, 21])
def test_launcher_rejects_request_counts_outside_the_twenty_request_cap(
    request_count: int,
) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="between 1 and 20"):
        launcher.validate_request_count(request_count)


class _SmokeHandler(BaseHTTPRequestHandler):
    payloads: list[dict[str, object]] = []
    payload_lock = Lock()

    def do_POST(self) -> None:
        assert self.path == "/v1/chat/completions"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        with self.payload_lock:
            self.payloads.append(payload)
        body = json.dumps(
            {
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.8-27b-static",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "42"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_smoke_client_executes_exactly_twenty_requests_and_emits_summary() -> None:
    launcher = _load_launcher()
    _SmokeHandler.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHandler)
    server_thread = Thread(target=server.serve_forever)
    server_thread.start()
    try:
        manifest = launcher.load_manifest(EXPERIMENT_ROOT / "baseline.yaml")
        summary = launcher.benchmark_server(
            manifest,
            base_url=f"http://127.0.0.1:{server.server_port}",
            request_count=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert len(_SmokeHandler.payloads) == 20
    assert [payload["seed"] for payload in _SmokeHandler.payloads] == list(
        range(1000, 1020)
    )
    assert summary["schema_version"] == 1
    assert summary["status"] == "passed"
    assert summary["requested_requests"] == 20
    assert summary["completed_requests"] == 20
    assert summary["failed_requests"] == 0
    assert summary["tokens"] == {
        "prompt": 200,
        "completion": 40,
        "total": 240,
    }
    assert len(summary["requests"]) == 20


@pytest.mark.parametrize("arm", ["baseline", "dflash2"])
def test_slurm_harness_dry_run_is_executable_without_submitting(arm: str) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "ARM": arm,
            "CONTAINER_IMAGE": (
                "/lustre/user/containers/vllm-openai-nightly-f94666b_20260824_123.sqsh"
            ),
            "REPO_ROOT": "/home/user/Nemo-RL",
            "OUTPUT_ROOT": "/lustre/user/results/dflash2-smoke",
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_ROOT / "run.slurm"), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "srun" in result.stdout
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in result.stdout
    assert "NRL_VLLM_SOURCE_COMMIT" in result.stdout
    assert "f94666b60d4c58ec0807d22c837cfae322a1dde9" in result.stdout
    assert "NRL_VLLM_SOURCE_INDEX_DIGEST" in result.stdout
    assert (
        "f50b406f696712019a673e317a0db6e029c430cf81ec7bdea2ebd7111e55aef7"
        in result.stdout
    )
    assert "NRL_VLLM_SOURCE_ARM64_DIGEST" in result.stdout
    assert (
        "4db6d42b66ad393faa3da7341db580f443b7aeb9a7de5597cd11b724eabff6f6"
        in result.stdout
    )
    assert f"{arm}.yaml" in result.stdout
    assert "--request-count\\ 20" in result.stdout
    assert (
        "--container-image=/lustre/user/containers/"
        "vllm-openai-nightly-f94666b_20260824_123.sqsh" in result.stdout
    )


def test_stage_script_rejects_an_image_contract_downgrade_before_staging(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CONTAINER_DIR": str(tmp_path),
            "REPO_ROOT": str(EXPERIMENT_ROOT.parents[1]),
            "SLURM_JOB_ID": "123",
            "SOURCE_IMAGE": (
                "docker.io/vllm/vllm-openai:"
                "nightly-d626108b1841888ec90aced33367149a6bbc7e4b"
            ),
        }
    )

    result = subprocess.run(
        ["bash", str(EXPERIMENT_ROOT / "stage_enroot_image.sbatch")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SOURCE_IMAGE does not match the pinned image contract" in result.stderr
