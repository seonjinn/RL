#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import cast
from urllib.request import Request, urlopen


_EXPECTED_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "source_image": (
        "docker.io/vllm/vllm-openai:nightly-f94666b60d4c58ec0807d22c837cfae322a1dde9"
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
_METADATA_CONTRACT_KEYS = frozenset(
    {
        "source_image",
        "source_commit",
        "source_index_digest",
        "source_arm64_digest",
        "dflash2_merge_ancestor",
    }
)
_METADATA_KEYS = _METADATA_CONTRACT_KEYS | {"platform", "sha256"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def load_contract(path: Path) -> dict[str, object]:
    values = dict(_mapping(json.loads(path.read_text()), name="image contract"))
    if values != _EXPECTED_CONTRACT:
        raise ValueError("image contract does not match the published pinned contract")
    return values


def enroot_image_uri(source_image: str) -> str:
    registry, separator, repository_and_tag = source_image.partition("/")
    if not separator or not repository_and_tag:
        raise ValueError("source image must include registry and repository")
    enroot_registry = "registry-1.docker.io" if registry == "docker.io" else registry
    return f"docker://{enroot_registry}#{repository_and_tag}"


def validate_registry_index(
    contract: Mapping[str, object],
    *,
    registry_digest: object,
    registry_index: object,
) -> None:
    expected_index_digest = _string(
        contract.get("source_index_digest"), name="source_index_digest"
    )
    if registry_digest != expected_index_digest:
        raise ValueError(
            f"registry index digest mismatch: {registry_digest!r} != "
            f"{expected_index_digest!r}"
        )
    index = _mapping(registry_index, name="registry index")
    manifests = index.get("manifests")
    if not isinstance(manifests, list):
        raise ValueError("registry index manifests must be a list")

    platform_digests: dict[str, str] = {}
    for position, raw_manifest in enumerate(manifests):
        manifest = _mapping(raw_manifest, name=f"manifest {position}")
        platform = _mapping(manifest.get("platform"), name=f"platform {position}")
        platform_name = (
            f"{_string(platform.get('os'), name='platform os')}/"
            f"{_string(platform.get('architecture'), name='platform architecture')}"
        )
        if platform_name in platform_digests:
            raise ValueError(f"duplicate registry platform: {platform_name}")
        platform_digests[platform_name] = _string(
            manifest.get("digest"), name=f"digest for {platform_name}"
        )

    required_platforms = contract.get("required_platforms")
    if not isinstance(required_platforms, list) or not all(
        isinstance(platform, str) for platform in required_platforms
    ):
        raise ValueError("required_platforms must be a list of strings")
    missing_platforms = set(required_platforms) - platform_digests.keys()
    if missing_platforms:
        raise ValueError(
            f"registry index lacks required platforms: {missing_platforms}"
        )
    expected_arm64_digest = _string(
        contract.get("source_arm64_digest"), name="source_arm64_digest"
    )
    if platform_digests["linux/arm64"] != expected_arm64_digest:
        raise ValueError("registry ARM64 digest does not match the pinned contract")


def validate_image_config(contract: Mapping[str, object], image_config: object) -> None:
    config = _mapping(image_config, name="image config")
    if config.get("architecture") != "arm64" or config.get("os") != "linux":
        raise ValueError("image config is not linux/arm64")
    runtime_config = _mapping(config.get("config"), name="runtime config")
    labels = _mapping(runtime_config.get("Labels"), name="image labels")
    expected_commit = contract.get("source_commit")
    for label in ("ai.vllm.build.commit", "org.opencontainers.image.revision"):
        if labels.get(label) != expected_commit:
            raise ValueError(f"image label {label} does not match source_commit")
    if labels.get("org.opencontainers.image.source") != (
        "https://github.com/vllm-project/vllm"
    ):
        raise ValueError("image source label is not the official vLLM repository")


def _request_json(request: Request) -> tuple[object, Mapping[str, str]]:
    with urlopen(request, timeout=30.0) as response:  # noqa: S310
        return json.loads(response.read()), response.headers


def verify_registry(contract: Mapping[str, object]) -> dict[str, object]:
    source_image = _string(contract.get("source_image"), name="source_image")
    prefix = "docker.io/vllm/vllm-openai:"
    if not source_image.startswith(prefix):
        raise ValueError("source_image is not the official vLLM Docker Hub repository")
    tag = source_image.removeprefix(prefix)
    token_values, _ = _request_json(
        Request(
            "https://auth.docker.io/token?"
            "service=registry.docker.io&scope=repository:vllm/vllm-openai:pull"
        )
    )
    token = _string(
        _mapping(token_values, name="registry token").get("token"), name="token"
    )
    authorization = {"Authorization": f"Bearer {token}"}
    index_request = Request(
        f"https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/{tag}",
        headers={
            **authorization,
            "Accept": "application/vnd.docker.distribution.manifest.list.v2+json",
        },
    )
    registry_index, index_headers = _request_json(index_request)
    registry_digest = index_headers.get("Docker-Content-Digest")
    validate_registry_index(
        contract,
        registry_digest=registry_digest,
        registry_index=registry_index,
    )

    arm64_digest = _string(
        contract.get("source_arm64_digest"), name="source_arm64_digest"
    )
    arm_manifest, _ = _request_json(
        Request(
            "https://registry-1.docker.io/v2/vllm/vllm-openai/"
            f"manifests/{arm64_digest}",
            headers={
                **authorization,
                "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            },
        )
    )
    arm_manifest_values = _mapping(arm_manifest, name="ARM64 manifest")
    config_descriptor = _mapping(
        arm_manifest_values.get("config"), name="config descriptor"
    )
    config_digest = _string(config_descriptor.get("digest"), name="config digest")
    image_config, _ = _request_json(
        Request(
            f"https://registry-1.docker.io/v2/vllm/vllm-openai/blobs/{config_digest}",
            headers=authorization,
        )
    )
    validate_image_config(contract, image_config)
    return {
        "source_image": source_image,
        "source_commit": contract["source_commit"],
        "source_index_digest": registry_digest,
        "source_arm64_digest": arm64_digest,
        "arm64_config_digest": config_digest,
        "verified": True,
    }


def _parse_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid metadata line {line_number}")
        values[key] = value
    return values


def validate_metadata(
    contract: Mapping[str, object], metadata_path: Path
) -> dict[str, str]:
    metadata = _parse_metadata(metadata_path)
    if metadata.keys() != _METADATA_KEYS:
        raise ValueError(f"metadata fields must be exactly {sorted(_METADATA_KEYS)}")
    for key in _METADATA_CONTRACT_KEYS:
        if metadata[key] != contract.get(key):
            raise ValueError(f"metadata {key} does not match image contract")
    if metadata["platform"] != "linux/arm64":
        raise ValueError("metadata platform must be linux/arm64")
    if _SHA256_PATTERN.fullmatch(metadata["sha256"]) is None:
        raise ValueError("metadata sha256 must be 64 lowercase hexadecimal characters")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("image_contract.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("field", choices=sorted(_EXPECTED_CONTRACT))
    subparsers.add_parser("verify-registry")
    subparsers.add_parser("enroot-uri")
    metadata_parser = subparsers.add_parser("validate-metadata")
    metadata_parser.add_argument("metadata", type=Path)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    if args.command == "get":
        value = contract[args.field]
        print(json.dumps(value) if isinstance(value, list) else value)
    elif args.command == "verify-registry":
        print(json.dumps(verify_registry(contract), sort_keys=True))
    elif args.command == "enroot-uri":
        print(enroot_image_uri(_string(contract["source_image"], name="source_image")))
    else:
        print(json.dumps(validate_metadata(contract, args.metadata), sort_keys=True))


if __name__ == "__main__":
    main()
