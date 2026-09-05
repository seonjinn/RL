from __future__ import annotations

import errno
import hashlib
import http.client
import io
import json
import os
import shutil
import traceback
import urllib.error
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, cast

import pytest

import tools.stage_precision_policy_source_metadata as metadata_stager
from tools.capture_precision_policy_source_evidence import (
    CheckpointArtifactSpec,
    checkpoint_artifact_specs,
)
from tools.stage_precision_policy_source_metadata import (
    EXPECTED_STAGED_METADATA_MANIFEST_SHA256,
    MAX_CONFIG_BYTES,
    MAX_INDEX_BYTES,
    MAX_SAFETENSORS_HEADER_BYTES,
    MAX_SAFETENSORS_FILE_BYTES,
    HttpResponse,
    HttpResponseHeaderPolicy,
    MetadataStagingError,
    UrllibHttpTransport,
    download_safetensors_header,
    expected_staged_metadata_manifest,
    stage_checkpoint_metadata,
    stage_source_metadata,
)


def test_pinned_metadata_manifest_has_exact_content_address() -> None:
    manifest = expected_staged_metadata_manifest()

    assert len(manifest.splitlines()) == 19
    assert manifest.splitlines() == sorted(manifest.splitlines())
    assert (
        hashlib.sha256(manifest).hexdigest()
        == EXPECTED_STAGED_METADATA_MANIFEST_SHA256
        == "d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b"
    )


def test_metadata_stager_consumes_capture_tools_public_checkpoint_pins() -> None:
    assert tuple(spec.artifact_id for spec in checkpoint_artifact_specs()) == (
        "qwen3_bf16",
        "kimi_k2",
        "kimi_k25",
        "kimi_k3",
        "nemotron_lightning_nvfp4",
        "qwen_a95b_fp8",
    )


class ScriptedTransport:
    def __init__(
        self,
        responses: Mapping[tuple[str, str | None], HttpResponse],
    ) -> None:
        self.responses: dict[tuple[str, str | None], HttpResponse] = dict(responses)
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.max_body_bytes: list[int] = []
        self.response_header_policies: list[HttpResponseHeaderPolicy | None] = []

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_body_bytes: int,
        response_header_policy: HttpResponseHeaderPolicy | None = None,
    ) -> HttpResponse:
        normalized_headers = dict(headers)
        self.requests.append((url, normalized_headers))
        self.max_body_bytes.append(max_body_bytes)
        self.response_header_policies.append(response_header_policy)
        key = (url, normalized_headers.get("Range"))
        try:
            return self.responses[key]
        except KeyError as error:
            raise AssertionError(f"unexpected HTTP request: {key}") from error


class FakeHttpStream:
    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str] | FakeHttpHeaders,
        body: bytes,
        forbid_read: bool = False,
        read_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.headers = headers
        self.body = body
        self.forbid_read = forbid_read
        self.read_error = read_error
        self.read_sizes: list[int] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        if self.forbid_read:
            raise AssertionError("transport read a response that can contain payload")
        self.read_sizes.append(size)
        if self.read_error is not None:
            raise self.read_error
        return self.body[:size]

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, stream: FakeHttpStream) -> None:
        self.stream = stream
        self.requests: list[object] = []

    def open(self, request: object, *, timeout: float) -> FakeHttpStream:
        del timeout
        self.requests.append(request)
        return self.stream


class SequenceOpener:
    def __init__(self, streams: tuple[FakeHttpStream, ...]) -> None:
        self.streams = streams
        self.requests: list[object] = []

    def open(self, request: object, *, timeout: float) -> FakeHttpStream:
        del timeout
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.streams):
            raise AssertionError("HTTP opener received an unexpected extra request")
        return self.streams[index]


class FakeHttpHeaders:
    def __init__(self, pairs: tuple[tuple[str, str], ...]) -> None:
        self._pairs = pairs

    def items(self) -> tuple[tuple[str, str], ...]:
        return self._pairs


class RaisingOpener:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def open(self, request: object, *, timeout: float) -> FakeHttpStream:
        del request, timeout
        raise self.error


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _response(
    status_code: int,
    body: bytes,
    *,
    content_range: str | None = None,
    revision: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    headers = {"Content-Length": str(len(body))}
    if content_range is not None:
        headers["Content-Range"] = content_range
    if revision is not None:
        headers["X-Repo-Commit"] = revision
    if extra_headers is not None:
        headers.update(extra_headers)
    return HttpResponse(status_code=status_code, headers=headers, body=body)


def _miniature_checkpoint(
    *,
    repository: str = "example/model",
    revision: str = "0123456789abcdef0123456789abcdef01234567",
    indexed_shard: str = "model-00001-of-00001.safetensors",
    raw_header_tensor_name: str | None = None,
) -> tuple[
    CheckpointArtifactSpec,
    dict[tuple[str, str | None], HttpResponse],
    bytes,
]:
    config = _json_bytes({"hidden_size": 2, "model_type": "example"})
    tensor_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    header_tensor_name = raw_header_tensor_name or tensor_name
    index = _json_bytes({"weight_map": {tensor_name: indexed_shard}})
    raw_header = _json_bytes(
        {
            header_tensor_name: {
                "data_offsets": [0, 8],
                "dtype": "BF16",
                "shape": [2, 2],
            }
        }
    )
    raw_header += b" " * (-len(raw_header) % 8)
    header_manifest = {
        header_tensor_name: {
            "dtype": "BF16",
            "shape": [2, 2],
            "shard": indexed_shard,
        }
    }
    artifact = {
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "header_manifest_sha256": hashlib.sha256(
            _json_bytes(header_manifest)
        ).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "kind": "immutable_hf_metadata",
        "repository": repository,
        "revision": revision,
        "shards": 1,
        "tensors": 1,
    }
    spec = CheckpointArtifactSpec(
        artifact_id="example",
        artifact=artifact,
        observations=(),
    )
    base_url = f"https://huggingface.co/{repository}/resolve/{revision}"
    shard_total = 8 + len(raw_header) + 8
    responses: dict[tuple[str, str | None], HttpResponse] = {
        (f"{base_url}/config.json", None): _response(200, config, revision=revision),
        (f"{base_url}/model.safetensors.index.json", None): _response(
            200, index, revision=revision
        ),
        (f"{base_url}/{indexed_shard}", "bytes=0-7"): _response(
            206,
            len(raw_header).to_bytes(8, "little"),
            content_range=f"bytes 0-7/{shard_total}",
            revision=revision,
        ),
        (
            f"{base_url}/{indexed_shard}",
            f"bytes=8-{7 + len(raw_header)}",
        ): _response(
            206,
            raw_header,
            content_range=f"bytes 8-{7 + len(raw_header)}/{shard_total}",
            revision=revision,
        ),
    }
    return spec, responses, raw_header


def test_checkpoint_stager_reads_only_safetensors_header_ranges(
    tmp_path: Path,
) -> None:
    spec, responses, raw_header = _miniature_checkpoint()
    transport = ScriptedTransport(responses)
    artifact_root = tmp_path / "example"

    staged = stage_checkpoint_metadata(spec, artifact_root, transport=transport)

    manifest = json.loads(
        (artifact_root / "safetensors_header_manifest.json").read_bytes()
    )
    assert manifest == {
        "model.layers.0.mlp.experts.0.gate_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "shard": "model-00001-of-00001.safetensors",
        }
    }
    assert {item.path.name for item in staged} == {
        "config.json",
        "model.safetensors.index.json",
        "safetensors_header_manifest.json",
    }
    ranges = [headers.get("Range") for _, headers in transport.requests]
    assert ranges == [None, None, "bytes=0-7", f"bytes=8-{7 + len(raw_header)}"]
    assert all(
        request_range is None
        or int(request_range.rsplit("-", maxsplit=1)[1]) <= 7 + len(raw_header)
        for request_range in ranges
    )


def test_stdlib_transport_never_reads_body_when_server_ignores_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeHttpStream(
        status=200,
        headers={"Content-Length": str(4 * 1024 * 1024 * 1024)},
        body=b"",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    response = transport.request(
        "https://huggingface.co/example/model/resolve/"
        + "0" * 40
        + "/model.safetensors",
        headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"},
        max_body_bytes=8,
    )

    assert response.status_code == 200
    assert response.body == b""
    assert stream.read_sizes == []
    assert stream.closed


def test_stdlib_transport_reads_exact_requested_range_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeHttpStream(
        status=206,
        headers={"Content-Length": "8", "Content-Range": "bytes 0-7/128"},
        body=b"12345678payload-must-not-be-read",
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    response = transport.request(
        "https://huggingface.co/example/model/resolve/"
        + "0" * 40
        + "/model.safetensors",
        headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"},
        max_body_bytes=8,
    )

    assert response.body == b"12345678"
    assert stream.read_sizes == [8]


@pytest.mark.parametrize("invalid_bound", (False, 0, 1.5, "8"))
def test_stdlib_transport_rejects_non_integer_or_non_positive_body_bound(
    monkeypatch: pytest.MonkeyPatch,
    invalid_bound: object,
) -> None:
    transport = UrllibHttpTransport()
    monkeypatch.setattr(
        transport,
        "_opener",
        RaisingOpener(AssertionError("invalid bound reached HTTP opener")),
    )

    with pytest.raises(ValueError, match="positive integer"):
        transport.request(
            "https://huggingface.co/example/model/resolve/" + "0" * 40 + "/config.json",
            headers={},
            max_body_bytes=cast(int, invalid_bound),
        )


@pytest.mark.parametrize(
    "content_length",
    (str(MAX_CONFIG_BYTES + 1), "9" * 5000),
    ids=("over-config-limit", "absurd-decimal"),
)
def test_config_download_rejects_oversized_length_before_body_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_length: str,
) -> None:
    spec, _, _ = _miniature_checkpoint()
    revision = cast(str, spec.artifact["revision"])
    stream = FakeHttpStream(
        status=200,
        headers={
            "Content-Length": content_length,
            "X-Repo-Commit": revision,
        },
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    with pytest.raises(MetadataStagingError, match="Content-Length"):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=transport,
        )

    assert stream.read_sizes == []
    assert stream.closed


def test_index_download_rejects_oversized_length_before_body_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    base_url = f"https://huggingface.co/{repository}/resolve/{revision}"
    config_body = responses[(f"{base_url}/config.json", None)].body
    config_stream = FakeHttpStream(
        status=200,
        headers={
            "Content-Length": str(len(config_body)),
            "X-Repo-Commit": revision,
        },
        body=config_body,
    )
    index_stream = FakeHttpStream(
        status=200,
        headers={
            "Content-Length": str(MAX_INDEX_BYTES + 1),
            "X-Repo-Commit": revision,
        },
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    opener = SequenceOpener((config_stream, index_stream))
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", opener)

    with pytest.raises(MetadataStagingError, match="Content-Length"):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=transport,
        )

    assert config_stream.read_sizes == [len(config_body)]
    assert config_stream.closed
    assert index_stream.read_sizes == []
    assert index_stream.closed
    assert len(opener.requests) == 2


@pytest.mark.parametrize("filename", ("config.json", "model.safetensors.index.json"))
@pytest.mark.parametrize(
    ("invalid_header", "invalid_value", "message"),
    (
        ("X-Repo-Commit", "f" * 40, "X-Repo-Commit"),
        ("Content-Encoding", "gzip", "identity"),
        ("Content-Type", "text/html; charset=utf-8", "Content-Type"),
    ),
    ids=("wrong-revision", "encoded", "html"),
)
def test_stdlib_metadata_download_rejects_direct_response_headers_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    invalid_header: str,
    invalid_value: str,
    message: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    base_url = f"https://huggingface.co/{repository}/resolve/{revision}"
    streams: list[FakeHttpStream] = []
    if filename == "model.safetensors.index.json":
        config_body = responses[(f"{base_url}/config.json", None)].body
        streams.append(
            FakeHttpStream(
                status=200,
                headers={
                    "Content-Length": str(len(config_body)),
                    "X-Repo-Commit": revision,
                },
                body=config_body,
            )
        )
    target_body = responses[(f"{base_url}/{filename}", None)].body
    target_headers = {
        "Content-Length": str(len(target_body)),
        "X-Repo-Commit": revision,
    }
    target_headers[invalid_header] = invalid_value
    target_stream = FakeHttpStream(
        status=200,
        headers=target_headers,
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    streams.append(target_stream)
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", SequenceOpener(tuple(streams)))

    with pytest.raises(MetadataStagingError, match=message):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=transport,
        )

    assert target_stream.read_sizes == []
    assert target_stream.closed


def test_stdlib_direct_range_rejects_wrong_revision_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    url = f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    stream = FakeHttpStream(
        status=206,
        headers={
            "Content-Length": "8",
            "Content-Range": "bytes 0-7/128",
            "X-Repo-Commit": "f" * 40,
        },
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    with pytest.raises(MetadataStagingError, match="X-Repo-Commit"):
        download_safetensors_header(
            url,
            repository="example/model",
            revision=revision,
            transport=transport,
        )

    assert stream.read_sizes == []
    assert stream.closed


def test_stdlib_second_range_rejects_changed_total_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    url = f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    length_stream = FakeHttpStream(
        status=206,
        headers={
            "Content-Length": "8",
            "Content-Range": "bytes 0-7/24",
            "X-Repo-Commit": revision,
        },
        body=(8).to_bytes(8, "little"),
    )
    header_stream = FakeHttpStream(
        status=206,
        headers={
            "Content-Length": "8",
            "Content-Range": "bytes 8-15/25",
            "X-Repo-Commit": revision,
        },
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(
        transport,
        "_opener",
        SequenceOpener((length_stream, header_stream)),
    )

    with pytest.raises(MetadataStagingError, match="total changed"):
        download_safetensors_header(
            url,
            repository="example/model",
            revision=revision,
            transport=transport,
        )

    assert length_stream.read_sizes == [8]
    assert header_stream.read_sizes == []
    assert header_stream.closed


@pytest.mark.parametrize(
    ("requested_range", "response_headers", "message", "opens_response"),
    (
        (
            "bytes=0-7",
            {"Content-Range": "bytes 0-7/128"},
            "Content-Length",
            True,
        ),
        (
            "bytes=0-7",
            {"Content-Length": "7", "Content-Range": "bytes 0-7/128"},
            "Content-Length",
            True,
        ),
        (
            "bytes=0-7",
            {"Content-Length": "8", "Content-Range": "bytes 0-6/128"},
            "Content-Range",
            True,
        ),
        (
            "bytes=0-7",
            {"Content-Length": "8", "Content-Range": "bytes 0-7/7"},
            "outside bounds",
            True,
        ),
        (
            "bytes=0-7",
            {
                "Content-Encoding": "gzip",
                "Content-Length": "8",
                "Content-Range": "bytes 0-7/128",
            },
            "identity",
            True,
        ),
        (
            "bytes=7-0",
            {"Content-Length": "8", "Content-Range": "bytes 0-7/128"},
            "Range",
            False,
        ),
        (
            "bytes=0-" + "9" * 5000,
            {"Content-Length": "8", "Content-Range": "bytes 0-7/128"},
            "Range",
            False,
        ),
        (
            "bytes=0-7",
            {"Content-Length": "8", "Content-Range": "bytes 0-7/" + "9" * 5000},
            "Content-Range",
            True,
        ),
    ),
    ids=(
        "missing-content-length",
        "wrong-content-length",
        "wrong-content-range",
        "total-not-larger-than-end",
        "encoded-body",
        "reversed-request-range",
        "oversized-request-decimal",
        "oversized-total-decimal",
    ),
)
def test_stdlib_transport_rejects_malformed_206_before_body_read(
    monkeypatch: pytest.MonkeyPatch,
    requested_range: str,
    response_headers: Mapping[str, str],
    message: str,
    opens_response: bool,
) -> None:
    stream = FakeHttpStream(
        status=206,
        headers=response_headers,
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    opener = FakeOpener(stream)
    monkeypatch.setattr(transport, "_opener", opener)

    with pytest.raises(MetadataStagingError, match=message):
        transport.request(
            "https://huggingface.co/example/model/resolve/"
            + "0" * 40
            + "/model.safetensors",
            headers={"Range": requested_range, "Accept-Encoding": "identity"},
            max_body_bytes=8,
        )

    assert stream.read_sizes == []
    assert stream.closed is opens_response
    assert len(opener.requests) == int(opens_response)


def test_stdlib_transport_accepts_last_byte_of_maximum_safetensors_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    last_byte = MAX_SAFETENSORS_FILE_BYTES - 1
    stream = FakeHttpStream(
        status=206,
        headers={
            "Content-Length": "1",
            "Content-Range": f"bytes {last_byte}-{last_byte}/"
            f"{MAX_SAFETENSORS_FILE_BYTES}",
        },
        body=b"xpayload-must-not-be-read",
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    response = transport.request(
        "https://huggingface.co/example/model/resolve/"
        + "0" * 40
        + "/model.safetensors",
        headers={"Range": f"bytes={last_byte}-{last_byte}"},
        max_body_bytes=1,
    )

    assert response.body == b"x"
    assert stream.read_sizes == [1]
    assert stream.closed


def test_stdlib_transport_rejects_range_at_file_bound_before_http_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_invalid_byte = MAX_SAFETENSORS_FILE_BYTES
    transport = UrllibHttpTransport()
    monkeypatch.setattr(
        transport,
        "_opener",
        RaisingOpener(AssertionError("invalid Range reached HTTP opener")),
    )

    with pytest.raises(MetadataStagingError, match="Range start.*outside"):
        transport.request(
            "https://huggingface.co/example/model/resolve/"
            + "0" * 40
            + "/model.safetensors",
            headers={"Range": f"bytes={first_invalid_byte}-{first_invalid_byte}"},
            max_body_bytes=1,
        )


@pytest.mark.parametrize(
    "header_name",
    (
        "Content-Length",
        "Content-Range",
        "Content-Encoding",
        "Location",
        "X-Repo-Commit",
        "X-Linked-Size",
    ),
)
def test_stdlib_transport_rejects_duplicate_security_header_before_dict_collapse(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
) -> None:
    base_headers = {
        "Content-Length": "8",
        "Content-Range": "bytes 0-7/128",
    }
    base_headers.pop(header_name, None)
    pairs = tuple(base_headers.items()) + (
        (header_name, "first"),
        (header_name.swapcase(), "second"),
    )
    stream = FakeHttpStream(
        status=206,
        headers=FakeHttpHeaders(pairs),
        body=b"payload-must-not-be-read",
        forbid_read=True,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))

    with pytest.raises(MetadataStagingError, match=f"repeats {header_name}"):
        transport.request(
            "https://huggingface.co/example/model/resolve/"
            + "0" * 40
            + "/model.safetensors",
            headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"},
            max_body_bytes=8,
        )

    assert stream.read_sizes == []
    assert stream.closed


@pytest.mark.parametrize(
    "network_error",
    (
        urllib.error.URLError("authorization-secret"),
        OSError("authorization-secret"),
        http.client.HTTPException("authorization-secret"),
    ),
)
def test_stdlib_transport_redacts_secrets_from_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    network_error: BaseException,
) -> None:
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", RaisingOpener(network_error))
    url = (
        "https://signed-user:signed-password@us.aws.cdn.hf.co/object"
        "?X-Amz-Signature=signed-query#signed-fragment"
    )
    request_headers = {"Authorization": "Bearer authorization-secret"}

    with pytest.raises(MetadataStagingError) as exc_info:
        transport.request(url, headers=request_headers, max_body_bytes=8)

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "https://us.aws.cdn.hf.co/object" in rendered
    assert all(
        secret not in rendered
        for secret in (
            "signed-user",
            "signed-password",
            "signed-query",
            "signed-fragment",
            "authorization-secret",
        )
    )


@pytest.mark.parametrize(
    "invalid_character",
    tuple(chr(codepoint) for codepoint in range(0x20))
    + (chr(0x7F),)
    + tuple(chr(codepoint) for codepoint in range(0x80, 0xA0))
    + ("\u2603",),
    ids=lambda value: f"U+{ord(value):04X}",
)
def test_stdlib_transport_rejects_unsafe_authorization_before_http_open(
    monkeypatch: pytest.MonkeyPatch,
    invalid_character: str,
) -> None:
    transport = UrllibHttpTransport()
    monkeypatch.setattr(
        transport,
        "_opener",
        RaisingOpener(AssertionError("unsafe Authorization reached HTTP opener")),
    )
    secret = f"Bearer prefix{invalid_character}authorization-secret"

    with pytest.raises(MetadataStagingError) as exc_info:
        transport.request(
            "https://huggingface.co/example/model/resolve/" + "0" * 40 + "/config.json",
            headers={"Authorization": secret},
            max_body_bytes=8,
        )

    _assert_redacted_exception(exc_info.value, secret, "authorization-secret")


def test_stdlib_transport_accepts_safe_latin1_authorization_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeHttpStream(
        status=200,
        headers={"Content-Length": "2"},
        body=b"{}",
    )
    transport = UrllibHttpTransport()
    opener = FakeOpener(stream)
    monkeypatch.setattr(transport, "_opener", opener)

    response = transport.request(
        "https://huggingface.co/example/model/resolve/" + "0" * 40 + "/config.json",
        headers={"Authorization": "Bearer opaque-\u00a0-\u00ff"},
        max_body_bytes=8,
    )

    assert response.body == b"{}"
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "read_error",
    (
        http.client.IncompleteRead(b"partial", 1),
        http.client.HTTPException("authorization-secret"),
        OSError("authorization-secret"),
    ),
)
def test_stdlib_transport_redacts_secrets_from_read_failure(
    monkeypatch: pytest.MonkeyPatch,
    read_error: BaseException,
) -> None:
    stream = FakeHttpStream(
        status=206,
        headers={"Content-Length": "8", "Content-Range": "bytes 0-7/128"},
        body=b"",
        read_error=read_error,
    )
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", FakeOpener(stream))
    url = (
        "https://signed-user:signed-password@us.aws.cdn.hf.co/object"
        "?X-Amz-Signature=signed-query#signed-fragment"
    )
    request_headers = {
        "Authorization": "Bearer authorization-secret",
        "Range": "bytes=0-7",
    }

    with pytest.raises(MetadataStagingError) as exc_info:
        transport.request(url, headers=request_headers, max_body_bytes=8)

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert "https://us.aws.cdn.hf.co/object" in rendered
    assert all(
        secret not in rendered
        for secret in (
            "signed-user",
            "signed-password",
            "signed-query",
            "signed-fragment",
            "authorization-secret",
        )
    )
    assert stream.closed


@pytest.mark.parametrize(
    ("changed_response", "message"),
    (
        (
            _response(
                200,
                b"12345678",
                revision="0123456789abcdef0123456789abcdef01234567",
            ),
            "must return HTTP 206",
        ),
        (
            _response(
                206,
                b"12345678",
                content_range="bytes 0-6/99",
                revision="0123456789abcdef0123456789abcdef01234567",
            ),
            "Content-Range",
        ),
        (
            _response(
                206,
                b"12345678",
                content_range="bytes 0-7/99",
                revision="0123456789abcdef0123456789abcdef01234567",
                extra_headers={"Content-Length": "7"},
            ),
            "Content-Length",
        ),
        (
            _response(
                206,
                b"1234567",
                content_range="bytes 0-7/99",
                revision="0123456789abcdef0123456789abcdef01234567",
                extra_headers={"Content-Length": "8"},
            ),
            "response body length",
        ),
    ),
)
def test_header_range_fetch_fails_closed_on_partial_or_malformed_response(
    changed_response: HttpResponse,
    message: str,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    url = f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    transport = ScriptedTransport({(url, "bytes=0-7"): changed_response})

    with pytest.raises(MetadataStagingError, match=message):
        download_safetensors_header(
            url,
            repository="example/model",
            revision=revision,
            transport=transport,
        )


def test_checkpoint_stager_rejects_index_header_disagreement(tmp_path: Path) -> None:
    spec, responses, _ = _miniature_checkpoint(
        raw_header_tensor_name="different.weight"
    )
    transport = ScriptedTransport(responses)

    with pytest.raises(MetadataStagingError, match="index/header"):
        stage_checkpoint_metadata(spec, tmp_path / "example", transport=transport)


def test_checkpoint_stager_rejects_duplicate_index_tensor_names(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    index_url = (
        f"https://huggingface.co/{repository}/resolve/{revision}/"
        "model.safetensors.index.json"
    )
    shard = "model-00001-of-00001.safetensors"
    duplicate_index = (
        b'{"weight_map":{"weight":"'
        + shard.encode()
        + b'","weight":"'
        + shard.encode()
        + b'"}}'
    )
    responses[(index_url, None)] = _response(
        200,
        duplicate_index,
        revision=revision,
    )
    changed_spec = replace(
        spec,
        artifact={
            **spec.artifact,
            "index_sha256": hashlib.sha256(duplicate_index).hexdigest(),
        },
    )

    with pytest.raises(MetadataStagingError, match="repeats key: weight"):
        stage_checkpoint_metadata(
            changed_spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )


@pytest.mark.parametrize(
    ("digest_field", "parsed_field"),
    (
        ("config_sha256", "config.json"),
        ("index_sha256", "model.safetensors.index.json"),
    ),
    ids=("config", "index"),
)
def test_checkpoint_stager_verifies_digest_before_parsing_pinned_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest_field: str,
    parsed_field: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    changed_spec = replace(
        spec,
        artifact={**spec.artifact, digest_field: "0" * 64},
    )
    parsed_fields: list[str] = []
    original_parse = metadata_stager._parse_json_object

    def recording_parse(raw: bytes, field: str) -> Mapping[str, object]:
        parsed_fields.append(field)
        return original_parse(raw, field)

    monkeypatch.setattr(metadata_stager, "_parse_json_object", recording_parse)

    with pytest.raises(MetadataStagingError, match="SHA256"):
        stage_checkpoint_metadata(
            changed_spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )

    assert parsed_field not in parsed_fields


def test_checkpoint_stager_parses_accepted_index_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    parsed_fields: list[str] = []
    original_parse = metadata_stager._parse_json_object

    def recording_parse(raw: bytes, field: str) -> Mapping[str, object]:
        parsed_fields.append(field)
        return original_parse(raw, field)

    monkeypatch.setattr(metadata_stager, "_parse_json_object", recording_parse)

    stage_checkpoint_metadata(
        spec,
        tmp_path / "example",
        transport=ScriptedTransport(responses),
    )

    assert parsed_fields.count("model.safetensors.index.json") == 1


def test_checkpoint_stager_normalizes_excessive_json_integer(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    raw_digits = "7" * 5000
    oversized_integer_config = f'{{"integer":{raw_digits}}}'.encode()
    responses[(config_url, None)] = _response(
        200,
        oversized_integer_config,
        revision=revision,
    )
    changed_spec = replace(
        spec,
        artifact={
            **spec.artifact,
            "config_sha256": hashlib.sha256(oversized_integer_config).hexdigest(),
        },
    )

    with pytest.raises(MetadataStagingError) as exc_info:
        stage_checkpoint_metadata(
            changed_spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert raw_digits[:128] not in rendered
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("raw_float", ("1e999999", "-1e999999"))
def test_checkpoint_stager_rejects_nonfinite_exponent_float(
    tmp_path: Path,
    raw_float: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    overflowing_config = f'{{"secret_float":{raw_float}}}'.encode()
    responses[(config_url, None)] = _response(
        200,
        overflowing_config,
        revision=revision,
    )
    changed_spec = replace(
        spec,
        artifact={
            **spec.artifact,
            "config_sha256": hashlib.sha256(overflowing_config).hexdigest(),
        },
    )

    with pytest.raises(MetadataStagingError) as exc_info:
        stage_checkpoint_metadata(
            changed_spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )

    rendered = "".join(traceback.format_exception(exc_info.value))
    assert raw_float not in rendered
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("raw_float", ("1.25", "-1.25", "6.022e23"))
def test_checkpoint_stager_accepts_finite_json_float(
    tmp_path: Path,
    raw_float: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    finite_config = f'{{"finite_float":{raw_float}}}'.encode()
    responses[(config_url, None)] = _response(
        200,
        finite_config,
        revision=revision,
    )
    changed_spec = replace(
        spec,
        artifact={
            **spec.artifact,
            "config_sha256": hashlib.sha256(finite_config).hexdigest(),
        },
    )

    staged = stage_checkpoint_metadata(
        changed_spec,
        tmp_path / "example",
        transport=ScriptedTransport(responses),
    )

    assert staged


def test_checkpoint_stager_rejects_auth_html_before_hash_check(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    responses[(config_url, None)] = _response(
        200,
        b"<!doctype html><html>login</html>",
        revision=revision,
        extra_headers={"Content-Type": "text/html"},
    )

    with pytest.raises(MetadataStagingError, match="HTML instead of JSON"):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )


def test_checkpoint_stager_accepts_one_revision_bound_hf_cache_redirect(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    redirected_path = (
        f"/api/resolve-cache/models/{repository}/{revision}/config.json?etag=pinned"
    )
    redirected_url = "https://huggingface.co" + redirected_path
    config_body = responses[(config_url, None)].body
    responses[(config_url, None)] = _response(
        307,
        b"",
        revision=revision,
        extra_headers={"Location": redirected_path},
    )
    responses[(redirected_url, None)] = _response(200, config_body)
    transport = ScriptedTransport(responses)

    stage_checkpoint_metadata(
        spec,
        tmp_path / "example",
        transport=transport,
        authorization="Bearer origin-only",
    )

    assert transport.requests[0][0] == config_url
    assert transport.requests[0][1]["Authorization"] == "Bearer origin-only"
    assert transport.requests[1][0] == redirected_url
    assert "Authorization" not in transport.requests[1][1]


@pytest.mark.parametrize(
    "location",
    (
        "https://evil.example/api/resolve-cache/models/example/model/"
        + "0123456789abcdef0123456789abcdef01234567/config.json",
        "/api/resolve-cache/models/example/other/"
        + "0123456789abcdef0123456789abcdef01234567/config.json",
        "/api/resolve-cache/models/example/model/" + "f" * 40 + "/config.json",
        "/api/resolve-cache/models/example/model/"
        + "0123456789abcdef0123456789abcdef01234567/other.json",
    ),
)
def test_checkpoint_stager_rejects_unbound_metadata_redirect(
    tmp_path: Path,
    location: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    config_url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    responses[(config_url, None)] = _response(
        307,
        b"",
        revision=revision,
        extra_headers={"Location": location},
    )

    with pytest.raises(MetadataStagingError, match="metadata redirect"):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )


def test_checkpoint_stager_records_and_checks_a95b_header_length(
    tmp_path: Path,
) -> None:
    spec, responses, raw_header = _miniature_checkpoint()
    shard = "model-00001-of-00001.safetensors"
    pinned_spec = replace(
        spec,
        artifact={
            **spec.artifact,
            "mtp_header_byte_lengths": {shard: len(raw_header)},
        },
    )

    stage_checkpoint_metadata(
        pinned_spec,
        tmp_path / "accepted",
        transport=ScriptedTransport(responses),
    )

    assert (
        tmp_path / "accepted" / "safetensors_header_byte_lengths.json"
    ).read_bytes() == _json_bytes({shard: len(raw_header)})

    wrong_spec = replace(
        pinned_spec,
        artifact={
            **pinned_spec.artifact,
            "mtp_header_byte_lengths": {shard: len(raw_header) + 8},
        },
    )
    with pytest.raises(MetadataStagingError, match="header byte lengths"):
        stage_checkpoint_metadata(
            wrong_spec,
            tmp_path / "rejected",
            transport=ScriptedTransport(responses),
        )


def test_checkpoint_stager_rejects_symlink_destination(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    real_destination = tmp_path / "real"
    real_destination.mkdir()
    linked_destination = tmp_path / "linked"
    linked_destination.symlink_to(real_destination, target_is_directory=True)

    with pytest.raises(MetadataStagingError, match="must not exist"):
        stage_checkpoint_metadata(
            spec,
            linked_destination,
            transport=ScriptedTransport(responses),
        )


def test_header_fetch_rejects_oversized_length_without_second_request() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    url = f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    claimed_length = MAX_SAFETENSORS_HEADER_BYTES + 8
    transport = ScriptedTransport(
        {
            (url, "bytes=0-7"): _response(
                206,
                claimed_length.to_bytes(8, "little"),
                content_range=f"bytes 0-7/{claimed_length + 16}",
                revision=revision,
            )
        }
    )

    with pytest.raises(MetadataStagingError, match="header length"):
        download_safetensors_header(
            url,
            repository="example/model",
            revision=revision,
            transport=transport,
        )

    assert len(transport.requests) == 1


def test_header_fetch_rejects_second_range_that_would_return_payload() -> None:
    spec, responses, raw_header = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    shard = "model-00001-of-00001.safetensors"
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{shard}"
    responses[(url, f"bytes=8-{7 + len(raw_header)}")] = _response(
        200,
        b"",
        revision=revision,
        extra_headers={"Content-Length": str(4 * 1024 * 1024 * 1024)},
    )
    transport = ScriptedTransport(responses)

    with pytest.raises(MetadataStagingError, match="must return HTTP 206"):
        download_safetensors_header(
            url,
            repository=repository,
            revision=revision,
            transport=transport,
        )

    assert transport.requests[-1][1]["Range"] == (f"bytes=8-{7 + len(raw_header)}")


def test_checkpoint_stager_rejects_duplicate_tensor_in_header_json(
    tmp_path: Path,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    shard = "model-00001-of-00001.safetensors"
    tensor = "model.layers.0.mlp.experts.0.gate_proj.weight"
    tensor_metadata = b'{"data_offsets":[0,8],"dtype":"BF16","shape":[2,2]}'
    duplicate_header = (
        b'{"'
        + tensor.encode()
        + b'":'
        + tensor_metadata
        + b',"'
        + tensor.encode()
        + b'":'
        + tensor_metadata
        + b"}"
    )
    duplicate_header += b" " * (-len(duplicate_header) % 8)
    total = 8 + len(duplicate_header) + 8
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{shard}"
    for key in tuple(responses):
        if key[0] == url:
            del responses[key]
    responses[(url, "bytes=0-7")] = _response(
        206,
        len(duplicate_header).to_bytes(8, "little"),
        content_range=f"bytes 0-7/{total}",
        revision=revision,
    )
    responses[(url, f"bytes=8-{7 + len(duplicate_header)}")] = _response(
        206,
        duplicate_header,
        content_range=f"bytes 8-{7 + len(duplicate_header)}/{total}",
        revision=revision,
    )

    with pytest.raises(MetadataStagingError, match="repeats key"):
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=ScriptedTransport(responses),
        )


class ExplodingTransport:
    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_body_bytes: int,
        response_header_policy: HttpResponseHeaderPolicy | None = None,
    ) -> HttpResponse:
        del url, headers, max_body_bytes, response_header_policy
        raise AssertionError("validated published metadata must not hit the network")


def _configure_miniature_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[tuple[str, str | None], HttpResponse],
    bytes,
    str,
]:
    spec, responses, _ = _miniature_checkpoint()
    prefix = "checkpoints/example"
    expected_manifest = "".join(
        sorted(
            (
                f"{spec.artifact['config_sha256']}  {prefix}/config.json\n",
                (
                    f"{spec.artifact['index_sha256']}  "
                    f"{prefix}/model.safetensors.index.json\n"
                ),
                (
                    f"{spec.artifact['header_manifest_sha256']}  "
                    f"{prefix}/safetensors_header_manifest.json\n"
                ),
            )
        )
    ).encode()
    expected_digest = hashlib.sha256(expected_manifest).hexdigest()
    monkeypatch.setattr(metadata_stager, "checkpoint_artifact_specs", lambda: (spec,))
    monkeypatch.setattr(
        metadata_stager,
        "expected_staged_metadata_manifest",
        lambda: expected_manifest,
    )
    monkeypatch.setattr(
        metadata_stager,
        "EXPECTED_STAGED_METADATA_MANIFEST_SHA256",
        expected_digest,
    )
    return responses, expected_manifest, expected_digest


def _write_single_file_verification_tree(
    root: Path,
    *,
    filename: str,
    body: bytes,
) -> tuple[Path, Path, bytes]:
    relative_path = f"checkpoints/example/{filename}"
    candidate = root / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(body)
    expected_manifest = (
        f"{hashlib.sha256(body).hexdigest()}  {relative_path}\n".encode()
    )
    manifest_path = root / metadata_stager.STAGED_METADATA_MANIFEST_FILENAME
    manifest_path.write_bytes(expected_manifest)
    return candidate, manifest_path, expected_manifest


def test_published_tree_rejects_wrong_manifest_size_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    _, manifest_path, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )
    manifest_path.write_bytes(expected_manifest + b"x")

    def forbidden_read(file_descriptor: int, size: int) -> bytes:
        del file_descriptor, size
        raise AssertionError("oversized manifest body was read")

    monkeypatch.setattr(os, "read", forbidden_read)

    with pytest.raises(MetadataStagingError, match="manifest differs"):
        metadata_stager._verify_published_tree(root, expected_manifest)


@pytest.mark.parametrize(
    ("filename", "maximum_size"),
    (
        ("config.json", MAX_CONFIG_BYTES),
        ("model.safetensors.index.json", MAX_INDEX_BYTES),
        ("safetensors_header_byte_lengths.json", MAX_CONFIG_BYTES),
        (
            "safetensors_header_manifest.json",
            metadata_stager.MAX_COMBINED_HEADER_MANIFEST_BYTES,
        ),
    ),
    ids=("config", "index", "header-lengths", "header-manifest"),
)
def test_published_tree_rejects_oversized_artifact_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    maximum_size: int,
) -> None:
    root = tmp_path / "published"
    candidate, _, expected_manifest = _write_single_file_verification_tree(
        root,
        filename=filename,
        body=b"",
    )
    os.truncate(candidate, maximum_size + 1)
    original_open = os.open
    original_read = os.read
    candidate_descriptor: int | None = None

    def recording_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal candidate_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == candidate:
            candidate_descriptor = descriptor
        return descriptor

    def guarded_read(file_descriptor: int, size: int) -> bytes:
        if file_descriptor == candidate_descriptor:
            raise AssertionError("oversized artifact body was read")
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "read", guarded_read)

    with pytest.raises(MetadataStagingError, match="size.*bound"):
        metadata_stager._verify_published_tree(root, expected_manifest)


def test_published_tree_rejects_unknown_artifact_kind_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    candidate, _, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="unknown.json",
        body=b"{}",
    )
    original_open = os.open

    def guarded_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == candidate:
            raise AssertionError("unknown artifact body was opened")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", guarded_open)

    with pytest.raises(MetadataStagingError, match="unsupported metadata path"):
        metadata_stager._verify_published_tree(root, expected_manifest)


def test_published_tree_hashes_artifacts_with_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_size = metadata_stager.VERIFICATION_HASH_CHUNK_BYTES
    body = b"x" * (2 * chunk_size + 17)
    root = tmp_path / "published"
    candidate, _, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=body,
    )
    original_read = os.read
    read_sizes: list[int] = []

    def recording_read(file_descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "read", recording_read)

    metadata_stager._verify_published_tree(root, expected_manifest)

    assert read_sizes
    assert all(0 < size <= chunk_size for size in read_sizes)


@pytest.mark.parametrize("target_kind", ("manifest", "artifact"))
def test_published_tree_rejects_fifo_swapped_before_secure_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    root = tmp_path / "published"
    candidate, manifest_path, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )
    target = manifest_path if target_kind == "manifest" else candidate
    target_body = expected_manifest if target_kind == "manifest" else b"{}"
    original_path_stat = Path.stat
    original_path_open = Path.open
    original_os_open = os.open
    original_os_close = os.close
    swapped = False
    target_stat_count = [0]
    target_descriptors: list[int] = []
    closed_descriptors: list[int] = []

    def swap_to_fifo() -> None:
        nonlocal swapped
        if swapped:
            return
        swapped = True
        target.unlink()
        os.mkfifo(target)

    def swapping_path_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        result = original_path_stat(path, *args, **kwargs)
        if path == target and kwargs.get("follow_symlinks", True):
            target_stat_count[0] += 1
            if target_stat_count[0] == 2:
                swap_to_fifo()
        return result

    def swapping_os_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == target:
            swap_to_fifo()
        descriptor = original_os_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == target:
            target_descriptors.append(descriptor)
        return descriptor

    def recording_os_close(file_descriptor: int) -> None:
        closed_descriptors.append(file_descriptor)
        original_os_close(file_descriptor)

    def feed_exact_bytes(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        if path == target:
            return io.BytesIO(target_body)
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", swapping_path_stat)
    monkeypatch.setattr(Path, "open", feed_exact_bytes)
    monkeypatch.setattr(os, "open", swapping_os_open)
    monkeypatch.setattr(os, "close", recording_os_close)

    with pytest.raises(MetadataStagingError, match="regular|changed|cannot be read"):
        metadata_stager._verify_published_tree(root, expected_manifest)

    assert swapped
    assert target_descriptors
    assert set(target_descriptors) <= set(closed_descriptors)


def test_published_tree_secure_open_uses_fail_closed_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    candidate, manifest_path, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )
    original_open = os.open
    opened_flags: dict[Path, int] = {}

    def recording_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        opened_flags[Path(path)] = flags
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", recording_open)

    metadata_stager._verify_published_tree(root, expected_manifest)

    required_flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    assert set(opened_flags) == {candidate, manifest_path}
    assert all(
        flags & required_flags == required_flags for flags in opened_flags.values()
    )


def test_published_tree_redacts_read_error_and_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    candidate, _, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )
    original_open = os.open
    original_read = os.read
    original_close = os.close
    candidate_descriptor: int | None = None
    closed_descriptors: list[int] = []

    def recording_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal candidate_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == candidate:
            candidate_descriptor = descriptor
        return descriptor

    def failing_read(file_descriptor: int, size: int) -> bytes:
        if file_descriptor == candidate_descriptor:
            raise OSError("published-read-secret")
        return original_read(file_descriptor, size)

    def recording_close(file_descriptor: int) -> None:
        closed_descriptors.append(file_descriptor)
        original_close(file_descriptor)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "read", failing_read)
    monkeypatch.setattr(os, "close", recording_close)

    with pytest.raises(MetadataStagingError) as exc_info:
        metadata_stager._verify_published_tree(root, expected_manifest)

    _assert_redacted_exception(exc_info.value, "published-read-secret")
    assert candidate_descriptor is not None
    assert candidate_descriptor in closed_descriptors


@pytest.mark.parametrize("target_kind", ("manifest", "artifact"))
def test_published_tree_rejects_path_replaced_after_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    root = tmp_path / "published"
    candidate, manifest_path, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )
    target = manifest_path if target_kind == "manifest" else candidate
    displaced = tmp_path / f"displaced-{target_kind}"
    original_open = os.open
    original_read = os.read
    target_descriptor: int | None = None
    swapped = False

    def recording_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal target_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == target:
            target_descriptor = descriptor
        return descriptor

    def swapping_read(file_descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if file_descriptor == target_descriptor and not swapped:
            swapped = True
            target.rename(displaced)
            os.mkfifo(target)
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "read", swapping_read)

    with pytest.raises(MetadataStagingError, match="changed"):
        metadata_stager._verify_published_tree(root, expected_manifest)

    assert swapped


def test_published_tree_verification_does_not_use_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "published"
    _, _, expected_manifest = _write_single_file_verification_tree(
        root,
        filename="config.json",
        body=b"{}",
    )

    def forbidden_read_bytes(path: Path) -> bytes:
        raise AssertionError(f"verification called Path.read_bytes: {path.name}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    metadata_stager._verify_published_tree(root, expected_manifest)


def test_atomic_content_addressed_publish_reuses_only_exact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, responses, _ = _miniature_checkpoint()
    prefix = "checkpoints/example"
    expected_manifest = "".join(
        sorted(
            (
                f"{spec.artifact['config_sha256']}  {prefix}/config.json\n",
                (
                    f"{spec.artifact['index_sha256']}  "
                    f"{prefix}/model.safetensors.index.json\n"
                ),
                (
                    f"{spec.artifact['header_manifest_sha256']}  "
                    f"{prefix}/safetensors_header_manifest.json\n"
                ),
            )
        )
    ).encode()
    expected_digest = hashlib.sha256(expected_manifest).hexdigest()
    monkeypatch.setattr(metadata_stager, "checkpoint_artifact_specs", lambda: (spec,))
    monkeypatch.setattr(
        metadata_stager,
        "expected_staged_metadata_manifest",
        lambda: expected_manifest,
    )
    monkeypatch.setattr(
        metadata_stager,
        "EXPECTED_STAGED_METADATA_MANIFEST_SHA256",
        expected_digest,
    )

    published = stage_source_metadata(
        tmp_path / "metadata",
        transport=ScriptedTransport(responses),
    )

    assert published.name == f"sha256-{expected_digest}"
    assert (published / "SHA256SUMS").read_bytes() == expected_manifest
    assert not tuple(published.parent.glob(".precision-policy-source-metadata-*"))
    assert (
        stage_source_metadata(
            tmp_path / "metadata",
            transport=ExplodingTransport(),
        )
        == published
    )

    (published / "unexpected-empty-directory").mkdir()
    with pytest.raises(MetadataStagingError, match="unexpected"):
        stage_source_metadata(
            tmp_path / "metadata",
            transport=ExplodingTransport(),
        )


@pytest.mark.parametrize("collision_errno", (errno.ENOTEMPTY, errno.EEXIST))
def test_concurrent_identical_publication_verifies_winner_and_cleans_loser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_errno: int,
) -> None:
    responses, expected_manifest, expected_digest = _configure_miniature_publication(
        monkeypatch
    )

    def competing_rename(source: Path, target: str | os.PathLike[str]) -> Path:
        target_path = Path(target)
        shutil.copytree(source, target_path)
        raise OSError(collision_errno, "simulated publication collision")

    monkeypatch.setattr(Path, "rename", competing_rename)
    output_root = tmp_path / "metadata"

    published = stage_source_metadata(
        output_root,
        transport=ScriptedTransport(responses),
    )

    assert published == output_root / f"sha256-{expected_digest}"
    assert (published / "SHA256SUMS").read_bytes() == expected_manifest
    assert not tuple(output_root.glob(".precision-policy-source-metadata-*"))


def test_concurrent_publication_rejects_bad_winner_and_cleans_loser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, _, _ = _configure_miniature_publication(monkeypatch)

    def competing_rename(source: Path, target: str | os.PathLike[str]) -> Path:
        target_path = Path(target)
        shutil.copytree(source, target_path)
        (target_path / "SHA256SUMS").write_bytes(b"corrupt\n")
        raise OSError(errno.ENOTEMPTY, "simulated publication collision")

    monkeypatch.setattr(Path, "rename", competing_rename)
    output_root = tmp_path / "metadata"

    with pytest.raises(MetadataStagingError, match="manifest differs"):
        stage_source_metadata(
            output_root,
            transport=ScriptedTransport(responses),
        )

    assert not tuple(output_root.glob(".precision-policy-source-metadata-*"))


def test_publication_does_not_swallow_unrelated_rename_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, _, _ = _configure_miniature_publication(monkeypatch)

    def failed_rename(source: Path, target: str | os.PathLike[str]) -> Path:
        del source, target
        raise OSError(errno.EACCES, "simulated permission failure")

    monkeypatch.setattr(Path, "rename", failed_rename)
    output_root = tmp_path / "metadata"

    with pytest.raises(OSError) as exc_info:
        stage_source_metadata(
            output_root,
            transport=ScriptedTransport(responses),
        )

    assert exc_info.value.errno == errno.EACCES
    assert not tuple(output_root.glob(".precision-policy-source-metadata-*"))


@pytest.mark.parametrize(
    "shard_name",
    (
        "../escape.safetensors",
        "/absolute.safetensors",
        "nested/model.safetensors",
        "model.bin",
        "..safetensors",
        "model%2Fescape.safetensors",
        "model\n.safetensors",
    ),
)
def test_checkpoint_stager_rejects_unsafe_shard_names(
    tmp_path: Path,
    shard_name: str,
) -> None:
    spec, responses, _ = _miniature_checkpoint(indexed_shard=shard_name)
    transport = ScriptedTransport(responses)

    with pytest.raises(MetadataStagingError, match="safe safetensors basename"):
        stage_checkpoint_metadata(spec, tmp_path / "example", transport=transport)


def test_header_fetch_accepts_one_reviewed_redirect_without_auth_leak() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    origin = (
        f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    )
    redirected = "https://us.aws.cdn.hf.co/object?Signature=example"
    header = _json_bytes(
        {"weight": {"data_offsets": [0, 2], "dtype": "U8", "shape": [2]}}
    )
    header += b" " * (-len(header) % 8)
    total = 8 + len(header) + 2
    transport = ScriptedTransport(
        {
            (origin, "bytes=0-7"): _response(
                302,
                b"redirect",
                revision=revision,
                extra_headers={
                    "Location": redirected,
                    "X-Linked-Size": str(total),
                },
            ),
            (redirected, "bytes=0-7"): _response(
                206,
                len(header).to_bytes(8, "little"),
                content_range=f"bytes 0-7/{total}",
            ),
            (redirected, f"bytes=8-{7 + len(header)}"): _response(
                206,
                header,
                content_range=f"bytes 8-{7 + len(header)}/{total}",
            ),
        }
    )

    observed = download_safetensors_header(
        origin,
        repository="example/model",
        revision=revision,
        transport=transport,
        authorization="Bearer do-not-forward",
    )

    assert observed.header_length == len(header)
    assert transport.requests[0][1]["Authorization"] == "Bearer do-not-forward"
    assert all("Authorization" not in headers for _, headers in transport.requests[1:])


def _assert_redacted_exception(
    error: MetadataStagingError,
    *secrets: str,
) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert all(secret not in str(error) for secret in secrets)
    assert all(secret not in rendered for secret in secrets)


@pytest.mark.parametrize(
    "invalid_port",
    ("lfs-invalid-port-marker", "65536"),
    ids=("nonnumeric", "overflow"),
)
def test_stdlib_header_fetch_rejects_invalid_redirect_port_without_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
    invalid_port: str,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    origin = (
        f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    )
    query_secret = "lfs-query-secret"
    authorization_secret = "lfs-authorization-secret"
    redirected = (
        f"https://us.aws.cdn.hf.co:{invalid_port}/object?Signature={query_secret}"
    )
    stream = FakeHttpStream(
        status=302,
        headers={
            "Content-Length": "8",
            "Location": redirected,
            "X-Linked-Size": "128",
            "X-Repo-Commit": revision,
        },
        body=b"redirect",
        forbid_read=True,
    )
    opener = SequenceOpener((stream,))
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", opener)
    authorization = f"Bearer {authorization_secret}"

    with pytest.raises(MetadataStagingError) as exc_info:
        download_safetensors_header(
            origin,
            repository="example/model",
            revision=revision,
            transport=transport,
            authorization=authorization,
        )

    _assert_redacted_exception(
        exc_info.value,
        redirected,
        invalid_port,
        query_secret,
        authorization_secret,
    )
    assert stream.read_sizes == []
    assert stream.closed
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "invalid_port",
    ("metadata-invalid-port-marker", "65536"),
    ids=("nonnumeric", "overflow"),
)
def test_stdlib_metadata_download_rejects_invalid_redirect_port_without_secret_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_port: str,
) -> None:
    spec, _, _ = _miniature_checkpoint()
    repository = cast(str, spec.artifact["repository"])
    revision = cast(str, spec.artifact["revision"])
    query_secret = "metadata-query-secret"
    authorization_secret = "metadata-authorization-secret"
    location = (
        f"https://huggingface.co:{invalid_port}/api/resolve-cache/models/"
        f"{repository}/{revision}/config.json?etag={query_secret}"
    )
    stream = FakeHttpStream(
        status=307,
        headers={
            "Content-Length": "8",
            "Location": location,
            "X-Repo-Commit": revision,
        },
        body=b"redirect",
        forbid_read=True,
    )
    opener = SequenceOpener((stream,))
    transport = UrllibHttpTransport()
    monkeypatch.setattr(transport, "_opener", opener)
    authorization = f"Bearer {authorization_secret}"

    with pytest.raises(MetadataStagingError) as exc_info:
        stage_checkpoint_metadata(
            spec,
            tmp_path / "example",
            transport=transport,
            authorization=authorization,
        )

    _assert_redacted_exception(
        exc_info.value,
        location,
        invalid_port,
        query_secret,
        authorization_secret,
    )
    assert stream.read_sizes == []
    assert stream.closed
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    ("response_changes", "message"),
    (
        ({"status_code": 307}, "HTTP 302"),
        ({"x_repo_commit": "f" * 40}, "X-Repo-Commit"),
        ({"x_linked_size": "0"}, "X-Linked-Size"),
        ({"location": "http://us.aws.cdn.hf.co/object"}, "HTTPS"),
        ({"location": "https://evil.example/object"}, "redirect host"),
    ),
)
def test_header_fetch_rejects_untrusted_redirect(
    response_changes: Mapping[str, object],
    message: str,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    origin = (
        f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    )
    status_code = cast(int, response_changes.get("status_code", 302))
    response_revision = cast(str, response_changes.get("x_repo_commit", revision))
    location = cast(
        str,
        response_changes.get(
            "location", "https://us.aws.cdn.hf.co/object?Signature=example"
        ),
    )
    linked_size = cast(str, response_changes.get("x_linked_size", "128"))
    transport = ScriptedTransport(
        {
            (origin, "bytes=0-7"): _response(
                status_code,
                b"redirect",
                revision=response_revision,
                extra_headers={
                    "Location": location,
                    "X-Linked-Size": linked_size,
                },
            )
        }
    )

    with pytest.raises(MetadataStagingError, match=message):
        download_safetensors_header(
            origin,
            repository="example/model",
            revision=revision,
            transport=transport,
        )


def test_header_fetch_rejects_redirect_chain() -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    origin = (
        f"https://huggingface.co/example/model/resolve/{revision}/model.safetensors"
    )
    redirected = "https://us.aws.cdn.hf.co/object?Signature=example"
    transport = ScriptedTransport(
        {
            (origin, "bytes=0-7"): _response(
                302,
                b"redirect",
                revision=revision,
                extra_headers={"Location": redirected, "X-Linked-Size": "128"},
            ),
            (redirected, "bytes=0-7"): _response(
                302,
                b"redirect again",
                extra_headers={"Location": "https://cas-bridge.xethub.hf.co/object"},
            ),
        }
    )

    with pytest.raises(MetadataStagingError, match="redirect chain"):
        download_safetensors_header(
            origin,
            repository="example/model",
            revision=revision,
            transport=transport,
        )
