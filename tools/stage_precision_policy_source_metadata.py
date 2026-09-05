# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage pinned checkpoint metadata without downloading tensor payloads."""

from __future__ import annotations

import argparse
import errno
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Protocol, cast

from tools.precision_policy_source_artifacts import (
    STAGED_CHECKPOINT_DIRECTORY,
    STAGED_CONFIG_FILENAME,
    STAGED_HEADER_LENGTHS_FILENAME,
    STAGED_HEADER_MANIFEST_FILENAME,
    STAGED_INDEX_FILENAME,
    CheckpointMetadataArtifactSpec,
    checkpoint_metadata_artifact_identities,
)


EXPECTED_STAGED_METADATA_MANIFEST_SHA256 = (
    "d766a56f8fed37c085ac490db26dc088d3bfdadd09ea84e325b05c5e8c715c4b"
)
STAGED_METADATA_MANIFEST_FILENAME = "SHA256SUMS"
MAX_CONFIG_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 512 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_FILE_BYTES = 64 * 1024 * 1024 * 1024
MAX_JSON_INTEGER_DECIMAL_DIGITS = 128
MAX_JSON_FLOAT_CHARACTERS = 128
# The combined tensor projection can exceed source configs but remains metadata-only.
MAX_COMBINED_HEADER_MANIFEST_BYTES = 512 * 1024 * 1024
VERIFICATION_HASH_CHUNK_BYTES = 1024 * 1024
_ARTIFACT_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_REPOSITORY_PART_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHARD_BASENAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_CONTENT_RANGE_PATTERN = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")
_REQUEST_RANGE_PATTERN = re.compile(r"bytes=([0-9]+)-([0-9]+)\Z")
_SECURITY_CRITICAL_RESPONSE_HEADERS = {
    "content-encoding": "Content-Encoding",
    "content-length": "Content-Length",
    "content-range": "Content-Range",
    "location": "Location",
    "x-linked-size": "X-Linked-Size",
    "x-repo-commit": "X-Repo-Commit",
}
_REVIEWED_REDIRECT_HOSTS = frozenset(
    {
        "cas-bridge.xethub.hf.co",
        "us.aws.cdn.hf.co",
    }
)


class MetadataStagingError(ValueError):
    """Remote metadata is incomplete or differs from its immutable pin."""


@dataclass(frozen=True)
class HttpResponse:
    """One non-automatically-redirected HTTP response."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class HttpResponseHeaderPolicy:
    """Transport-independent response checks that must precede body reads."""

    required_values: tuple[tuple[str, str], ...] = ()
    require_identity_encoding: bool = False
    rejected_media_types: frozenset[str] = frozenset()
    expected_range_total: int | None = None
    error_context: str | None = None


class HttpTransport(Protocol):
    """Minimal injectable HTTP boundary used by the metadata stager."""

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_body_bytes: int,
        response_header_policy: HttpResponseHeaderPolicy | None = None,
    ) -> HttpResponse: ...


class _ResponseHeaders(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


@dataclass(frozen=True)
class SafetensorsHeader:
    """Validated safetensors header bytes and immutable file bounds."""

    body: bytes
    header_length: int
    total_size: int


@dataclass(frozen=True)
class StagedFile:
    """One validated metadata file staged on disk."""

    path: Path
    sha256: str


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        return None


class UrllibHttpTransport:
    """Stdlib transport that exposes redirects to fail-closed policy code."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_body_bytes: int,
        response_header_policy: HttpResponseHeaderPolicy | None = None,
    ) -> HttpResponse:
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes <= 0
        ):
            raise ValueError("max_body_bytes must be a positive integer")
        _validate_request_authorization(headers)
        requested_range = _validated_requested_range(headers)
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        safe_url = _sanitized_url(url)
        try:
            response = self._opener.open(request, timeout=self._timeout_seconds)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, http.client.HTTPException):
            raise MetadataStagingError(f"HTTP request failed for {safe_url}") from None
        processing_failed = False
        try:
            status_code = cast(int, response.status)
            response_headers = _collect_response_headers(
                response.headers,
                response_header_policy,
            )
            body = b""
            if requested_range is not None:
                start, end = requested_range
                if status_code == 206:
                    read_length = _validate_range_headers_before_read(
                        status_code=status_code,
                        headers=response_headers,
                        start=start,
                        end=end,
                        expected_total=(
                            response_header_policy.expected_range_total
                            if response_header_policy is not None
                            else None
                        ),
                    )
                    _validate_response_header_policy(
                        response_headers,
                        response_header_policy,
                    )
                    if read_length > max_body_bytes:
                        raise MetadataStagingError(
                            "range Content-Length exceeds request body bound"
                        )
                    body = response.read(read_length)
            elif requested_range is None and status_code == 200:
                content_length = _positive_bounded_int(
                    _header_value(response_headers, "Content-Length"),
                    "HTTP response Content-Length",
                    max_body_bytes,
                )
                _validate_response_header_policy(
                    response_headers,
                    response_header_policy,
                )
                body = response.read(content_length)
            result = HttpResponse(
                status_code=status_code,
                headers=response_headers,
                body=body,
            )
        except MetadataStagingError:
            processing_failed = True
            raise
        except (OSError, http.client.HTTPException):
            processing_failed = True
            raise MetadataStagingError(f"HTTP read failed for {safe_url}") from None
        finally:
            try:
                response.close()
            except (OSError, http.client.HTTPException):
                if not processing_failed:
                    raise MetadataStagingError(
                        f"HTTP close failed for {safe_url}"
                    ) from None
        return result


def _sanitized_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<redacted-url>"
    if not parsed.scheme or hostname is None:
        return "<redacted-url>"
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        rendered_host = f"{rendered_host}:{port}"
    return urllib.parse.urlunsplit((parsed.scheme, rendered_host, parsed.path, "", ""))


def _split_untrusted_url(
    url: str,
    field: str,
) -> tuple[urllib.parse.SplitResult, int | None]:
    _require_visible_ascii_url(url, field)
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        raise MetadataStagingError(f"{field} contains an invalid port") from None
    return parsed, port


def _require_visible_ascii_url(url: str, field: str) -> None:
    if not url or any(not 0x21 <= ord(character) <= 0x7E for character in url):
        raise MetadataStagingError(f"{field} must contain only visible ASCII")


def _collect_response_headers(
    headers: _ResponseHeaders,
    policy: HttpResponseHeaderPolicy | None,
) -> dict[str, str]:
    collected: dict[str, str] = {}
    critical_headers = dict(_SECURITY_CRITICAL_RESPONSE_HEADERS)
    if policy is not None:
        critical_headers.update(
            (name.lower(), name) for name, _ in policy.required_values
        )
        if policy.require_identity_encoding:
            critical_headers["content-encoding"] = "Content-Encoding"
        if policy.rejected_media_types:
            critical_headers["content-type"] = "Content-Type"
        if policy.expected_range_total is not None:
            critical_headers["content-range"] = "Content-Range"
    security_headers_seen: set[str] = set()
    for name, value in headers.items():
        normalized_name = name.lower()
        canonical_name = critical_headers.get(normalized_name)
        if canonical_name is not None:
            if normalized_name in security_headers_seen:
                context = (
                    f"{policy.error_context}: "
                    if policy is not None and policy.error_context is not None
                    else ""
                )
                raise MetadataStagingError(
                    f"{context}HTTP response repeats {canonical_name}"
                )
            security_headers_seen.add(normalized_name)
        collected[name] = value
    return collected


def _validate_authorization_value(value: str) -> None:
    if not value:
        raise MetadataStagingError("authorization header is malformed")
    if any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F or ord(character) > 0xFF
        for character in value
    ):
        raise MetadataStagingError("authorization header is malformed")


def _validate_request_authorization(headers: Mapping[str, str]) -> None:
    values = [
        value for name, value in headers.items() if name.lower() == "authorization"
    ]
    if len(values) > 1:
        raise MetadataStagingError("authorization header is malformed")
    if values:
        _validate_authorization_value(values[0])


def _validate_response_header_policy(
    headers: Mapping[str, str],
    policy: HttpResponseHeaderPolicy | None,
) -> None:
    if policy is None:
        return
    context = f"{policy.error_context}: " if policy.error_context is not None else ""
    for name, expected_value in policy.required_values:
        if _header_value(headers, name) != expected_value:
            raise MetadataStagingError(
                f"{context}HTTP response {name} differs from policy"
            )
    if policy.require_identity_encoding:
        encoding = _header_value(headers, "Content-Encoding")
        if encoding not in (None, "", "identity"):
            raise MetadataStagingError(
                f"{context}HTTP response must use identity content encoding"
            )
    raw_content_type = _header_value(headers, "Content-Type")
    if raw_content_type is not None:
        media_type = raw_content_type.partition(";")[0].strip().lower()
        if media_type in policy.rejected_media_types:
            raise MetadataStagingError(
                f"{context}HTTP response Content-Type is forbidden by policy"
            )


def _validated_requested_range(
    headers: Mapping[str, str],
) -> tuple[int, int] | None:
    requested_range = _header_value(headers, "Range")
    if requested_range is None:
        return None
    match = _REQUEST_RANGE_PATTERN.fullmatch(requested_range)
    if match is None:
        raise MetadataStagingError("HTTP Range request is malformed")
    raw_start, raw_end = match.groups()
    start = _bounded_nonnegative_int(
        raw_start,
        "HTTP Range start",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    end = _bounded_nonnegative_int(
        raw_end,
        "HTTP Range end",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    requested_length = end - start + 1
    if (
        end < start
        or end >= MAX_SAFETENSORS_FILE_BYTES
        or requested_length > MAX_SAFETENSORS_HEADER_BYTES
    ):
        raise MetadataStagingError("HTTP Range request is outside bounds")
    return start, end


def _validate_range_headers_before_read(
    *,
    status_code: int,
    headers: Mapping[str, str],
    start: int,
    end: int,
    expected_total: int | None,
) -> int:
    if status_code != 206:
        raise MetadataStagingError("safetensors range must return HTTP 206")
    expected_length = end - start + 1
    content_length = _positive_bounded_int(
        _header_value(headers, "Content-Length"),
        "range Content-Length",
        expected_length,
    )
    if content_length != expected_length:
        raise MetadataStagingError("range Content-Length differs from request")
    content_range = _header_value(headers, "Content-Range")
    match = _CONTENT_RANGE_PATTERN.fullmatch(content_range or "")
    if match is None:
        raise MetadataStagingError("range Content-Range is malformed")
    raw_start, raw_end, raw_total = match.groups()
    observed_start = _bounded_nonnegative_int(
        raw_start,
        "range Content-Range start",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    observed_end = _bounded_nonnegative_int(
        raw_end,
        "range Content-Range end",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    total = _bounded_nonnegative_int(
        raw_total,
        "range Content-Range total",
        MAX_SAFETENSORS_FILE_BYTES,
    )
    if (observed_start, observed_end) != (start, end):
        raise MetadataStagingError("range Content-Range differs from request")
    if total <= end or total > MAX_SAFETENSORS_FILE_BYTES:
        raise MetadataStagingError("range Content-Range total is outside bounds")
    if expected_total is not None and total != expected_total:
        raise MetadataStagingError("range Content-Range total changed between requests")
    encoding = _header_value(headers, "Content-Encoding")
    if encoding not in (None, "", "identity"):
        raise MetadataStagingError("HTTP response must use identity content encoding")
    return expected_length


def _canonical_compact_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def expected_staged_metadata_manifest() -> bytes:
    """Return the reviewed sorted SHA256 manifest for all staged metadata."""
    entries: list[tuple[str, str]] = []
    for spec in checkpoint_metadata_artifact_identities():
        prefix = f"{STAGED_CHECKPOINT_DIRECTORY}/{spec.artifact_id}"
        for filename, digest_field in (
            (STAGED_CONFIG_FILENAME, "config_sha256"),
            (STAGED_INDEX_FILENAME, "index_sha256"),
            (STAGED_HEADER_MANIFEST_FILENAME, "header_manifest_sha256"),
        ):
            entries.append(
                (cast(str, spec.artifact[digest_field]), f"{prefix}/{filename}")
            )
        header_lengths = spec.artifact.get("mtp_header_byte_lengths")
        if header_lengths is not None:
            entries.append(
                (
                    hashlib.sha256(_canonical_compact_json(header_lengths)).hexdigest(),
                    f"{prefix}/{STAGED_HEADER_LENGTHS_FILENAME}",
                )
            )
    manifest = "".join(
        sorted(f"{digest}  {path}\n" for digest, path in entries)
    ).encode()
    if len(manifest.splitlines()) != 19:
        raise MetadataStagingError("pinned metadata manifest must contain 19 files")
    if hashlib.sha256(manifest).hexdigest() != (
        EXPECTED_STAGED_METADATA_MANIFEST_SHA256
    ):
        raise MetadataStagingError("pinned metadata manifest digest differs")
    return manifest


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    values = [value for key, value in headers.items() if key.lower() == name.lower()]
    if len(values) > 1:
        raise MetadataStagingError(f"HTTP response repeats {name}")
    return values[0] if values else None


def _positive_bounded_int(value: str | None, field: str, maximum: int) -> int:
    parsed = _bounded_nonnegative_int(value, field, maximum)
    if parsed <= 0:
        raise MetadataStagingError(f"{field} is outside the reviewed byte bound")
    return parsed


def _bounded_nonnegative_int(
    value: str | None,
    field: str,
    maximum: int,
) -> int:
    if value is None or not value.isascii() or not value.isdecimal():
        raise MetadataStagingError(f"{field} must be a decimal integer")
    if len(value) > len(str(maximum)):
        raise MetadataStagingError(f"{field} is outside the reviewed byte bound")
    parsed = int(value)
    if parsed > maximum:
        raise MetadataStagingError(f"{field} is outside the reviewed byte bound")
    return parsed


def _validate_repository(repository: str) -> None:
    parts = repository.split("/")
    if len(parts) != 2 or any(
        _REPOSITORY_PART_PATTERN.fullmatch(part) is None for part in parts
    ):
        raise MetadataStagingError("repository must be an exact owner/name pair")


def _validate_revision(revision: str) -> None:
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise MetadataStagingError("revision must be an exact lowercase 40-hex commit")


def _validated_shard_basename(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise MetadataStagingError(
            "shard must be a traversal-safe safetensors basename"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name != value
        or path.suffix != ".safetensors"
        or _SHARD_BASENAME_PATTERN.fullmatch(value) is None
    ):
        raise MetadataStagingError(
            "shard must be a traversal-safe safetensors basename"
        )
    return value


def _resource_url(repository: str, revision: str, filename: str) -> str:
    _validate_repository(repository)
    _validate_revision(revision)
    if filename in {STAGED_CONFIG_FILENAME, STAGED_INDEX_FILENAME}:
        safe_filename = filename
    else:
        safe_filename = _validated_shard_basename(filename)
    quoted_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    return (
        f"https://huggingface.co/{quoted_repository}/resolve/{revision}/"
        f"{urllib.parse.quote(safe_filename, safe='')}"
    )


def _validate_original_url(
    url: str,
    *,
    repository: str,
    revision: str,
    filename: str,
) -> None:
    if url != _resource_url(repository, revision, filename):
        raise MetadataStagingError("original URL differs from immutable HF resource")


def _base_headers(
    *,
    authorization: str | None,
    byte_range: tuple[int, int] | None,
) -> dict[str, str]:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "nemo-rl-precision-policy-metadata-stager/1",
    }
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    if authorization is not None:
        _validate_authorization_value(authorization)
        headers["Authorization"] = authorization
    return headers


def _validated_redirect(
    response: HttpResponse,
    *,
    revision: str,
    maximum_size: int,
) -> tuple[str, int]:
    if response.status_code != 302:
        raise MetadataStagingError("immutable HF redirect must use HTTP 302")
    if _header_value(response.headers, "X-Repo-Commit") != revision:
        raise MetadataStagingError("redirect X-Repo-Commit differs from revision")
    linked_size = _positive_bounded_int(
        _header_value(response.headers, "X-Linked-Size"),
        "redirect X-Linked-Size",
        maximum_size,
    )
    location = _header_value(response.headers, "Location")
    if location is None:
        raise MetadataStagingError("redirect Location is missing")
    parsed, port = _split_untrusted_url(location, "redirect Location")
    if parsed.scheme != "https":
        raise MetadataStagingError("redirect Location must use HTTPS")
    if (
        parsed.hostname not in _REVIEWED_REDIRECT_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise MetadataStagingError("redirect host is not in the reviewed allowlist")
    if parsed.fragment:
        raise MetadataStagingError("redirect Location must not contain a fragment")
    return location, linked_size


def _validated_metadata_redirect(
    response: HttpResponse,
    *,
    origin_url: str,
    repository: str,
    revision: str,
    filename: str,
) -> str:
    if response.status_code != 307:
        raise MetadataStagingError("metadata redirect must use HTTP 307")
    if _header_value(response.headers, "X-Repo-Commit") != revision:
        raise MetadataStagingError("metadata redirect X-Repo-Commit differs")
    location = _header_value(response.headers, "Location")
    if location is None:
        raise MetadataStagingError("metadata redirect Location is missing")
    _require_visible_ascii_url(location, "metadata redirect Location")
    endpoint = urllib.parse.urljoin(origin_url, location)
    parsed, port = _split_untrusted_url(endpoint, "metadata redirect Location")
    expected_path = f"/api/resolve-cache/models/{repository}/{revision}/{filename}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.path != expected_path
        or not parsed.query
        or parsed.fragment
    ):
        raise MetadataStagingError(
            "metadata redirect is not bound to the immutable HF cache path"
        )
    return endpoint


def _require_repo_commit(response: HttpResponse, revision: str) -> None:
    if _header_value(response.headers, "X-Repo-Commit") != revision:
        raise MetadataStagingError("response X-Repo-Commit differs from revision")


def _require_identity_encoding(response: HttpResponse) -> None:
    encoding = _header_value(response.headers, "Content-Encoding")
    if encoding not in (None, "", "identity"):
        raise MetadataStagingError("HTTP response must use identity content encoding")


def _validate_exact_range_response(
    response: HttpResponse,
    *,
    start: int,
    end: int,
    expected_total: int | None,
) -> int:
    if 300 <= response.status_code < 400:
        raise MetadataStagingError("redirect chain is forbidden")
    if response.status_code != 206:
        raise MetadataStagingError("safetensors range must return HTTP 206")
    expected_length = end - start + 1
    content_length = _positive_bounded_int(
        _header_value(response.headers, "Content-Length"),
        "range Content-Length",
        expected_length,
    )
    if content_length != expected_length:
        raise MetadataStagingError("range Content-Length differs from request")
    if len(response.body) != expected_length:
        raise MetadataStagingError("range response body length differs from request")
    content_range = _header_value(response.headers, "Content-Range")
    match = _CONTENT_RANGE_PATTERN.fullmatch(content_range or "")
    if match is None:
        raise MetadataStagingError("range Content-Range is malformed")
    raw_start, raw_end, raw_total = match.groups()
    observed_start = _bounded_nonnegative_int(
        raw_start,
        "range Content-Range start",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    observed_end = _bounded_nonnegative_int(
        raw_end,
        "range Content-Range end",
        MAX_SAFETENSORS_FILE_BYTES - 1,
    )
    total = _bounded_nonnegative_int(
        raw_total,
        "range Content-Range total",
        MAX_SAFETENSORS_FILE_BYTES,
    )
    if (observed_start, observed_end) != (start, end):
        raise MetadataStagingError("range Content-Range differs from request")
    if total <= end or total > MAX_SAFETENSORS_FILE_BYTES:
        raise MetadataStagingError("range Content-Range total is outside bounds")
    if expected_total is not None and total != expected_total:
        raise MetadataStagingError("range Content-Range total changed between requests")
    _require_identity_encoding(response)
    return total


def _range_response_header_policy(
    *,
    revision: str | None,
    expected_total: int | None,
) -> HttpResponseHeaderPolicy:
    required_values = (("X-Repo-Commit", revision),) if revision is not None else ()
    return HttpResponseHeaderPolicy(
        required_values=required_values,
        require_identity_encoding=True,
        expected_range_total=expected_total,
    )


def _metadata_response_header_policy(
    revision: str | None,
    *,
    filename: str,
    expected_content_length: int | None = None,
) -> HttpResponseHeaderPolicy:
    required_values: tuple[tuple[str, str], ...] = ()
    if revision is not None:
        required_values += (("X-Repo-Commit", revision),)
    if expected_content_length is not None:
        required_values += (("Content-Length", str(expected_content_length)),)
    return HttpResponseHeaderPolicy(
        required_values=required_values,
        require_identity_encoding=True,
        rejected_media_types=frozenset({"application/xhtml+xml", "text/html"}),
        error_context=filename,
    )


def _fetch_range(
    url: str,
    *,
    start: int,
    end: int,
    revision: str,
    transport: HttpTransport,
    authorization: str | None,
    allow_redirect: bool,
    require_repo_commit: bool,
    expected_total: int | None,
) -> tuple[HttpResponse, str, int, bool]:
    response = transport.request(
        url,
        headers=_base_headers(
            authorization=authorization,
            byte_range=(start, end),
        ),
        max_body_bytes=end - start + 1,
        response_header_policy=_range_response_header_policy(
            revision=revision if require_repo_commit else None,
            expected_total=expected_total,
        ),
    )
    redirected = 300 <= response.status_code < 400
    endpoint = url
    linked_size: int | None = None
    if redirected:
        if not allow_redirect:
            raise MetadataStagingError("redirect chain is forbidden")
        endpoint, linked_size = _validated_redirect(
            response,
            revision=revision,
            maximum_size=MAX_SAFETENSORS_FILE_BYTES,
        )
        response = transport.request(
            endpoint,
            headers=_base_headers(authorization=None, byte_range=(start, end)),
            max_body_bytes=end - start + 1,
            response_header_policy=_range_response_header_policy(
                revision=None,
                expected_total=(
                    expected_total if expected_total is not None else linked_size
                ),
            ),
        )
    elif require_repo_commit:
        _require_repo_commit(response, revision)
    total = _validate_exact_range_response(
        response,
        start=start,
        end=end,
        expected_total=expected_total if expected_total is not None else linked_size,
    )
    return response, endpoint, total, redirected


def download_safetensors_header(
    url: str,
    *,
    repository: str,
    revision: str,
    transport: HttpTransport,
    authorization: str | None = None,
) -> SafetensorsHeader:
    """Download exactly the 8-byte length and JSON header byte ranges."""
    filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[1])
    _validate_original_url(
        url,
        repository=repository,
        revision=revision,
        filename=filename,
    )
    length_response, endpoint, total_size, redirected = _fetch_range(
        url,
        start=0,
        end=7,
        revision=revision,
        transport=transport,
        authorization=authorization,
        allow_redirect=True,
        require_repo_commit=True,
        expected_total=None,
    )
    header_length = int.from_bytes(length_response.body, "little", signed=False)
    if (
        header_length <= 0
        or header_length > MAX_SAFETENSORS_HEADER_BYTES
        or header_length % 8 != 0
        or 8 + header_length > total_size
    ):
        raise MetadataStagingError("safetensors header length is outside bounds")
    header_response, _, second_total, _ = _fetch_range(
        endpoint,
        start=8,
        end=7 + header_length,
        revision=revision,
        transport=transport,
        authorization=None if redirected else authorization,
        allow_redirect=False,
        require_repo_commit=not redirected,
        expected_total=total_size,
    )
    if second_total != total_size:
        raise MetadataStagingError("safetensors file size changed between ranges")
    return SafetensorsHeader(
        body=header_response.body,
        header_length=header_length,
        total_size=total_size,
    )


def _reject_json_constant(value: str) -> object:
    raise MetadataStagingError(f"JSON non-finite number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataStagingError(f"JSON object repeats key: {key}")
        result[key] = value
    return result


def _parse_bounded_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > MAX_JSON_INTEGER_DECIMAL_DIGITS:
        raise MetadataStagingError("JSON integer exceeds the reviewed decimal bound")
    return int(value)


def _parse_bounded_json_float(value: str) -> float:
    if len(value) > MAX_JSON_FLOAT_CHARACTERS:
        raise MetadataStagingError("JSON float exceeds the reviewed character bound")
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        raise MetadataStagingError("JSON float is invalid") from None
    if not math.isfinite(parsed):
        raise MetadataStagingError("JSON float must be finite")
    return parsed


def _parse_json_object(raw: bytes, field: str) -> Mapping[str, object]:
    lowered = raw.lstrip()[:32].lower()
    if lowered.startswith((b"<html", b"<!doctype html")):
        raise MetadataStagingError(f"{field} returned HTML instead of JSON")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_bounded_json_float,
            parse_int=_parse_bounded_json_integer,
        )
    except MetadataStagingError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise MetadataStagingError(f"{field} is not valid UTF-8 JSON") from None
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise MetadataStagingError(f"{field} must be a JSON object")
    return cast(Mapping[str, object], value)


def _download_metadata_file(
    *,
    repository: str,
    revision: str,
    filename: str,
    maximum_size: int,
    transport: HttpTransport,
    authorization: str | None,
) -> bytes:
    url = _resource_url(repository, revision, filename)
    response = transport.request(
        url,
        headers=_base_headers(authorization=authorization, byte_range=None),
        max_body_bytes=maximum_size,
        response_header_policy=_metadata_response_header_policy(
            revision,
            filename=filename,
        ),
    )
    linked_size: int | None = None
    if response.status_code == 307:
        try:
            endpoint = _validated_metadata_redirect(
                response,
                origin_url=url,
                repository=repository,
                revision=revision,
                filename=filename,
            )
        except MetadataStagingError as error:
            raise MetadataStagingError(
                f"{filename} metadata redirect HTTP 307 rejected: {error}"
            ) from None
    elif response.status_code == 302:
        try:
            endpoint, linked_size = _validated_redirect(
                response,
                revision=revision,
                maximum_size=maximum_size,
            )
        except MetadataStagingError as error:
            raise MetadataStagingError(
                f"{filename} metadata redirect HTTP 302 rejected: {error}"
            ) from None
    elif 300 <= response.status_code < 400:
        raise MetadataStagingError(
            f"{filename} metadata redirect returned unsupported "
            f"HTTP {response.status_code}"
        )
    else:
        endpoint = None
        _require_repo_commit(response, revision)
    if endpoint is not None:
        response = transport.request(
            endpoint,
            headers=_base_headers(authorization=None, byte_range=None),
            max_body_bytes=linked_size if linked_size is not None else maximum_size,
            response_header_policy=_metadata_response_header_policy(
                None,
                filename=filename,
                expected_content_length=linked_size,
            ),
        )
        if 300 <= response.status_code < 400:
            raise MetadataStagingError(
                f"{filename} redirect chain returned HTTP {response.status_code}"
            )
    if response.status_code != 200:
        raise MetadataStagingError(
            f"{filename} must return HTTP 200, got {response.status_code}"
        )
    content_length = _positive_bounded_int(
        _header_value(response.headers, "Content-Length"),
        f"{filename} Content-Length",
        linked_size if linked_size is not None else maximum_size,
    )
    if linked_size is not None and content_length != linked_size:
        raise MetadataStagingError(
            f"{filename} Content-Length differs from redirect X-Linked-Size"
        )
    if content_length != len(response.body):
        raise MetadataStagingError(f"{filename} response body length differs")
    _require_identity_encoding(response)
    content_type = _header_value(response.headers, "Content-Type") or ""
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type in {"application/xhtml+xml", "text/html"}:
        raise MetadataStagingError(f"{filename} returned HTML instead of JSON")
    return response.body


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_hash(raw: bytes, expected: object, field: str) -> None:
    if not isinstance(expected, str) or _sha256(raw) != expected:
        raise MetadataStagingError(f"{field} differs from immutable pin")


def _index_weight_map(index: Mapping[str, object]) -> dict[str, str]:
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, Mapping) or not raw_weight_map:
        raise MetadataStagingError("index.weight_map must be a non-empty object")
    weight_map: dict[str, str] = {}
    for tensor_name, raw_shard in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise MetadataStagingError("index contains an invalid tensor name")
        weight_map[tensor_name] = _validated_shard_basename(raw_shard)
    return weight_map


def _positive_shape(value: object, tensor_name: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in value
    ):
        raise MetadataStagingError(f"header shape is invalid for {tensor_name}")
    return cast(list[int], value)


def _data_offsets(value: object, tensor_name: str, payload_size: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise MetadataStagingError(f"header data_offsets are invalid for {tensor_name}")
    start, end = cast(list[int], value)
    if start < 0 or end < start or end > payload_size:
        raise MetadataStagingError(
            f"header data_offsets are out of bounds for {tensor_name}"
        )


def _parse_safetensors_header(
    header: SafetensorsHeader,
    *,
    shard: str,
) -> dict[str, dict[str, object]]:
    document = _parse_json_object(header.body, f"{shard} header")
    raw_metadata = document.get("__metadata__")
    if raw_metadata is not None and (
        not isinstance(raw_metadata, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_metadata.items()
        )
    ):
        raise MetadataStagingError(f"{shard} __metadata__ is malformed")
    payload_size = header.total_size - 8 - header.header_length
    tensors: dict[str, dict[str, object]] = {}
    for tensor_name, raw_tensor in document.items():
        if tensor_name == "__metadata__":
            continue
        if not tensor_name:
            raise MetadataStagingError(f"{shard} contains an empty tensor name")
        if not isinstance(raw_tensor, Mapping) or set(raw_tensor) != {
            "data_offsets",
            "dtype",
            "shape",
        }:
            raise MetadataStagingError(f"header fields are invalid for {tensor_name}")
        dtype = raw_tensor.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise MetadataStagingError(f"header dtype is invalid for {tensor_name}")
        shape = _positive_shape(raw_tensor.get("shape"), tensor_name)
        _data_offsets(raw_tensor.get("data_offsets"), tensor_name, payload_size)
        tensors[tensor_name] = {"dtype": dtype, "shape": shape, "shard": shard}
    if not tensors:
        raise MetadataStagingError(f"{shard} header contains no tensors")
    return tensors


def _write_staged_file(path: Path, raw: bytes) -> StagedFile:
    if path.is_symlink() or path.exists():
        raise MetadataStagingError(f"staged path already exists: {path}")
    path.write_bytes(raw)
    return StagedFile(path=path, sha256=_sha256(raw))


def stage_checkpoint_metadata(
    spec: CheckpointMetadataArtifactSpec,
    artifact_root: Path,
    *,
    transport: HttpTransport,
    authorization: str | None = None,
) -> tuple[StagedFile, ...]:
    """Validate and stage one immutable HF checkpoint's metadata."""
    if _ARTIFACT_ID_PATTERN.fullmatch(spec.artifact_id) is None:
        raise MetadataStagingError("artifact id is invalid")
    if artifact_root.is_symlink() or artifact_root.exists():
        raise MetadataStagingError("artifact staging destination must not exist")
    artifact = spec.artifact
    repository = artifact.get("repository")
    revision = artifact.get("revision")
    if not isinstance(repository, str) or not isinstance(revision, str):
        raise MetadataStagingError("artifact lacks immutable HF identity")
    _validate_repository(repository)
    _validate_revision(revision)
    config = _download_metadata_file(
        repository=repository,
        revision=revision,
        filename=STAGED_CONFIG_FILENAME,
        maximum_size=MAX_CONFIG_BYTES,
        transport=transport,
        authorization=authorization,
    )
    _require_hash(config, artifact.get("config_sha256"), "config SHA256")
    _parse_json_object(config, STAGED_CONFIG_FILENAME)
    index_raw = _download_metadata_file(
        repository=repository,
        revision=revision,
        filename=STAGED_INDEX_FILENAME,
        maximum_size=MAX_INDEX_BYTES,
        transport=transport,
        authorization=authorization,
    )
    _require_hash(index_raw, artifact.get("index_sha256"), "index SHA256")
    index = _parse_json_object(index_raw, STAGED_INDEX_FILENAME)
    weight_map = _index_weight_map(index)
    shard_names = sorted(set(weight_map.values()))
    if artifact.get("tensors") != len(weight_map):
        raise MetadataStagingError("index tensor count differs from immutable pin")
    if artifact.get("shards") != len(shard_names):
        raise MetadataStagingError("index shard count differs from immutable pin")

    combined_header: dict[str, dict[str, object]] = {}
    observed_header_lengths: dict[str, int] = {}
    expected_header_lengths = artifact.get("mtp_header_byte_lengths")
    if expected_header_lengths is not None and not isinstance(
        expected_header_lengths, Mapping
    ):
        raise MetadataStagingError("pinned header lengths must be an object")
    for shard in shard_names:
        header = download_safetensors_header(
            _resource_url(repository, revision, shard),
            repository=repository,
            revision=revision,
            transport=transport,
            authorization=authorization,
        )
        if expected_header_lengths is not None and shard in expected_header_lengths:
            observed_header_lengths[shard] = header.header_length
        for tensor_name, tensor in _parse_safetensors_header(
            header, shard=shard
        ).items():
            indexed_shard = weight_map.get(tensor_name)
            if indexed_shard != shard:
                raise MetadataStagingError(
                    f"index/header shard mismatch for {tensor_name}"
                )
            if tensor_name in combined_header:
                raise MetadataStagingError(
                    f"duplicate tensor across safetensors headers: {tensor_name}"
                )
            combined_header[tensor_name] = tensor
    if set(combined_header) != set(weight_map):
        raise MetadataStagingError("index/header tensor keys differ")
    header_manifest = _canonical_compact_json(combined_header)
    _require_hash(
        header_manifest,
        artifact.get("header_manifest_sha256"),
        "header manifest SHA256",
    )
    if expected_header_lengths is not None and observed_header_lengths != dict(
        expected_header_lengths
    ):
        raise MetadataStagingError("A95B header byte lengths differ from pin")

    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    if artifact_root.parent.is_symlink():
        raise MetadataStagingError("artifact parent must not be a symlink")
    artifact_root.mkdir()
    staged = [
        _write_staged_file(artifact_root / STAGED_CONFIG_FILENAME, config),
        _write_staged_file(artifact_root / STAGED_INDEX_FILENAME, index_raw),
        _write_staged_file(
            artifact_root / STAGED_HEADER_MANIFEST_FILENAME,
            header_manifest,
        ),
    ]
    if expected_header_lengths is not None:
        staged.append(
            _write_staged_file(
                artifact_root / STAGED_HEADER_LENGTHS_FILENAME,
                _canonical_compact_json(observed_header_lengths),
            )
        )
    return tuple(staged)


def _actual_manifest(staging_root: Path, staged_files: list[StagedFile]) -> bytes:
    entries = [
        f"{item.sha256}  {item.path.relative_to(staging_root).as_posix()}\n"
        for item in staged_files
    ]
    return "".join(sorted(entries)).encode()


def _safe_cleanup_scratch(output_root: Path, scratch: Path) -> None:
    if (
        scratch.parent == output_root
        and scratch.name.startswith(".precision-policy-source-metadata-")
        and scratch.exists()
    ):
        shutil.rmtree(scratch)


def _published_metadata_file_bound(relative_path: str) -> int:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[0] != STAGED_CHECKPOINT_DIRECTORY
        or _ARTIFACT_ID_PATTERN.fullmatch(path.parts[1]) is None
    ):
        raise MetadataStagingError("published tree has an unsupported metadata path")
    bounds = {
        STAGED_CONFIG_FILENAME: MAX_CONFIG_BYTES,
        STAGED_INDEX_FILENAME: MAX_INDEX_BYTES,
        STAGED_HEADER_LENGTHS_FILENAME: MAX_CONFIG_BYTES,
        STAGED_HEADER_MANIFEST_FILENAME: MAX_COMBINED_HEADER_MANIFEST_BYTES,
    }
    try:
        return bounds[path.name]
    except KeyError:
        raise MetadataStagingError(
            "published tree has an unsupported metadata path"
        ) from None


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _consume_bounded_regular_file(
    path: Path,
    *,
    maximum_size: int,
    exact_size: int | None,
    field: str,
    consume: Callable[[bytes], None],
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise MetadataStagingError(f"{field} cannot be read") from None
    processing_failed = False
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise MetadataStagingError(f"{field} must be a regular file")
        if exact_size is not None and opened_stat.st_size != exact_size:
            raise MetadataStagingError(f"{field} differs")
        if opened_stat.st_size > maximum_size:
            raise MetadataStagingError(f"{field} size exceeds its bound")
        bytes_read = 0
        while True:
            read_size = min(
                VERIFICATION_HASH_CHUNK_BYTES,
                maximum_size - bytes_read + 1,
            )
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > maximum_size:
                raise MetadataStagingError(f"{field} size exceeds its bound")
            consume(chunk)
        final_descriptor_stat = os.fstat(descriptor)
        final_path_stat = os.lstat(path)
        opened_identity = _stat_identity(opened_stat)
        if (
            bytes_read != opened_stat.st_size
            or _stat_identity(final_descriptor_stat) != opened_identity
            or _stat_identity(final_path_stat) != opened_identity
        ):
            raise MetadataStagingError(f"{field} changed during verification")
    except MetadataStagingError:
        processing_failed = True
        raise
    except OSError:
        processing_failed = True
        raise MetadataStagingError(f"{field} cannot be read") from None
    except BaseException:
        processing_failed = True
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError:
            if not processing_failed:
                raise MetadataStagingError(f"{field} cannot be closed") from None


def _read_exact_published_manifest(path: Path, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    _consume_bounded_regular_file(
        path,
        maximum_size=expected_size,
        exact_size=expected_size,
        field="published metadata manifest",
        consume=chunks.append,
    )
    return b"".join(chunks)


def _bounded_file_sha256(path: Path, maximum_size: int) -> str:
    digest = hashlib.sha256()
    _consume_bounded_regular_file(
        path,
        maximum_size=maximum_size,
        exact_size=None,
        field="published metadata file",
        consume=digest.update,
    )
    return digest.hexdigest()


def _verify_published_tree(path: Path, expected_manifest: bytes) -> None:
    if path.is_symlink() or not path.is_dir():
        raise MetadataStagingError("published metadata tree is not a plain directory")
    manifest_path = path / STAGED_METADATA_MANIFEST_FILENAME
    if (
        _read_exact_published_manifest(manifest_path, len(expected_manifest))
        != expected_manifest
    ):
        raise MetadataStagingError("published metadata manifest differs")
    expected_paths = {STAGED_METADATA_MANIFEST_FILENAME}
    expected_directories: set[str] = set()
    for line in expected_manifest.decode().splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        maximum_size = _published_metadata_file_bound(relative_path)
        expected_paths.add(relative_path)
        parent = PurePosixPath(relative_path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        candidate = path / relative_path
        if _bounded_file_sha256(candidate, maximum_size) != digest:
            raise MetadataStagingError(
                f"published metadata digest differs: {relative_path}"
            )
    observed_paths: set[str] = set()
    observed_directories: set[str] = set()
    for directory, directory_names, filenames in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            if child.is_symlink():
                raise MetadataStagingError("published metadata tree contains a symlink")
            observed_directories.add(child.relative_to(path).as_posix())
        for name in filenames:
            child = directory_path / name
            if child.is_symlink():
                raise MetadataStagingError("published metadata tree contains a symlink")
            observed_paths.add(child.relative_to(path).as_posix())
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise MetadataStagingError("published metadata tree contains unexpected files")


def stage_source_metadata(
    output_root: Path,
    *,
    transport: HttpTransport | None = None,
    authorization: str | None = None,
) -> Path:
    """Atomically publish the complete pinned metadata tree by content hash."""
    if output_root.is_symlink():
        raise MetadataStagingError("output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    if not output_root.is_dir():
        raise MetadataStagingError("output root must be a directory")
    expected_manifest = expected_staged_metadata_manifest()
    final_path = output_root / f"sha256-{EXPECTED_STAGED_METADATA_MANIFEST_SHA256}"
    if final_path.exists() or final_path.is_symlink():
        _verify_published_tree(final_path, expected_manifest)
        return final_path
    active_transport = transport or UrllibHttpTransport()
    scratch = Path(
        tempfile.mkdtemp(
            prefix=".precision-policy-source-metadata-",
            dir=output_root,
        )
    )
    try:
        staged_files: list[StagedFile] = []
        checkpoints_root = scratch / STAGED_CHECKPOINT_DIRECTORY
        checkpoints_root.mkdir()
        for spec in checkpoint_metadata_artifact_identities():
            staged_files.extend(
                stage_checkpoint_metadata(
                    spec,
                    checkpoints_root / spec.artifact_id,
                    transport=active_transport,
                    authorization=authorization,
                )
            )
        actual_manifest = _actual_manifest(scratch, staged_files)
        if actual_manifest != expected_manifest:
            raise MetadataStagingError("staged 19-file manifest differs from pin")
        _write_staged_file(
            scratch / STAGED_METADATA_MANIFEST_FILENAME,
            actual_manifest,
        )
        try:
            scratch.rename(final_path)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            _verify_published_tree(final_path, expected_manifest)
            _safe_cleanup_scratch(output_root, scratch)
        _verify_published_tree(final_path, expected_manifest)
        return final_path
    except BaseException:
        _safe_cleanup_scratch(output_root, scratch)
        raise


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        published = stage_source_metadata(arguments.output_root)
    except MetadataStagingError as error:
        print(f"metadata staging failed: {error}", file=sys.stderr)
        return 2
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
