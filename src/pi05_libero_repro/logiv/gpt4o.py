from __future__ import annotations

import base64
import binascii
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
import struct
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen as stdlib_urlopen
import zlib

import numpy as np


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class Gpt4oRequestError(RuntimeError):
    """A secret-safe GPT-4o request or response failure."""


@dataclass(frozen=True)
class Gpt4oCallRecord:
    purpose: str
    response_id: str
    model: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    retries: int
    request_sha256: str
    response_sha256: str


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )


def encode_png_data_url(image: np.ndarray) -> str:
    """Encode one RGB uint8 array as a dependency-free PNG data URL."""

    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] <= 0
        or array.shape[1] <= 0
    ):
        raise ValueError("image must be a nonempty HxWx3 RGB uint8 array")
    contiguous = np.ascontiguousarray(array)
    height, width, _ = contiguous.shape
    scanlines = b"".join(
        b"\x00" + contiguous[row].tobytes(order="C") for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


class Gpt4oClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        urlopen: Callable[..., Any] = stdlib_urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not api_key:
            raise Gpt4oRequestError("OPENAI_API_KEY is not set")
        if model != "gpt-4o":
            raise ValueError("GPT-4o LOGIV requires model='gpt-4o'")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries nonnegative")
        self._api_key = api_key
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._urlopen = urlopen
        self._sleeper = sleeper
        self._clock = clock
        self._request_counts: Counter[str] = Counter()
        self._http_attempt_counts: Counter[str] = Counter()
        self._records: list[Gpt4oCallRecord] = []

    @classmethod
    def from_env(cls, **kwargs: Any) -> "Gpt4oClient":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise Gpt4oRequestError("OPENAI_API_KEY is not set")
        return cls(api_key, **kwargs)

    @property
    def request_counts(self) -> dict[str, int]:
        return dict(self._request_counts)

    @property
    def http_attempt_counts(self) -> dict[str, int]:
        return dict(self._http_attempt_counts)

    @property
    def records(self) -> tuple[Gpt4oCallRecord, ...]:
        return tuple(self._records)

    def _payload(
        self,
        *,
        system: str,
        text: str,
        images: Sequence[np.ndarray],
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> bytes:
        if not system or not text or not schema_name:
            raise ValueError("system, text, and schema_name must be nonempty")
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_png_data_url(image),
                    "detail": "low",
                },
            }
            for image in images
        )
        value = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
        }
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _parse_response(raw: bytes, expected_model: str) -> tuple[
        Mapping[str, Any], str, str, int, int
    ]:
        try:
            response = json.loads(raw)
            response_id = response["id"]
            model = response["model"]
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise Gpt4oRequestError("OpenAI returned a malformed response") from error
        if not isinstance(response_id, str) or not response_id:
            raise Gpt4oRequestError("OpenAI returned a malformed response")
        if not isinstance(model, str) or not (
            model == expected_model or model.startswith(expected_model + "-")
        ):
            raise Gpt4oRequestError("OpenAI returned an unexpected model")
        if not isinstance(message, dict):
            raise Gpt4oRequestError("OpenAI returned a malformed response")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            raise Gpt4oRequestError("GPT-4o refused the structured request")
        content = message.get("content")
        if not isinstance(content, str):
            raise Gpt4oRequestError("OpenAI returned a malformed response")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise Gpt4oRequestError("GPT-4o returned invalid structured JSON") from error
        if not isinstance(parsed, dict):
            raise Gpt4oRequestError("GPT-4o returned invalid structured JSON")
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        if type(input_tokens) is not int or input_tokens < 0:
            input_tokens = 0
        if type(output_tokens) is not int or output_tokens < 0:
            output_tokens = 0
        return parsed, response_id, model, input_tokens, output_tokens

    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        text: str,
        images: Sequence[np.ndarray],
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not purpose:
            raise ValueError("request purpose must be nonempty")
        body = self._payload(
            system=system,
            text=text,
            images=images,
            schema_name=schema_name,
            schema=schema,
        )
        request_hash = hashlib.sha256(body).hexdigest()
        self._request_counts[purpose] += 1
        started = self._clock()
        last_failure = "transport error"
        for retry in range(self.max_retries + 1):
            self._http_attempt_counts[purpose] += 1
            request = Request(
                OPENAI_CHAT_COMPLETIONS_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self._urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
            except HTTPError as error:
                last_failure = f"HTTP {error.code}"
                transient = error.code in {408, 409, 429} or error.code >= 500
                if not transient:
                    raise Gpt4oRequestError(
                        f"OpenAI API request failed ({last_failure})"
                    ) from None
            except (TimeoutError, URLError, OSError):
                last_failure = "transport error"
            else:
                parsed, response_id, model, input_tokens, output_tokens = (
                    self._parse_response(raw, self.model)
                )
                elapsed = self._clock() - started
                self._records.append(
                    Gpt4oCallRecord(
                        purpose=purpose,
                        response_id=response_id,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        elapsed_seconds=elapsed,
                        retries=retry,
                        request_sha256=request_hash,
                        response_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
                return parsed
            if retry < self.max_retries:
                self._sleeper(0.5 * (2**retry))
        attempts = self.max_retries + 1
        raise Gpt4oRequestError(
            f"OpenAI API request failed after {attempts} attempts ({last_failure})"
        )
