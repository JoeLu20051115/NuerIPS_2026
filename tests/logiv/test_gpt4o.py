from __future__ import annotations

import base64
import io
import json
from urllib.error import HTTPError

import numpy as np
import pytest

from pi05_libero_repro.logiv.gpt4o import (
    Gpt4oClient,
    Gpt4oRequestError,
    encode_png_data_url,
)


SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["OK"]}},
    "required": ["status"],
    "additionalProperties": False,
}


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


def _completion(
    *,
    content: str = '{"status":"OK"}',
    model: str = "gpt-4o-2024-08-06",
    refusal: str | None = None,
) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def test_client_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(Gpt4oRequestError, match="OPENAI_API_KEY is not set"):
        Gpt4oClient.from_env()


def test_png_data_url_has_valid_png_signature() -> None:
    url = encode_png_data_url(np.zeros((2, 3, 3), dtype=np.uint8))

    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_png_data_url_rejects_non_rgb_uint8_images(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        encode_png_data_url(value)


def test_complete_json_sends_gpt4o_images_and_strict_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-never-record")
    captured = []

    def urlopen(request, *, timeout):
        captured.append((request, timeout))
        return _Response(_completion())

    client = Gpt4oClient.from_env(
        timeout_seconds=12.5,
        urlopen=urlopen,
        sleeper=lambda _seconds: None,
        clock=iter((5.0, 5.25)).__next__,
    )
    result = client.complete_json(
        purpose="state_gate",
        system="Return the requested state.",
        text="Inspect both images.",
        images=(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ),
        schema_name="state_gate_v1",
        schema=SCHEMA,
    )

    assert result == {"status": "OK"}
    assert len(captured) == 1
    request, timeout = captured[0]
    assert timeout == 12.5
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-secret-never-record"
    payload = json.loads(request.data)
    assert payload["model"] == "gpt-4o"
    assert payload["temperature"] == 0
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "state_gate_v1",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    content = payload["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url", "image_url"]
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,")
        and item["image_url"]["detail"] == "low"
        for item in content[1:]
    )
    assert client.request_counts == {"state_gate": 1}
    assert client.http_attempt_counts == {"state_gate": 1}
    record = client.records[0]
    assert record.purpose == "state_gate"
    assert record.response_id == "chatcmpl-test"
    assert record.model == "gpt-4o-2024-08-06"
    assert (record.input_tokens, record.output_tokens) == (11, 3)
    assert record.elapsed_seconds == 0.25
    assert record.retries == 0
    assert len(record.request_sha256) == len(record.response_sha256) == 64
    assert "secret" not in json.dumps(record.__dict__)


def test_client_can_request_high_detail_for_small_robot_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = []

    def urlopen(request, *, timeout):
        captured.append(json.loads(request.data))
        return _Response(_completion())

    client = Gpt4oClient.from_env(
        image_detail="high",
        urlopen=urlopen,
        clock=iter((0.0, 0.1)).__next__,
    )
    client.complete_json(
        purpose="state_gate",
        system="s",
        text="t",
        images=(np.zeros((2, 2, 3), dtype=np.uint8),),
        schema_name="facts",
        schema=SCHEMA,
    )

    image = captured[0]["messages"][1]["content"][1]
    assert image["image_url"]["detail"] == "high"


def test_client_rejects_unknown_image_detail() -> None:
    with pytest.raises(ValueError, match="image_detail"):
        Gpt4oClient("sk-test", image_detail="ultra")


def test_complete_json_retries_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = 0
    sleeps = []

    def urlopen(request, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                429,
                "rate limited",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"do not surface"}}'),
            )
        return _Response(_completion())

    client = Gpt4oClient.from_env(
        max_retries=1,
        urlopen=urlopen,
        sleeper=sleeps.append,
        clock=iter((0.0, 0.5)).__next__,
    )

    assert client.complete_json(
        purpose="state_gate",
        system="s",
        text="t",
        images=(),
        schema_name="smoke",
        schema=SCHEMA,
    ) == {"status": "OK"}
    assert calls == 2
    assert sleeps == [0.5]
    assert client.request_counts == {"state_gate": 1}
    assert client.http_attempt_counts == {"state_gate": 2}
    assert client.records[0].retries == 1


def test_complete_json_timeout_exhaustion_is_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-this-must-never-appear"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def urlopen(_request, *, timeout):
        raise TimeoutError(f"timeout {timeout}: {secret}")

    client = Gpt4oClient.from_env(
        max_retries=1,
        urlopen=urlopen,
        sleeper=lambda _seconds: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(Gpt4oRequestError) as captured:
        client.complete_json(
            purpose="state_gate",
            system="s",
            text="t",
            images=(),
            schema_name="smoke",
            schema=SCHEMA,
        )
    assert secret not in str(captured.value)
    assert "after 2 attempts" in str(captured.value)
    assert client.request_counts == {"state_gate": 1}
    assert client.http_attempt_counts == {"state_gate": 2}
    assert client.records == ()


@pytest.mark.parametrize("purpose", ["initial_plan", "local_repair"])
def test_logiv_gpt4o_transport_rejects_non_vlm_purposes(purpose: str) -> None:
    opened = []
    client = Gpt4oClient(
        "secret", urlopen=lambda *args, **kwargs: opened.append(args)
    )

    with pytest.raises(Gpt4oRequestError, match="state_gate"):
        client.complete_json(
            purpose=purpose,
            system="system",
            text="facts",
            images=(),
            schema_name="facts",
            schema={"type": "object"},
        )

    assert opened == []
    assert client.request_counts == {}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_completion(refusal="cannot comply"), "refused"),
        (_completion(content="not-json"), "invalid structured JSON"),
        (_completion(model="gpt-5.6"), "unexpected model"),
        ({"id": "x", "model": "gpt-4o", "choices": []}, "malformed response"),
    ],
)
def test_complete_json_rejects_unusable_responses(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    message: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = Gpt4oClient.from_env(
        urlopen=lambda _request, timeout: _Response(payload),
        clock=iter((0.0, 0.1)).__next__,
    )

    with pytest.raises(Gpt4oRequestError, match=message):
        client.complete_json(
            purpose="state_gate",
            system="s",
            text="t",
            images=(),
            schema_name="smoke",
            schema=SCHEMA,
        )
