import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinyserve.api.app import app

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason="test GGUF model not present locally"),
]


def _set_model_env() -> None:
    os.environ["TINYSERVE_MODEL_PATH"] = str(MODEL_PATH)


def test_generate_non_streaming() -> None:
    _set_model_env()
    payload = {"prompt": "The capital of France is", "max_tokens": 8, "stream": False}

    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 0


def test_generate_streaming() -> None:
    _set_model_env()
    payload = {"prompt": "The capital of France is", "max_tokens": 8, "stream": True}

    with TestClient(app) as client, client.stream("POST", "/generate", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert len(events) > 0
