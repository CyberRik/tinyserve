import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinyserve.api.app import app

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="test GGUF model not present locally")
def test_generate_end_to_end() -> None:
    os.environ["TINYSERVE_MODEL_PATH"] = str(MODEL_PATH)

    payload = {"prompt": "The capital of France is", "max_tokens": 8}
    with TestClient(app) as client:
        response = client.post("/generate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prompt_tokens"] > 0
    assert body["completion_tokens"] > 0
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 0
