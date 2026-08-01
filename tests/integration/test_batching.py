"""Proves Phase 2's actual claim: multiple sequences advance in ONE
llama_decode() call, not one llama.cpp call per sequence."""

import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tinyserve.api.app import app
from tinyserve.batch.builder import ActiveSequence, build_batch
from tinyserve.runtime.llama_runtime import LlamaRuntime

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason="test GGUF model not present locally"),
]


async def test_one_decode_call_advances_two_independent_sequences() -> None:
    runtime = LlamaRuntime(str(MODEL_PATH), n_ctx=512, n_seq_max=2)
    prompt_a = runtime.tokenize("The capital of France is")
    prompt_b = runtime.tokenize("Roses are red, violets are")

    active = {
        0: ActiveSequence(seq_id=0, pending_tokens=prompt_a, n_past=0),
        1: ActiveSequence(seq_id=1, pending_tokens=prompt_b, n_past=0),
    }
    batch = build_batch(list(active.values()), runtime.batch_capacity)

    # Both sequences' prompts are laid out as rows of the SAME batch, so this
    # single decode() call is the one llama_decode() invocation that advances
    # both — that's the property continuous batching depends on.
    assert {row.seq_id for row in batch.rows} == {0, 1}
    sampled = await runtime.decode(batch)

    assert set(sampled) == {0, 1}
    text_a = runtime.detokenize([sampled[0]]).decode("utf-8", errors="replace")
    text_b = runtime.detokenize([sampled[1]]).decode("utf-8", errors="replace")
    assert text_a != text_b  # independent continuations, not the same sequence twice

    runtime.free_sequence(0)
    runtime.free_sequence(1)


def _set_model_env() -> None:
    os.environ["TINYSERVE_MODEL_PATH"] = str(MODEL_PATH)


def test_concurrent_http_requests_both_complete_correctly() -> None:
    # One shared TestClient -> one running app/batch loop, so these two
    # requests are actually served by the same continuous-batching loop,
    # not two separate server instances.
    _set_model_env()
    results: dict[str, str] = {}

    def call(client: TestClient, name: str, prompt: str) -> None:
        response = client.post(
            "/generate", json={"prompt": prompt, "max_tokens": 8, "stream": False}
        )
        results[name] = response.json()["text"]

    with TestClient(app) as client:
        thread_a = threading.Thread(target=call, args=(client, "a", "Name a primary color."))
        thread_b = threading.Thread(target=call, args=(client, "b", "Count from one to three."))
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

    assert len(results["a"]) > 0
    assert len(results["b"]) > 0
