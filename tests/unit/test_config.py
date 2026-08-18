import pytest
from pydantic import ValidationError

from tinyserve.config import Settings


def test_settings_defaults() -> None:
    settings = Settings(model_path="model.gguf")

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TINYSERVE_PORT", "9001")

    settings = Settings(model_path="model.gguf")

    assert settings.port == 9001


def test_settings_model_path_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear the env explicitly rather than trusting it to be unset. The
    # integration tests set TINYSERVE_MODEL_PATH via os.environ and never undo
    # it, so this test passed or failed depending on whether it ran before or
    # after them in the same session -- invisible in CI, which runs tests/unit
    # alone, and a confusing failure for anyone running the whole suite.
    monkeypatch.delenv("TINYSERVE_MODEL_PATH", raising=False)
    with pytest.raises(ValidationError):
        Settings()
