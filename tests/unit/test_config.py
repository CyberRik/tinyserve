from tinyserve.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TINYSERVE_PORT", "9001")

    settings = Settings()

    assert settings.port == 9001
