"""config.py の定数・ユーティリティ関数のテスト"""

from pathlib import Path
from unittest.mock import patch

from whisper_desk.config import (
    LANGUAGES,
    MODELS,
    SAMPLE_RATE,
    is_model_downloaded,
    load_hf_token,
    save_hf_token,
)

# ── 定数 ─────────────────────────────────────────────


class TestConstants:
    def test_models_has_expected_keys(self):
        expected = {"tiny", "base", "small", "medium", "large", "turbo"}
        assert set(MODELS.keys()) == expected

    def test_models_values_are_tuples(self):
        for name, val in MODELS.items():
            assert isinstance(val, tuple) and len(val) == 2, f"{name}: {val}"

    def test_languages_has_auto(self):
        assert "auto" in LANGUAGES

    def test_languages_has_japanese(self):
        assert "ja" in LANGUAGES

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000


# ── is_model_downloaded ──────────────────────────────


class TestIsModelDownloaded:
    def test_returns_false_when_not_exists(self, tmp_path: Path):
        with patch("whisper_desk.config.WHISPER_CACHE_DIR", tmp_path):
            assert is_model_downloaded("base") is False

    def test_returns_true_when_exists(self, tmp_path: Path):
        (tmp_path / "base.pt").touch()
        with patch("whisper_desk.config.WHISPER_CACHE_DIR", tmp_path):
            assert is_model_downloaded("base") is True


# ── HuggingFace トークン ─────────────────────────────


class TestHfToken:
    def test_load_returns_none_when_no_file(self, tmp_path: Path):
        with patch("whisper_desk.config.HF_TOKEN_PATH", tmp_path / "hf_token"):
            assert load_hf_token() is None

    def test_save_and_load(self, tmp_path: Path):
        token_path = tmp_path / "hf_token"
        with patch("whisper_desk.config.HF_TOKEN_PATH", token_path):
            save_hf_token("hf_test_token_123")
            assert load_hf_token() == "hf_test_token_123"

    def test_load_strips_whitespace(self, tmp_path: Path):
        token_path = tmp_path / "hf_token"
        token_path.write_text("  hf_abc  \n", encoding="utf-8")
        with patch("whisper_desk.config.HF_TOKEN_PATH", token_path):
            assert load_hf_token() == "hf_abc"

    def test_load_returns_none_for_empty_file(self, tmp_path: Path):
        token_path = tmp_path / "hf_token"
        token_path.write_text("", encoding="utf-8")
        with patch("whisper_desk.config.HF_TOKEN_PATH", token_path):
            assert load_hf_token() is None
