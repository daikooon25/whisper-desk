"""config.py の定数・ユーティリティ関数のテスト"""

from pathlib import Path
from unittest.mock import patch

from config import LANGUAGES, MODELS, SAMPLE_RATE, is_model_downloaded

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
        with patch("config.WHISPER_CACHE_DIR", tmp_path):
            assert is_model_downloaded("base") is False

    def test_returns_true_when_exists(self, tmp_path: Path):
        (tmp_path / "base.pt").touch()
        with patch("config.WHISPER_CACHE_DIR", tmp_path):
            assert is_model_downloaded("base") is True
