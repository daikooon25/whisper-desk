"""transcriber.py のユニットテスト"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from transcriber import Transcriber, TranscriptionResult


@pytest.fixture()
def transcriber():
    return Transcriber()


class TestTranscriberInit:
    def test_no_model_loaded(self, transcriber: Transcriber):
        assert transcriber.is_model_loaded is False


class TestEnsureModel:
    def test_loads_model(self, transcriber: Transcriber):
        with patch("transcriber.whisper.load_model", return_value=MagicMock()) as mock_load:
            asyncio.get_event_loop().run_until_complete(
                transcriber.ensure_model("base")
            )
            mock_load.assert_called_once_with("base")
            assert transcriber.is_model_loaded is True

    def test_skips_if_same_model(self, transcriber: Transcriber):
        mock_model = MagicMock()
        with patch("transcriber.whisper.load_model", return_value=mock_model) as mock_load:
            asyncio.get_event_loop().run_until_complete(
                transcriber.ensure_model("base")
            )
            asyncio.get_event_loop().run_until_complete(
                transcriber.ensure_model("base")
            )
            assert mock_load.call_count == 1

    def test_reloads_on_different_model(self, transcriber: Transcriber):
        with patch("transcriber.whisper.load_model", return_value=MagicMock()) as mock_load:
            asyncio.get_event_loop().run_until_complete(
                transcriber.ensure_model("base")
            )
            asyncio.get_event_loop().run_until_complete(
                transcriber.ensure_model("small")
            )
            assert mock_load.call_count == 2


class TestTranscribe:
    def test_raises_without_model(self, transcriber: Transcriber):
        with pytest.raises(RuntimeError, match="モデルが読み込まれていません"):
            asyncio.get_event_loop().run_until_complete(
                transcriber.transcribe("test.wav")
            )

    def test_returns_result(self, transcriber: Transcriber):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "  hello world  ",
            "language": "en",
        }
        transcriber._model = mock_model
        transcriber._loaded_model_name = "base"

        with patch("transcriber.sf.info") as mock_info:
            mock_info.return_value.duration = 5.0
            result = asyncio.get_event_loop().run_until_complete(
                transcriber.transcribe("test.wav", "en")
            )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.detected_language == "en"
        assert result.duration_sec == 5.0
        assert result.elapsed_sec > 0

    def test_auto_language(self, transcriber: Transcriber):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "テスト", "language": "ja"}
        transcriber._model = mock_model
        transcriber._loaded_model_name = "base"

        with patch("transcriber.sf.info") as mock_info:
            mock_info.return_value.duration = 1.0
            asyncio.get_event_loop().run_until_complete(
                transcriber.transcribe("test.wav", "auto")
            )

        mock_model.transcribe.assert_called_once_with("test.wav")
