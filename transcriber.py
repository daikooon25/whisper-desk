"""Whisper 文字起こしロジック (flet非依存)"""

import asyncio
import dataclasses
import logging
import time

import soundfile as sf
import whisper

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranscriptionResult:
    text: str
    detected_language: str
    duration_sec: float | None
    elapsed_sec: float


class Transcriber:
    def __init__(self):
        self._model = None
        self._loaded_model_name: str | None = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    async def ensure_model(self, model_name: str) -> None:
        """モデルが未ロードまたは別モデルの場合にロードする。"""
        if self._model is not None and self._loaded_model_name == model_name:
            return
        logger.info("モデル読込開始: %s", model_name)
        t0 = time.time()
        self._model = await asyncio.to_thread(whisper.load_model, model_name)
        self._loaded_model_name = model_name
        logger.info("モデル読込完了: %s (%.1f秒)", model_name, time.time() - t0)

    async def transcribe(self, audio_path: str, language: str = "auto") -> TranscriptionResult:
        """音声ファイルを文字起こしする。事前に ensure_model() を呼ぶこと。"""
        if self._model is None:
            raise RuntimeError(
                "モデルが読み込まれていません。ensure_model() を先に呼んでください。"
            )

        t0 = time.time()
        logger.info("文字起こし開始: file=%s, lang=%s", audio_path, language)

        transcribe_opts = {}
        if language != "auto":
            transcribe_opts["language"] = language

        result = await asyncio.to_thread(
            self._model.transcribe, audio_path, **transcribe_opts
        )

        elapsed = time.time() - t0
        text = result["text"].strip()
        detected_lang = result.get("language", "不明")

        # 音声ファイルの長さを取得
        try:
            info = sf.info(audio_path)
            duration_sec = round(info.duration, 1)
        except Exception:
            duration_sec = None

        logger.info(
            "文字起こし完了: lang=%s, %.1f秒, %d文字",
            detected_lang,
            elapsed,
            len(text),
        )

        return TranscriptionResult(
            text=text,
            detected_language=detected_lang,
            duration_sec=duration_sec,
            elapsed_sec=elapsed,
        )
