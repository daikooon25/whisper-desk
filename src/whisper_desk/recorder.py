"""マイク録音ロジック (flet非依存)"""

import logging
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import SAMPLE_RATE

logger = logging.getLogger(__name__)


class RecorderError(Exception):
    """録音関連のエラー"""


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = False
        self._audio_data: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._start_time: float = 0

    def start(self) -> None:
        """録音を開始する。マイクが利用できない場合は RecorderError を送出。"""
        self._audio_data = []
        self._start_time = time.time()

        def callback(indata, frames, time_info, status):
            if self.recording:
                self._audio_data.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            self.recording = True
            logger.info("録音開始")
        except (sd.PortAudioError, OSError) as e:
            logger.error("マイク初期化エラー: %s", e)
            raise RecorderError(f"マイクが利用できません: {e}") from e

    def stop(self) -> str | None:
        """録音を停止し、WAVファイルのパスを返す。データがない場合はNone。"""
        self.recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio_data:
            logger.info("録音データなし")
            return None

        audio = np.concatenate(self._audio_data, axis=0)
        duration = len(audio) / self.sample_rate
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, self.sample_rate)
        logger.info("録音完了: %.1f秒 -> %s", duration, tmp.name)
        return tmp.name

    @property
    def elapsed_seconds(self) -> float:
        """録音開始からの経過秒数。"""
        if not self.recording:
            return 0.0
        return time.time() - self._start_time

    @property
    def duration(self) -> float | None:
        """最後に録音した音声の長さ(秒)。録音データがない場合はNone。"""
        if not self._audio_data:
            return None
        total_samples = sum(chunk.shape[0] for chunk in self._audio_data)
        return total_samples / self.sample_rate
