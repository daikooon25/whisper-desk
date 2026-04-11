"""recorder.py のユニットテスト"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from whisper_desk.recorder import Recorder, RecorderError


class TestRecorderInit:
    def test_defaults(self):
        r = Recorder()
        assert r.recording is False
        assert r.sample_rate == 16000

    def test_custom_sample_rate(self):
        r = Recorder(sample_rate=44100)
        assert r.sample_rate == 44100


class TestRecorderStart:
    def test_start_sets_recording(self):
        r = Recorder()
        with patch("whisper_desk.recorder.sd.InputStream") as mock_stream:
            mock_stream.return_value = MagicMock()
            r.start()
            assert r.recording is True
            mock_stream.return_value.start.assert_called_once()

    def test_start_raises_on_mic_error(self):
        r = Recorder()
        with patch("whisper_desk.recorder.sd.InputStream", side_effect=OSError("no mic")):
            with pytest.raises(RecorderError, match="マイクが利用できません"):
                r.start()
            assert r.recording is False


class TestRecorderStop:
    def test_stop_without_data_returns_none(self):
        r = Recorder()
        r.recording = True
        r._stream = MagicMock()
        result = r.stop()
        assert result is None
        assert r.recording is False

    def test_stop_with_data_returns_path(self, tmp_path):
        r = Recorder()
        r.recording = True
        r._stream = MagicMock()
        r._audio_data = [np.zeros((1600, 1), dtype="float32")]
        path = r.stop()
        assert path is not None
        assert path.endswith(".wav")


class TestRecorderElapsed:
    def test_elapsed_when_not_recording(self):
        r = Recorder()
        assert r.elapsed_seconds == 0.0

    def test_elapsed_when_recording(self):
        r = Recorder()
        r.recording = True
        r._start_time = 0  # epoch
        assert r.elapsed_seconds > 0


class TestRecorderDuration:
    def test_duration_no_data(self):
        r = Recorder()
        assert r.duration is None

    def test_duration_with_data(self):
        r = Recorder(sample_rate=16000)
        r._audio_data = [np.zeros((16000, 1), dtype="float32")]
        assert r.duration == pytest.approx(1.0)
