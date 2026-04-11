"""diarizer.py のユニットテスト"""

from unittest.mock import MagicMock

from diarizer import DiarizedSegment, _assign_speakers, format_diarized_text


def _make_turn(start, end):
    """pyannote Segment のモックを作成。"""
    seg = MagicMock()
    seg.start = start
    seg.end = end
    return seg


class TestAssignSpeakers:
    def test_single_speaker(self):
        whisper_segments = [
            {"start": 0.0, "end": 2.0, "text": "こんにちは"},
            {"start": 2.0, "end": 4.0, "text": "元気ですか"},
        ]
        diarization = MagicMock()
        diarization.itertracks.return_value = [
            (_make_turn(0.0, 4.0), None, "SPEAKER_00"),
        ]

        result = _assign_speakers(whisper_segments, diarization)
        assert len(result) == 2
        assert all(seg.speaker == "Speaker A" for seg in result)

    def test_two_speakers(self):
        whisper_segments = [
            {"start": 0.0, "end": 2.0, "text": "はい"},
            {"start": 2.5, "end": 4.0, "text": "いいえ"},
        ]
        diarization = MagicMock()
        diarization.itertracks.return_value = [
            (_make_turn(0.0, 2.0), None, "SPEAKER_00"),
            (_make_turn(2.5, 4.0), None, "SPEAKER_01"),
        ]

        result = _assign_speakers(whisper_segments, diarization)
        assert result[0].speaker == "Speaker A"
        assert result[1].speaker == "Speaker B"

    def test_no_overlap_assigns_unknown(self):
        whisper_segments = [
            {"start": 10.0, "end": 12.0, "text": "遅い発話"},
        ]
        diarization = MagicMock()
        diarization.itertracks.return_value = [
            (_make_turn(0.0, 2.0), None, "SPEAKER_00"),
        ]

        result = _assign_speakers(whisper_segments, diarization)
        assert len(result) == 1
        assert "Speaker" in result[0].speaker  # UNKNOWN -> Speaker A

    def test_empty_text_skipped(self):
        whisper_segments = [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": "有効"},
        ]
        diarization = MagicMock()
        diarization.itertracks.return_value = [
            (_make_turn(0.0, 2.0), None, "SPEAKER_00"),
        ]

        result = _assign_speakers(whisper_segments, diarization)
        assert len(result) == 1
        assert result[0].text == "有効"

    def test_overlap_picks_best_speaker(self):
        whisper_segments = [
            {"start": 1.0, "end": 3.0, "text": "テスト"},
        ]
        diarization = MagicMock()
        diarization.itertracks.return_value = [
            (_make_turn(0.0, 1.5), None, "SPEAKER_00"),  # 0.5s overlap
            (_make_turn(1.5, 4.0), None, "SPEAKER_01"),  # 1.5s overlap
        ]

        result = _assign_speakers(whisper_segments, diarization)
        assert result[0].speaker == "Speaker A"  # SPEAKER_01 has more overlap -> first assigned


class TestFormatDiarizedText:
    def test_basic_format(self):
        segments = [
            DiarizedSegment("Speaker A", 0.0, 2.0, "こんにちは"),
            DiarizedSegment("Speaker B", 2.0, 4.0, "はい"),
            DiarizedSegment("Speaker A", 4.0, 6.0, "ありがとう"),
        ]
        text = format_diarized_text(segments)
        assert "[Speaker A]" in text
        assert "[Speaker B]" in text
        assert "こんにちは" in text
        assert "はい" in text
        assert "ありがとう" in text

    def test_consecutive_same_speaker_no_duplicate_label(self):
        segments = [
            DiarizedSegment("Speaker A", 0.0, 1.0, "一"),
            DiarizedSegment("Speaker A", 1.0, 2.0, "二"),
        ]
        text = format_diarized_text(segments)
        assert text.count("[Speaker A]") == 1

    def test_empty_segments(self):
        assert format_diarized_text([]) == ""
