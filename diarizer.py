"""pyannote-audio による話者分離ロジック (flet非依存)"""

import asyncio
import dataclasses
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from pyannote.audio import Pipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


@dataclasses.dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str


def _assign_speakers(
    whisper_segments: list[dict],
    diarization,
) -> list[DiarizedSegment]:
    """Whisper セグメントに pyannote の話者ラベルを割り当てる。"""
    # pyannote の話者ターンを取得
    turns = list(diarization.itertracks(yield_label=True))

    # 話者ID → 表示名マッピング (SPEAKER_00 -> Speaker A)
    speaker_ids: dict[str, str] = {}

    results = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"].strip()
        if not seg_text:
            continue

        # 各話者との重複時間を計算
        overlaps: dict[str, float] = {}
        for turn, _, speaker_id in turns:
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > 0:
                overlaps[speaker_id] = overlaps.get(speaker_id, 0.0) + overlap

        # 最も重複の大きい話者を割り当て
        if overlaps:
            best_speaker_id = max(overlaps, key=overlaps.get)
        else:
            best_speaker_id = "UNKNOWN"

        # 表示名に変換
        if best_speaker_id not in speaker_ids:
            idx = len(speaker_ids)
            speaker_ids[best_speaker_id] = f"Speaker {chr(ord('A') + idx)}"
        speaker_name = speaker_ids[best_speaker_id]

        results.append(
            DiarizedSegment(
                speaker=speaker_name,
                start=seg_start,
                end=seg_end,
                text=seg_text,
            )
        )

    return results


def format_diarized_text(segments: list[DiarizedSegment]) -> str:
    """話者分離結果を表示用テキストにフォーマットする。"""
    lines = []
    current_speaker = None
    for seg in segments:
        if seg.speaker != current_speaker:
            current_speaker = seg.speaker
            lines.append(f"\n[{seg.speaker}]")
        lines.append(seg.text)
    return "\n".join(lines).strip()


class Diarizer:
    def __init__(self):
        self._pipeline = None

    async def ensure_pipeline(self, hf_token: str) -> None:
        """pyannote パイプラインをロードする (キャッシュ済みならスキップ)。"""
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError(
                "pyannote.audio がインストールされていません。"
                " `uv sync --extra diarize` でインストールしてください。"
            )

        if self._pipeline is not None:
            return

        logger.info("話者分離パイプライン読込開始")

        def _load():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(device)
            logger.info("話者分離パイプライン読込完了 (device=%s)", device)
            return pipeline

        self._pipeline = await asyncio.to_thread(_load)

    async def diarize(
        self, audio_path: str, whisper_segments: list[dict]
    ) -> list[DiarizedSegment]:
        """音声ファイルに対して話者分離を実行し、Whisper セグメントと統合する。"""
        if self._pipeline is None:
            raise RuntimeError(
                "パイプラインが読み込まれていません。ensure_pipeline() を先に呼んでください。"
            )

        logger.info("話者分離開始: %s", audio_path)
        diarization = await asyncio.to_thread(self._pipeline, audio_path)
        segments = _assign_speakers(whisper_segments, diarization)
        logger.info("話者分離完了: %d セグメント", len(segments))
        return segments
