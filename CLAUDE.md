# WhisperDesk — Claude Context

## セットアップ

```bash
uv sync                        # 通常依存
uv sync --extra diarize        # 話者分離を使う場合
uv run whisper-desk            # アプリ起動
```

- FFmpeg が必要 (Whisper の内部依存): `winget install ffmpeg` (Windows) / `brew install ffmpeg` (Mac)

## 環境

- Python 3.13+, パッケージマネージャは `uv`
- ログ・設定の保存先: `~/.whisper-desk/` (`LOG_DIR`)

## 開発コマンド

- テスト: `uv run pytest`
- リンター: `uv run ruff check .`
- フォーマット: `uv run ruff format .`
- 型チェック: `uv run ty check`

## アーキテクチャ

- `app.py` — エントリポイント (Flet app 起動)
- `config.py` — 定数・モデル一覧・HF トークン管理
- `transcriber.py` — Whisper ラッパー (`TranscriptionResult`, `Transcriber`)
- `diarizer.py` — pyannote.audio 話者分離 (`Diarizer`, `DiarizedSegment`)
- `recorder.py` — マイク録音
- `history.py` — SQLite 履歴 DB
- `ui/main_view.py` — Flet UI (`WhisperApp`)

## コードスタイル

- コード内コメントは日本語、コミットメッセージは英語
- ruff line-length=100, target=py313

## 注意事項

- HF トークンは `LOG_DIR/hf_token` に平文保存される (config.py の `load/save_hf_token`)
- 話者分離は pyannote.audio 未インストール時は UI で無効化される (`PYANNOTE_AVAILABLE` フラグ)
- Whisper セグメントと pyannote diarization の統合は `diarizer._assign_speakers()` で行う
- `import config` はロギング初期化の副作用あり (app.py で意図的に import)
