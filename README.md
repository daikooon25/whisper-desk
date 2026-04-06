# WhisperDesk

Whisper ローカルモデルによる音声文字起こしデスクトップアプリ。

OpenAI Whisper をローカルで実行し、音声ファイルやマイク録音からテキストへの文字起こしを行います。

## 機能

- **音声ファイルからの文字起こし** — wav, mp3, m4a, flac, ogg, webm, mp4 に対応
- **マイク録音からの文字起こし** — リアルタイム録音してそのまま文字起こし
- **複数モデル対応** — tiny / base / small / medium / large / turbo から選択
- **多言語対応** — 日本語・英語など11言語 + 自動検出
- **履歴管理** — 文字起こし結果を SQLite に自動保存・閲覧・削除
- **結果のコピー・保存** — クリップボードへのコピー、テキストファイルへの保存

## 必要環境

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (推奨パッケージマネージャ)
- FFmpeg (Whisper が内部で使用)

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd whisper-desk

# 依存パッケージをインストール
uv sync
```

### FFmpeg のインストール

Whisper は内部で FFmpeg を使用します。未インストールの場合：

- **Windows**: `winget install ffmpeg` または [公式サイト](https://ffmpeg.org/download.html) からダウンロード
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## 使い方

```bash
uv run whisper-desk
```

1. モデルを選択（初回は自動ダウンロード）
2. 言語を選択（「自動検出」または特定言語を指定すると精度向上）
3. 音声ファイルを選択、またはマイクで録音
4. 「文字起こし実行」をクリック
5. 結果をコピーまたはファイルに保存

## 開発

```bash
# テスト実行
uv run pytest

# リンター
uv run ruff check .

# 型チェック
uv run ty check
```

## ライセンス

MIT
