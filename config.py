"""アプリケーション全体の定数・設定"""

import logging
import os
from pathlib import Path

# ── ログ設定 ──────────────────────────────────────────

LOG_DIR = Path.home() / ".whisper-desk"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ── Whisper モデル ────────────────────────────────────

MODELS = {
    "tiny": ("39M", "~75 MB"),
    "base": ("74M", "~142 MB"),
    "small": ("244M", "~466 MB"),
    "medium": ("769M", "~1.5 GB"),
    "large": ("1550M", "~2.9 GB"),
    "turbo": ("809M", "~1.5 GB"),
}

WHISPER_CACHE_DIR = Path(
    os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
) / "whisper"


def is_model_downloaded(name: str) -> bool:
    return (WHISPER_CACHE_DIR / f"{name}.pt").exists()


# ── 言語 ──────────────────────────────────────────────

LANGUAGES = {
    "auto": "自動検出",
    "ja": "日本語",
    "en": "英語",
    "zh": "中国語",
    "ko": "韓国語",
    "fr": "フランス語",
    "de": "ドイツ語",
    "es": "スペイン語",
    "pt": "ポルトガル語",
    "it": "イタリア語",
    "ru": "ロシア語",
}

# ── 音声 ──────────────────────────────────────────────

SAMPLE_RATE = 16000
