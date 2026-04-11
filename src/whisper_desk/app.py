"""WhisperDesk エントリーポイント"""

import flet as ft

from . import config  # noqa: F401 — ログ設定の初期化
from .ui.main_view import WhisperApp


def main(page: ft.Page):
    WhisperApp(page)


if __name__ == "__main__":
    ft.run(main)
