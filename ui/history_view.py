"""履歴ビュー"""

import logging
from collections.abc import Callable

import flet as ft

from history import HistoryDB
from ui.theme import ACCENT

logger = logging.getLogger(__name__)


class HistoryView:
    """文字起こし履歴の一覧・詳細・削除を管理するビュー。"""

    def __init__(
        self,
        page: ft.Page,
        history_db: HistoryDB,
        on_select: Callable[[dict], None],
    ):
        self.page = page
        self.history_db = history_db
        self.on_select = on_select

        self.history_list = ft.Column(spacing=4)
        self.container = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.HISTORY, color=ACCENT, size=20),
                            ft.Text("文字起こし履歴", size=16, weight=ft.FontWeight.W_600),
                        ],
                        spacing=8,
                    ),
                    self.history_list,
                ],
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
            ),
            padding=ft.Padding.symmetric(horizontal=24, vertical=8),
            expand=True,
            visible=False,
        )

    def refresh(self):
        """履歴一覧を再読み込みする。"""
        rows = self.history_db.get_all(limit=50)
        self.history_list.controls.clear()

        if not rows:
            self.history_list.controls.append(
                ft.Container(
                    content=ft.Text(
                        "履歴はまだありません",
                        size=14,
                        color=ft.Colors.GREY_500,
                    ),
                    padding=20,
                )
            )
            return

        for row in rows:
            self.history_list.controls.append(self._build_card(row))

    def _build_card(self, row: dict) -> ft.Card:
        created = row["created_at"][:16].replace("T", " ")
        source = row["source_name"] or "(マイク録音)"
        lang = row["detected_language"] or row["language"]
        duration = f"{row['duration_sec']:.1f}秒" if row["duration_sec"] else ""
        preview = row["text"][:120] + ("..." if len(row["text"]) > 120 else "")

        return ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(created, size=12, weight=ft.FontWeight.BOLD),
                                ft.Container(
                                    content=ft.Text(
                                        row["model"], size=11, color=ft.Colors.WHITE
                                    ),
                                    bgcolor=ACCENT,
                                    border_radius=4,
                                    padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                                ),
                                ft.Text(lang, size=12, color=ft.Colors.GREY_600),
                                ft.Text(duration, size=12, color=ft.Colors.GREY_600),
                            ],
                            spacing=8,
                        ),
                        ft.Text(source, size=12, color=ft.Colors.GREY_600),
                        ft.Text(preview, size=13),
                        ft.Row(
                            [
                                ft.Button(
                                    "結果を表示",
                                    icon=ft.Icons.OPEN_IN_NEW,
                                    on_click=lambda _, rid=row["id"]: self._on_show(rid),
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_OUTLINE,
                                    tooltip="削除",
                                    icon_color=ft.Colors.RED_400,
                                    on_click=lambda _, rid=row["id"]: self._on_delete(rid),
                                ),
                            ],
                            spacing=4,
                        ),
                    ],
                    spacing=4,
                ),
                padding=12,
            ),
        )

    def _on_show(self, row_id: int):
        row = self.history_db.get_by_id(row_id)
        if not row:
            return
        logger.info("履歴から読込: id=%d", row_id)
        self.on_select(row)

    def _on_delete(self, row_id: int):
        def on_confirm(e):
            self.page.pop_dialog()
            if e.control.data == "confirm":
                self.history_db.delete(row_id)
                self.refresh()
                self.page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("履歴の削除"),
            content=ft.Text("この履歴を削除しますか？"),
            actions=[
                ft.TextButton("キャンセル", on_click=on_confirm),
                ft.TextButton("削除", data="confirm", on_click=on_confirm),
            ],
        )
        self.page.show_dialog(dialog)
