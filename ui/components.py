"""共有 UI コンポーネント"""

import flet as ft

from ui.theme import ACCENT


def section_card(title: str, icon, controls: list, expand=False) -> ft.Card:
    """セクションカードを生成する。"""
    return ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(icon, color=ACCENT, size=20),
                            ft.Text(title, size=16, weight=ft.FontWeight.W_600),
                        ],
                        spacing=8,
                    ),
                    *controls,
                ],
                spacing=10,
            ),
            padding=20,
        ),
        expand=expand,
    )
