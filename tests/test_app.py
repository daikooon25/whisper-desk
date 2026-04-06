"""ui/components.py の UI ヘルパーのテスト"""

import flet as ft

from ui.components import section_card


class TestSectionCard:
    def test_returns_card(self):
        card = section_card("タイトル", ft.Icons.HOME, [])
        assert isinstance(card, ft.Card)
