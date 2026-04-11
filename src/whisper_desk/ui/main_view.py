"""メインアプリケーション UI"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import flet as ft

from ..config import LANGUAGES, MODELS, is_model_downloaded, load_hf_token, save_hf_token
from ..diarizer import PYANNOTE_AVAILABLE, Diarizer, format_diarized_text
from ..history import HistoryDB
from ..recorder import Recorder, RecorderError
from ..transcriber import Transcriber
from .components import section_card
from .history_view import HistoryView
from .theme import ACCENT, RECORD_COLOR

logger = logging.getLogger(__name__)


class WhisperApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "WhisperDesk"
        self.page.width = 860
        self.page.height = 780
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT

        self.model_name = "base"
        self.audio_path: str | None = None
        self._is_recording_source = False

        self.recorder = Recorder()
        self.transcriber = Transcriber()
        self.diarizer = Diarizer()
        self.history_db = HistoryDB()

        self._build_ui()
        self.page.add(self.container)
        logger.info("アプリ起動")

    # ── UI構築 ──────────────────────────────────────────

    def _build_model_row(self, name: str, params: str, size: str) -> ft.Row:
        downloaded = is_model_downloaded(name)
        badge = ft.Container(
            content=ft.Text("DL済", size=11, color=ft.Colors.WHITE),
            bgcolor=ft.Colors.GREEN_400,
            border_radius=4,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
            visible=downloaded,
        )
        need_dl = ft.Container(
            content=ft.Text("未DL", size=11, color=ft.Colors.WHITE),
            bgcolor=ft.Colors.GREY_400,
            border_radius=4,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
            visible=not downloaded,
        )
        return ft.Row(
            [
                ft.Radio(value=name),
                ft.Text(name, weight=ft.FontWeight.BOLD, width=70),
                ft.Text(f"{params} params", width=110, size=13),
                ft.Text(size, width=80, size=13, color=ft.Colors.GREY_700),
                badge,
                need_dl,
            ],
            spacing=4,
        )

    def _build_ui(self):
        # ── ヘッダー ──
        self.theme_toggle = ft.IconButton(
            icon=ft.Icons.DARK_MODE,
            tooltip="ダークモード切替",
            on_click=self._toggle_theme,
        )
        self.history_toggle = ft.IconButton(
            icon=ft.Icons.HISTORY,
            tooltip="履歴を表示",
            on_click=self._toggle_history,
        )

        header = ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.TRANSCRIBE, size=32, color=ACCENT),
                            ft.Text(
                                "WhisperDesk",
                                size=26,
                                weight=ft.FontWeight.BOLD,
                            ),
                        ],
                        spacing=12,
                    ),
                    ft.Row(
                        [self.history_toggle, self.theme_toggle],
                        spacing=0,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding(left=30, top=20, right=30, bottom=10),
        )

        # ── モデル選択 ──
        self.model_rows = {
            name: self._build_model_row(name, params, size)
            for name, (params, size) in MODELS.items()
        }
        self.model_radio = ft.RadioGroup(
            value="base",
            on_change=self._on_model_change,
            content=ft.Column(
                list(self.model_rows.values()),
                spacing=2,
            ),
        )
        model_card = section_card(
            "モデル選択", ft.Icons.MODEL_TRAINING, [self.model_radio]
        )

        # ── 言語選択 ──
        self.language_dropdown = ft.Dropdown(
            label="言語",
            value="auto",
            options=[
                ft.dropdown.Option(
                    key=code,
                    text=f"{label} ({code})" if code != "auto" else label,
                )
                for code, label in LANGUAGES.items()
            ],
            width=220,
        )
        language_card = section_card(
            "言語設定",
            ft.Icons.LANGUAGE,
            [
                self.language_dropdown,
                ft.Text(
                    "言語を指定すると認識精度が向上します",
                    size=12,
                    color=ft.Colors.GREY_600,
                ),
            ],
        )

        # ── 話者分離設定 ──
        self.diarize_checkbox = ft.Checkbox(
            label="話者分離 (Speaker Diarization)",
            value=False,
            on_change=self._on_diarize_toggle,
            disabled=not PYANNOTE_AVAILABLE,
        )
        saved_token = load_hf_token() or ""
        self.hf_token_field = ft.TextField(
            label="HuggingFace Token",
            value=saved_token,
            password=True,
            can_reveal_password=True,
            hint_text="hf_...",
            width=400,
            visible=False,
        )
        diarize_help = ft.Text(
            "pyannote.audio 未インストール: uv sync --extra diarize",
            size=12,
            color=ft.Colors.ORANGE_700,
            visible=not PYANNOTE_AVAILABLE,
        )
        diarize_card = section_card(
            "話者分離",
            ft.Icons.PEOPLE,
            [self.diarize_checkbox, self.hf_token_field, diarize_help],
        )

        # ── サービス登録 ──
        self.clipboard = ft.Clipboard()
        self.file_picker = ft.FilePicker()
        self.save_picker = ft.FilePicker()
        self.page.services.extend(
            [self.clipboard, self.file_picker, self.save_picker]
        )

        # ── ファイル選択 ──
        self.file_button = ft.Button(
            "選択",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._on_file_click,
        )
        self.file_path_field = ft.TextField(
            label="ファイルパスを貼り付け / 入力してEnter",
            hint_text="C:\\...\\audio.mp3" if sys.platform == "win32" else "/path/to/audio.mp3",
            on_submit=self._on_file_path_submit,
            on_change=self._on_file_path_change,
            expand=True,
        )
        self.selected_file_text = ft.Text("", size=13)

        file_card = section_card(
            "音声ファイルから文字起こし",
            ft.Icons.AUDIO_FILE,
            [
                ft.Row(
                    [self.file_button, self.file_path_field],
                    vertical_alignment=ft.CrossAxisAlignment.END,
                ),
                self.selected_file_text,
            ],
        )

        # ── 録音 ──
        self.record_button = ft.Button(
            "録音開始",
            icon=ft.Icons.MIC,
            on_click=self._toggle_recording,
            color=ft.Colors.WHITE,
            bgcolor=RECORD_COLOR,
        )
        self.record_status = ft.Text("", size=13, color=RECORD_COLOR)
        self.record_timer = ft.Text("", size=13, color=ft.Colors.GREY_600)

        record_card = section_card(
            "マイクで録音して文字起こし",
            ft.Icons.MIC_EXTERNAL_ON,
            [ft.Row([self.record_button, self.record_status, self.record_timer])],
        )

        # ── 実行 ──
        self.transcribe_button = ft.Button(
            "文字起こし実行",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_transcribe,
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.GREY_400,
            disabled=True,
            height=48,
            width=220,
            style=ft.ButtonStyle(text_style=ft.TextStyle(size=16)),
        )
        self.progress = ft.ProgressBar(visible=False)
        self.status_text = ft.Text("", size=13)

        # ── 結果 ──
        self.result_field = ft.TextField(
            label="文字起こし結果",
            multiline=True,
            min_lines=6,
            max_lines=20,
            read_only=True,
            expand=True,
        )
        self.copy_button = ft.Button(
            "コピー",
            icon=ft.Icons.COPY,
            on_click=self._on_copy,
            disabled=True,
        )
        self.save_button = ft.Button(
            "保存",
            icon=ft.Icons.SAVE_ALT,
            on_click=self._on_save,
            disabled=True,
        )

        result_card = section_card(
            "結果",
            ft.Icons.TEXT_SNIPPET,
            [
                self.result_field,
                ft.Row([self.copy_button, self.save_button], spacing=8),
            ],
            expand=True,
        )

        # ── メインビュー ──
        self.main_body = ft.Container(
            content=ft.Column(
                [
                    model_card,
                    language_card,
                    diarize_card,
                    file_card,
                    record_card,
                    ft.Container(
                        content=ft.Column(
                            [self.transcribe_button, self.progress, self.status_text],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=8,
                        ),
                        alignment=ft.Alignment(0, 0),
                        padding=ft.Padding.symmetric(vertical=4),
                    ),
                    result_card,
                ],
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=12,
            ),
            padding=ft.Padding.symmetric(horizontal=24, vertical=8),
            expand=True,
            visible=True,
        )

        # ── 履歴ビュー ──
        self.history_view = HistoryView(
            page=self.page,
            history_db=self.history_db,
            on_select=self._on_history_select,
        )

        self.container = ft.Column(
            [header, self.main_body, self.history_view.container],
            expand=True,
            spacing=0,
        )

    # ── テーマ切替 ──────────────────────────────────────

    def _toggle_theme(self, _):
        if self.page.theme_mode == ft.ThemeMode.LIGHT:
            self.page.theme_mode = ft.ThemeMode.DARK
            self.theme_toggle.icon = ft.Icons.LIGHT_MODE
        else:
            self.page.theme_mode = ft.ThemeMode.LIGHT
            self.theme_toggle.icon = ft.Icons.DARK_MODE
        self.page.update()

    # ── 履歴 ───────────────────────────────────────────

    def _toggle_history(self, _):
        show_history = not self.history_view.container.visible
        self.history_view.container.visible = show_history
        self.main_body.visible = not show_history
        if show_history:
            self.history_view.refresh()
            self.history_toggle.icon = ft.Icons.HOME
            self.history_toggle.tooltip = "メインに戻る"
        else:
            self.history_toggle.icon = ft.Icons.HISTORY
            self.history_toggle.tooltip = "履歴を表示"
        self.page.update()

    def _on_history_select(self, row: dict):
        """履歴から選択された結果をメインビューに表示する。"""
        self.result_field.value = row["text"]
        self.copy_button.disabled = False
        self.save_button.disabled = False
        self.history_view.container.visible = False
        self.main_body.visible = True
        self.history_toggle.icon = ft.Icons.HISTORY
        self.history_toggle.tooltip = "履歴を表示"
        self.status_text.value = (
            f"履歴から読込 ({row['created_at'][:16].replace('T', ' ')})"
        )
        self.page.update()

    # ── ボタン状態管理 ──────────────────────────────────

    def _set_transcribe_enabled(self, enabled: bool):
        self.transcribe_button.disabled = not enabled
        self.transcribe_button.bgcolor = ACCENT if enabled else ft.Colors.GREY_400

    # ── 話者分離 ────────────────────────────────────────

    def _on_diarize_toggle(self, e):
        self.hf_token_field.visible = self.diarize_checkbox.value
        self.page.update()

    # ── モデル ──────────────────────────────────────────

    def _refresh_download_status(self):
        for name, row in self.model_rows.items():
            downloaded = is_model_downloaded(name)
            row.controls[4].visible = downloaded
            row.controls[5].visible = not downloaded

    def _on_model_change(self, e):
        self.model_name = self.model_radio.value
        logger.info("モデル変更: %s", self.model_name)

    # ── ファイル選択 ────────────────────────────────────

    def _set_audio_file(self, path_str: str):
        path_str = path_str.strip().strip('"').strip("'")
        p = Path(path_str)
        if p.is_file():
            self.audio_path = str(p)
            self._is_recording_source = False
            self.file_path_field.value = str(p)
            self.selected_file_text.value = f"{p.name} (OK)"
            self.selected_file_text.color = ft.Colors.GREEN_700
            self._set_transcribe_enabled(True)
            logger.info("ファイル選択: %s", p.name)
        else:
            self.selected_file_text.value = "ファイルが見つかりません"
            self.selected_file_text.color = ft.Colors.RED_400
            self._set_transcribe_enabled(False)
            logger.warning("ファイル未発見: %s", path_str)
        self.page.update()

    async def _on_file_click(self, _):
        files = await self.file_picker.pick_files(
            allowed_extensions=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"],
            dialog_title="音声ファイルを選択",
        )
        if files:
            self._set_audio_file(files[0].path)

    def _on_file_path_submit(self, e):
        if e.control.value:
            self._set_audio_file(e.control.value)

    def _on_file_path_change(self, e):
        val = (e.control.value or "").strip().strip('"').strip("'")
        if not val:
            self.audio_path = None
            self.selected_file_text.value = ""
            self._set_transcribe_enabled(False)
            self.page.update()
        elif Path(val).is_file():
            self._set_audio_file(val)
        else:
            self.audio_path = None
            self.selected_file_text.value = ""
            self._set_transcribe_enabled(False)
            self.page.update()

    # ── 録音 ───────────────────────────────────────────

    def _toggle_recording(self, _):
        if not self.recorder.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.record_button.content = "録音停止"
        self.record_button.icon = ft.Icons.STOP
        self.record_button.bgcolor = ft.Colors.GREY_700
        self.record_status.value = "録音中..."
        self.record_timer.value = "00:00"
        self.page.update()

        try:
            self.recorder.start()
        except RecorderError:
            self.record_button.content = "録音開始"
            self.record_button.icon = ft.Icons.MIC
            self.record_button.bgcolor = RECORD_COLOR
            self.record_status.value = "マイクが利用できません"
            self.record_status.color = ft.Colors.RED_400
            self.record_timer.value = ""
            self.page.update()
            return

        async def update_timer():
            while self.recorder.recording:
                elapsed = self.recorder.elapsed_seconds
                mins, secs = divmod(int(elapsed), 60)
                self.record_timer.value = f"{mins:02d}:{secs:02d}"
                self.page.update()
                await asyncio.sleep(0.5)

        self.page.run_task(update_timer)

    def _stop_recording(self):
        audio_path = self.recorder.stop()

        self.record_button.content = "録音開始"
        self.record_button.icon = ft.Icons.MIC
        self.record_button.bgcolor = RECORD_COLOR

        if audio_path:
            self.audio_path = audio_path
            self._is_recording_source = True
            duration = self.recorder.duration or 0
            mins, secs = divmod(int(duration), 60)
            self.record_status.value = f"録音完了 ({mins:02d}:{secs:02d})"
            self.record_timer.value = ""
            self._set_transcribe_enabled(True)
        else:
            self.record_status.value = "録音データなし"
            self.record_timer.value = ""

        self.page.update()

    # ── 文字起こし ─────────────────────────────────────

    def _on_transcribe(self, _):
        if not self.audio_path:
            return

        self._set_transcribe_enabled(False)
        self.progress.visible = True
        self.status_text.value = ""
        self.result_field.value = ""
        self.copy_button.disabled = True
        self.save_button.disabled = True
        self.page.update()

        self.page.run_task(self._run_transcription_async)

    async def _run_transcription_async(self):
        try:
            self.status_text.value = f"モデル「{self.model_name}」を読み込み中..."
            self.page.update()
            await self.transcriber.ensure_model(self.model_name)

            lang = self.language_dropdown.value
            lang_label = LANGUAGES.get(lang, lang)
            if lang == "auto":
                self.status_text.value = "文字起こし中 (言語自動検出)..."
            else:
                self.status_text.value = f"文字起こし中 ({lang_label})..."
            self.page.update()

            audio_path = self.audio_path
            enable_diarize = self.diarize_checkbox.value
            result = await self.transcriber.transcribe(
                audio_path, lang, diarize=enable_diarize
            )

            # 話者分離
            if enable_diarize and result.segments:
                hf_token = self.hf_token_field.value or load_hf_token()
                if not hf_token:
                    self.status_text.value = "エラー: HuggingFace トークンを入力してください"
                    self.progress.visible = False
                    self._set_transcribe_enabled(True)
                    self.page.update()
                    return
                save_hf_token(hf_token)

                self.status_text.value = "話者分離モデルを読み込み中..."
                self.page.update()
                await self.diarizer.ensure_pipeline(hf_token)

                self.status_text.value = "話者分離を実行中..."
                self.page.update()
                diarized = await self.diarizer.diarize(audio_path, result.segments)
                display_text = format_diarized_text(diarized)
            else:
                display_text = result.text

            self.result_field.value = display_text
            self.status_text.value = (
                f"完了 (言語: {result.detected_language}, {result.elapsed_sec:.1f}秒)"
            )
            self.copy_button.disabled = False
            self.save_button.disabled = False
            self.progress.visible = False
            self._set_transcribe_enabled(True)
            self._refresh_download_status()
            self.page.update()

            # 履歴に保存
            source_path = Path(audio_path)
            self.history_db.add(
                source_type="recording" if self._is_recording_source else "file",
                source_name=None if self._is_recording_source else source_path.name,
                model=self.model_name,
                language=lang,
                detected_language=result.detected_language,
                duration_sec=result.duration_sec,
                text=display_text,
            )

            # 録音の一時ファイルを削除
            if self._is_recording_source:
                try:
                    source_path.unlink()
                    logger.info("一時ファイル削除: %s", source_path)
                except OSError as e:
                    logger.warning("一時ファイル削除失敗: %s", e)

        except Exception as e:
            self.status_text.value = f"エラー: {e}"
            logger.exception("文字起こしエラー")
            self.progress.visible = False
            self._set_transcribe_enabled(True)
            self.page.update()

    # ── コピー・保存 ───────────────────────────────────

    async def _on_copy(self, _):
        if self.result_field.value:
            await self.clipboard.set(self.result_field.value)
            self.status_text.value = "クリップボードにコピーしました"
            self.page.update()

    def _default_save_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.audio_path and not self._is_recording_source:
            stem = Path(self.audio_path).stem
            return f"{stem}_{timestamp}.txt"
        return f"transcription_{timestamp}.txt"

    async def _on_save(self, _):
        path = await self.save_picker.save_file(
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt"],
            file_name=self._default_save_name(),
        )
        if path and self.result_field.value:
            if not path.endswith(".txt"):
                path += ".txt"
            Path(path).write_text(self.result_field.value, encoding="utf-8")
            self.status_text.value = f"保存しました: {path}"
            logger.info("ファイル保存: %s", path)
            self.page.update()
