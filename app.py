import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import flet as ft
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from history import HistoryDB

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
logger = logging.getLogger(__name__)

# ── 定数 ─────────────────────────────────────────────

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

SAMPLE_RATE = 16000

ACCENT = ft.Colors.BLUE_600
ACCENT_LIGHT = ft.Colors.BLUE_50
RECORD_COLOR = ft.Colors.RED_400


def _section_card(title: str, icon, controls: list, expand=False) -> ft.Card:
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


# ── メインアプリ ─────────────────────────────────────

class WhisperApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "WhisperDesk"
        self.page.width = 860
        self.page.height = 780
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT

        self.model = None
        self.model_name = "base"
        self.recording = False
        self.audio_data: list[np.ndarray] = []
        self.audio_path: str | None = None

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
        model_card = _section_card(
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
        language_card = _section_card(
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
            hint_text="C:\\...\\audio.mp3",
            on_submit=self._on_file_path_submit,
            on_change=self._on_file_path_change,
            expand=True,
        )
        self.selected_file_text = ft.Text("", size=13)

        file_card = _section_card(
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

        record_card = _section_card(
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

        result_card = _section_card(
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
        self.history_list = ft.Column(spacing=4)
        self.history_body = ft.Container(
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

        self.container = ft.Column(
            [header, self.main_body, self.history_body],
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

    # ── 履歴ビュー ─────────────────────────────────────

    def _toggle_history(self, _):
        show_history = not self.history_body.visible
        self.history_body.visible = show_history
        self.main_body.visible = not show_history
        if show_history:
            self._refresh_history_list()
            self.history_toggle.icon = ft.Icons.HOME
            self.history_toggle.tooltip = "メインに戻る"
        else:
            self.history_toggle.icon = ft.Icons.HISTORY
            self.history_toggle.tooltip = "履歴を表示"
        self.page.update()

    def _refresh_history_list(self):
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
            created = row["created_at"][:16].replace("T", " ")
            source = row["source_name"] or "(マイク録音)"
            lang = row["detected_language"] or row["language"]
            duration = (
                f"{row['duration_sec']:.1f}秒" if row["duration_sec"] else ""
            )
            text_preview = row["text"][:120] + ("..." if len(row["text"]) > 120 else "")

            card = ft.Card(
                content=ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        created,
                                        size=12,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Container(
                                        content=ft.Text(
                                            row["model"],
                                            size=11,
                                            color=ft.Colors.WHITE,
                                        ),
                                        bgcolor=ACCENT,
                                        border_radius=4,
                                        padding=ft.Padding.symmetric(
                                            horizontal=6, vertical=2
                                        ),
                                    ),
                                    ft.Text(
                                        lang, size=12, color=ft.Colors.GREY_600
                                    ),
                                    ft.Text(
                                        duration, size=12, color=ft.Colors.GREY_600
                                    ),
                                ],
                                spacing=8,
                            ),
                            ft.Text(
                                source, size=12, color=ft.Colors.GREY_600
                            ),
                            ft.Text(text_preview, size=13),
                            ft.Row(
                                [
                                    ft.Button(
                                        "結果を表示",
                                        icon=ft.Icons.OPEN_IN_NEW,
                                        on_click=lambda _, rid=row["id"]: self._show_history_detail(rid),
                                    ),
                                    ft.IconButton(
                                        icon=ft.Icons.DELETE_OUTLINE,
                                        tooltip="削除",
                                        icon_color=ft.Colors.RED_400,
                                        on_click=lambda _, rid=row["id"]: self._delete_history(rid),
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
            self.history_list.controls.append(card)

    def _show_history_detail(self, row_id: int):
        row = self.history_db.get_by_id(row_id)
        if not row:
            return
        self.result_field.value = row["text"]
        self.copy_button.disabled = False
        self.save_button.disabled = False
        # メインビューに戻る
        self.history_body.visible = False
        self.main_body.visible = True
        self.history_toggle.icon = ft.Icons.HISTORY
        self.history_toggle.tooltip = "履歴を表示"
        self.status_text.value = (
            f"履歴から読込 ({row['created_at'][:16].replace('T', ' ')})"
        )
        logger.info("履歴から読込: id=%d", row_id)
        self.page.update()

    def _delete_history(self, row_id: int):
        def on_confirm(e):
            self.page.pop_dialog()
            if e.control.data == "confirm":
                self.history_db.delete(row_id)
                self._refresh_history_list()
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

    # ── ボタン状態管理 ──────────────────────────────────

    def _set_transcribe_enabled(self, enabled: bool):
        self.transcribe_button.disabled = not enabled
        self.transcribe_button.bgcolor = ACCENT if enabled else ft.Colors.GREY_400

    # ── モデル ──────────────────────────────────────────

    def _refresh_download_status(self):
        for name, row in self.model_rows.items():
            downloaded = is_model_downloaded(name)
            row.controls[4].visible = downloaded
            row.controls[5].visible = not downloaded

    def _on_model_change(self, e):
        self.model_name = self.model_radio.value
        self.model = None
        logger.info("モデル変更: %s", self.model_name)

    # ── ファイル選択 ────────────────────────────────────

    def _set_audio_file(self, path_str: str):
        path_str = path_str.strip().strip('"').strip("'")
        p = Path(path_str)
        if p.is_file():
            self.audio_path = str(p)
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
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.recording = True
        self.audio_data = []
        self._record_start_time = time.time()
        self.record_button.content = "録音停止"
        self.record_button.icon = ft.Icons.STOP
        self.record_button.bgcolor = ft.Colors.GREY_700
        self.record_status.value = "録音中..."
        self.record_timer.value = "00:00"
        self.page.update()
        logger.info("録音開始")

        def callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

        async def update_timer():
            while self.recording:
                elapsed = time.time() - self._record_start_time
                mins, secs = divmod(int(elapsed), 60)
                self.record_timer.value = f"{mins:02d}:{secs:02d}"
                self.page.update()
                await asyncio.sleep(0.5)

        self.page.run_task(update_timer)

    def _stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()

        self.record_button.content = "録音開始"
        self.record_button.icon = ft.Icons.MIC
        self.record_button.bgcolor = RECORD_COLOR

        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            duration = len(audio) / SAMPLE_RATE
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio, SAMPLE_RATE)
            self.audio_path = tmp.name
            mins, secs = divmod(int(duration), 60)
            self.record_status.value = f"録音完了 ({mins:02d}:{secs:02d})"
            self.record_timer.value = ""
            self._set_transcribe_enabled(True)
            logger.info("録音完了: %.1f秒", duration)
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
        t0 = time.time()
        try:
            if self.model is None or self.model_name != getattr(
                self, "_loaded_model_name", None
            ):
                self.status_text.value = f"モデル「{self.model_name}」を読み込み中..."
                self.page.update()
                logger.info("モデル読込開始: %s", self.model_name)
                model_name = self.model_name
                self.model = await asyncio.to_thread(whisper.load_model, model_name)
                self._loaded_model_name = model_name
                logger.info("モデル読込完了: %s (%.1f秒)", model_name, time.time() - t0)

            lang = self.language_dropdown.value
            lang_label = LANGUAGES.get(lang, lang)
            if lang == "auto":
                self.status_text.value = "文字起こし中 (言語自動検出)..."
            else:
                self.status_text.value = f"文字起こし中 ({lang_label})..."
            self.page.update()

            audio_path = self.audio_path
            model_name = self.model_name
            logger.info(
                "文字起こし開始: file=%s, model=%s, lang=%s",
                audio_path,
                model_name,
                lang,
            )

            transcribe_opts = {}
            if lang != "auto":
                transcribe_opts["language"] = lang

            result = await asyncio.to_thread(
                self.model.transcribe, audio_path, **transcribe_opts
            )

            elapsed = time.time() - t0
            text = result["text"].strip()
            detected_lang = result.get("language", "不明")

            # 音声ファイルの実際の長さを取得
            try:
                info = sf.info(audio_path)
                audio_duration = info.duration
            except Exception:
                audio_duration = None

            self.result_field.value = text
            self.status_text.value = f"完了 (言語: {detected_lang}, {elapsed:.1f}秒)"
            self.copy_button.disabled = False
            self.save_button.disabled = False
            self.progress.visible = False
            self._set_transcribe_enabled(True)
            self._refresh_download_status()
            self.page.update()

            logger.info(
                "文字起こし完了: lang=%s, %.1f秒, %d文字",
                detected_lang,
                elapsed,
                len(text),
            )

            # 履歴に保存 (UI更新後に実行)
            source_path = Path(audio_path)
            is_recording = str(source_path.parent) == tempfile.gettempdir()
            self.history_db.add(
                source_type="recording" if is_recording else "file",
                source_name=None if is_recording else source_path.name,
                model=model_name,
                language=lang,
                detected_language=detected_lang,
                duration_sec=round(audio_duration, 1) if audio_duration else None,
                text=text,
            )

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
        if self.audio_path:
            source = Path(self.audio_path)
            if not source.name.startswith("tmp"):
                stem = source.stem
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


def main(page: ft.Page):
    WhisperApp(page)


if __name__ == "__main__":
    ft.run(main)
