"""
Video editing page for News Short Generator Studio.
Separated from the main window to keep the UI modular.
"""
from __future__ import annotations

from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox


class VideoEditPage:
    """Simple editing UI embedded into the main studio window."""

    def __init__(self, parent: ctk.CTkFrame, *, colors: dict[str, str], on_log):
        self.colors = colors
        self.on_log = on_log
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self._build_header()
        self._build_body()

    def _build_header(self):
        header = ctk.CTkFrame(self.frame, corner_radius=18, fg_color=self.colors["panel"])
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="動画編集",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=12)

    def _build_body(self):
        body = ctk.CTkScrollableFrame(self.frame, corner_radius=18, fg_color=self.colors["panel"])
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)

        row = 0
        self._section_label(body, "素材動画").grid(row=row, column=0, sticky="w", pady=(10, 4)); row += 1
        self.source_entry = ctk.CTkEntry(body, height=34, corner_radius=12)
        self.source_entry.grid(row=row, column=0, sticky="ew"); row += 1
        ctk.CTkButton(
            body,
            text="動画を選択",
            command=self._browse_source,
            height=36,
            corner_radius=12,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
        ).grid(row=row, column=0, sticky="w", pady=(6, 14)); row += 1

        self._section_label(body, "トリミング").grid(row=row, column=0, sticky="w", pady=(0, 4)); row += 1
        trim_row = ctk.CTkFrame(body, fg_color="transparent")
        trim_row.grid(row=row, column=0, sticky="ew", pady=(0, 10)); row += 1
        trim_row.grid_columnconfigure((0, 1), weight=1)
        self.start_entry = ctk.CTkEntry(trim_row, placeholder_text="開始 (秒)", height=34, corner_radius=12)
        self.end_entry = ctk.CTkEntry(trim_row, placeholder_text="終了 (秒)", height=34, corner_radius=12)
        self.start_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.end_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._section_label(body, "書き出し").grid(row=row, column=0, sticky="w", pady=(0, 4)); row += 1
        export_row = ctk.CTkFrame(body, fg_color="transparent")
        export_row.grid(row=row, column=0, sticky="ew", pady=(0, 10)); row += 1
        export_row.grid_columnconfigure(0, weight=1)
        self.output_entry = ctk.CTkEntry(export_row, height=34, corner_radius=12)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            export_row,
            text="保存先",
            command=self._browse_output,
            height=34,
            corner_radius=12,
            fg_color=self.colors["button"],
            hover_color=self.colors["button_hover"],
            width=110,
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        ctk.CTkButton(
            body,
            text="プレビュー",
            command=self._preview,
            height=40,
            corner_radius=12,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
        ).grid(row=row, column=0, sticky="ew", pady=(0, 8)); row += 1

        ctk.CTkButton(
            body,
            text="書き出し",
            command=self._export,
            height=44,
            corner_radius=14,
            fg_color=self.colors["ok"],
            hover_color=self.colors["ok_hover"],
        ).grid(row=row, column=0, sticky="ew", pady=(0, 12)); row += 1

        hint = (
            "・開始/終了の秒数を指定すると該当区間だけを書き出します。\n"
            "・出力先を空にすると元動画と同じフォルダに保存します。\n"
            "・実際の編集ロジックは別途ワークフローに差し込めます。"
        )
        ctk.CTkLabel(
            body,
            text=hint,
            justify="left",
            text_color=self.colors["muted"],
        ).grid(row=row, column=0, sticky="w")

    def _section_label(self, parent, text: str):
        return ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
        )

    def _browse_source(self):
        path = filedialog.askopenfilename(title="素材動画を選択", filetypes=[("動画ファイル", "*.mp4;*.mov;*.mkv;*.avi"), ("すべて", "*.*")])
        if path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, path)
            self.on_log(f"🎞️ 素材を選択: {path}")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="保存先を指定", defaultextension=".mp4", filetypes=[("MP4", "*.mp4"), ("すべて", "*.*")])
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)
            self.on_log(f"💾 出力先を指定: {path}")

    def _preview(self):
        src = self.source_entry.get().strip()
        if not src:
            messagebox.showerror("エラー", "プレビューする動画を選んでください")
            return
        self.on_log(f"▶ プレビュー要求: {src}")
        messagebox.showinfo("プレビュー", "プレビュー機能はダミーです。別途編集処理を組み込んでください。")

    def _export(self):
        src = self.source_entry.get().strip()
        if not src:
            messagebox.showerror("エラー", "書き出す動画を選んでください")
            return
        start = self.start_entry.get().strip()
        end = self.end_entry.get().strip()
        dest = self.output_entry.get().strip()
        if not dest:
            dest = str(Path(src).with_name(Path(src).stem + "_edited.mp4"))
            self.output_entry.insert(0, dest)
        info = f"開始: {start or '未指定'} / 終了: {end or '未指定'} / 保存先: {dest}"
        self.on_log(f"📤 書き出し要求: {info}")
        messagebox.showinfo("書き出し", "書き出し処理はスタブです。実際の編集処理を接続してください。")

