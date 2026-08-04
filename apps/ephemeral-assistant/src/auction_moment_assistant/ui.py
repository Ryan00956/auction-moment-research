from __future__ import annotations

from concurrent.futures import Future
import tkinter as tk
from tkinter import messagebox, ttk

from .map_editor import MapEditor
from .models import CatalogItem
from .observations import ObservationStore
from .presentation import format_prediction
from .predictor import PredictionResult
from .runtime import (
    CapturePipeline,
    EphemeralStateMonitor,
    InferenceCoordinator,
)
from .semantics import event_semantics


class AssistantWindow:
    def __init__(
        self,
        *,
        store: ObservationStore,
        pipeline: CapturePipeline,
        inference: InferenceCoordinator,
        catalog: tuple[CatalogItem, ...],
    ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.inference = inference
        self.catalog = catalog
        self.monitor = EphemeralStateMonitor(pipeline, store)
        self._prediction_future: Future | None = None

        self.root = tk.Tk()
        self.root.title("Auction Moment · 半自动 OCR 预测助手")
        self.root.geometry("1320x860")
        self.root.minsize(1100, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.refresh()
        self.monitor.start()
        self.root.after(100, self._poll_monitor)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="正在连接并监视窗口…")
        ttk.Label(
            top,
            textvariable=self.status_var,
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#0f172a",
        ).pack(side=tk.LEFT)
        self.round_var = tk.IntVar(value=1)
        ttk.Label(top, text="轮次").pack(side=tk.LEFT, padx=(22, 4))
        ttk.Spinbox(top, from_=1, to=5, width=4, textvariable=self.round_var).pack(
            side=tk.LEFT
        )
        ttk.Button(top, text="修正轮次", command=self.apply_round).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="立即重扫本轮", command=self.request_scan).pack(
            side=tk.LEFT, padx=(12, 4)
        )
        self.pause_button = ttk.Button(
            top, text="暂停监视", command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT)
        ttk.Button(top, text="清空本局内存", command=self.reset).pack(side=tk.RIGHT)

        ttk.Label(
            self.root,
            text=(
                "只读取画面；自动输入仅限左侧地图滚动。无抓包、无点击、无按键、"
                "无自动出价、无截图/事件/预测/结算落盘。"
            ),
            padding=(8, 0),
            foreground="#64748b",
        ).pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.LabelFrame(body, text="地图建模与快速纠错", padding=8)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)
        self.map_editor = MapEditor(
            left, store=self.store, catalog=self.catalog, on_change=self._changed
        )
        self.map_editor.pack(fill=tk.BOTH, expand=True)

        event_box = ttk.LabelFrame(right, text="事件列表与人工修正", padding=8)
        event_box.pack(fill=tk.BOTH, expand=True)
        columns = ("round", "kind", "text", "confidence", "source")
        self.event_tree = ttk.Treeview(
            event_box, columns=columns, show="headings", height=9
        )
        for name, label, width in (
            ("round", "轮", 36),
            ("kind", "类型", 55),
            ("text", "识别文本", 240),
            ("confidence", "置信", 52),
            ("source", "来源", 48),
        ):
            self.event_tree.heading(name, text=label)
            self.event_tree.column(name, width=width, anchor=tk.W)
        self.event_tree.pack(fill=tk.BOTH, expand=True)
        self.event_tree.bind("<<TreeviewSelect>>", self._select_event)

        editor = ttk.Frame(event_box)
        editor.pack(fill=tk.X, pady=(7, 0))
        self.event_label_var = tk.StringVar(value="选择上方事件")
        ttk.Label(editor, textvariable=self.event_label_var, width=14).pack(
            side=tk.LEFT
        )
        self.event_text_var = tk.StringVar()
        ttk.Entry(editor, textvariable=self.event_text_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        ttk.Button(editor, text="应用修正", command=self.apply_event).pack(
            side=tk.RIGHT
        )

        evidence_box = ttk.LabelFrame(right, text="本轮证据", padding=8)
        evidence_box.pack(fill=tk.X, pady=(8, 0))
        self.bankroll_var = tk.StringVar()
        self.map_rows_var = tk.StringVar()
        self.map_exact_var = tk.BooleanVar(value=False)
        self.proof_var = tk.StringVar(value="等待自动扫描")
        ttk.Label(evidence_box, text="持有金额").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(evidence_box, textvariable=self.bankroll_var, width=16).grid(
            row=0, column=1, sticky=tk.EW, padx=(4, 10)
        )
        ttk.Button(evidence_box, text="修正金额", command=self.apply_bankroll).grid(
            row=0, column=2
        )
        ttk.Label(evidence_box, text="地图行数").grid(
            row=1, column=0, sticky=tk.W, pady=(6, 0)
        )
        ttk.Entry(evidence_box, textvariable=self.map_rows_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(4, 10), pady=(6, 0)
        )
        ttk.Checkbutton(
            evidence_box,
            text="已人工确认到底部",
            variable=self.map_exact_var,
        ).grid(row=1, column=1, sticky=tk.E, pady=(6, 0))
        ttk.Button(evidence_box, text="应用行数", command=self.apply_map_extent).grid(
            row=1, column=2, pady=(6, 0)
        )
        ttk.Label(
            evidence_box,
            textvariable=self.proof_var,
            foreground="#475569",
            wraplength=420,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(7, 0))
        evidence_box.columnconfigure(1, weight=1)

        prediction_box = ttk.LabelFrame(
            right,
            text="v6 + world-model-v2 估值与建议出价（仅人工参考）",
            padding=8,
        )
        prediction_box.pack(fill=tk.X, pady=(8, 0))
        self.prediction_var = tk.StringVar(value="等待每轮自动扫描")
        ttk.Label(
            prediction_box,
            textvariable=self.prediction_var,
            justify=tk.LEFT,
            wraplength=470,
        ).pack(fill=tk.X)

    def _changed(self) -> None:
        self.refresh()
        self.request_prediction()

    def _poll_monitor(self) -> None:
        for update in self.monitor.drain():
            self.round_var.set(update.round_number)
            self.status_var.set(update.message)
            if update.kind in {
                "session_started",
                "session_ended",
                "round_changed",
                "scan_finished",
            }:
                self.refresh()
            if update.kind == "scan_finished":
                if update.scan and update.scan.errors:
                    self.status_var.set(
                        update.message + "；" + "；".join(update.scan.errors)
                    )
                self.request_prediction()
        self.root.after(100, self._poll_monitor)

    def toggle_pause(self) -> None:
        paused = not self.monitor.paused
        self.monitor.set_paused(paused)
        self.pause_button.configure(text="继续监视" if paused else "暂停监视")
        self.status_var.set("监视已暂停" if paused else "监视已恢复")

    def request_scan(self) -> None:
        self.monitor.request_scan(self.round_var.get())
        self.status_var.set("已请求内存重扫；仅会滚动左侧地图")

    def apply_round(self) -> None:
        self.monitor.set_round(self.round_var.get())
        self.status_var.set(f"已人工修正为第 {self.round_var.get()} 轮")
        self.refresh()

    def _select_event(self, _event=None) -> None:
        selection = self.event_tree.selection()
        if not selection:
            return
        iid = selection[0]
        round_text, kind = iid.split(":", 1)
        values = self.event_tree.item(iid, "values")
        self.event_label_var.set(
            f"R{round_text} {'公共' if kind == 'public' else '个人'}"
        )
        self.event_text_var.set(str(values[2] or ""))

    def apply_event(self) -> None:
        selection = self.event_tree.selection()
        if not selection:
            messagebox.showinfo("尚未选择", "先在事件列表中选择一条")
            return
        round_text, kind = selection[0].split(":", 1)
        text = self.event_text_var.get().strip()
        self.store.correct_event(
            int(round_text), kind, text, event_semantics(text)
        )
        self._changed()

    def apply_bankroll(self) -> None:
        raw = self.bankroll_var.get().replace(",", "").strip()
        try:
            value = int(raw) if raw else None
        except ValueError:
            messagebox.showerror("金额无效", "持有金额必须是整数")
            return
        self.store.correct_bankroll(value)
        self._changed()

    def apply_map_extent(self) -> None:
        raw = self.map_rows_var.get().strip()
        try:
            rows = int(raw) if raw else None
            self.store.correct_map_extent(rows, bool(self.map_exact_var.get()))
        except ValueError as exc:
            messagebox.showerror("地图行数无效", str(exc))
            return
        self._changed()

    def request_prediction(self) -> None:
        self._prediction_future = self.inference.submit(self.store)
        self.root.after(35, self._poll_prediction)

    def _poll_prediction(self) -> None:
        future = self._prediction_future
        if future is None or not future.done():
            self.root.after(35, self._poll_prediction)
            return
        try:
            result: PredictionResult = future.result()
        except Exception as exc:
            self.prediction_var.set(f"预测失败：{exc}")
            return
        if not self.inference.current(self.store, result):
            self.request_prediction()
            return
        self.prediction_var.set(format_prediction(result))

    def refresh(self) -> None:
        snapshot = self.store.snapshot()
        self.round_var.set(snapshot.round_number)
        self.bankroll_var.set(
            f"{snapshot.bankroll:,}" if snapshot.bankroll is not None else ""
        )
        self.map_rows_var.set(
            str(snapshot.map_rows) if snapshot.map_rows is not None else ""
        )
        self.map_exact_var.set(snapshot.map_height_exact)
        self.proof_var.set(
            "扫描完整性："
            f"{snapshot.completeness_source}；地图行数：{snapshot.map_rows_source}；"
            f"出价前证明：{snapshot.pre_bid_source}"
        )
        selected = self.event_tree.selection()
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        events = {
            (int(value["round_number"]), str(value["kind"])): value
            for value in snapshot.events
        }
        for round_number in range(1, snapshot.round_number + 1):
            for kind in ("public", "personal"):
                value = events.get((round_number, kind), {})
                iid = f"{round_number}:{kind}"
                self.event_tree.insert(
                    "",
                    tk.END,
                    iid=iid,
                    values=(
                        round_number,
                        "公共" if kind == "public" else "个人",
                        value.get("text") or "（待识别/可手填）",
                        f"{float(value.get('confidence') or 0):.2f}",
                        "人工" if value.get("human_locked") else "OCR",
                    ),
                )
        if selected and self.event_tree.exists(selected[0]):
            self.event_tree.selection_set(selected[0])
        self.map_editor.refresh()

    def reset(self) -> None:
        self.store.reset()
        self.monitor.set_round(1)
        self.prediction_var.set("等待每轮自动扫描")
        self.status_var.set("本局临时状态已从内存清空，继续监视大厅")
        self.refresh()

    def close(self) -> None:
        self.monitor.close()
        self.store.reset()
        self.inference.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
