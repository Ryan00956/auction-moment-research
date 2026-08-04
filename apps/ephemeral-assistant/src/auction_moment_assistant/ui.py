from __future__ import annotations

from concurrent.futures import Future
import tkinter as tk
from tkinter import messagebox, ttk

from .models import CatalogItem
from .observations import MapObservation, ObservationStore
from .predictor import PredictionResult
from .runtime import (
    CapturePipeline,
    EphemeralStateMonitor,
    InferenceCoordinator,
)
from .semantics import event_semantics


CELL = 48
QUALITY_COLORS = {
    None: "#e2e8f0",
    "白": "#f8fafc",
    "蓝": "#bfdbfe",
    "紫": "#ddd6fe",
    "金": "#fde68a",
    "彩": "#fecdd3",
}


def grid_rectangle(
    start: tuple[int, int], end: tuple[int, int]
) -> tuple[int, int, int, int]:
    row_a, column_a = start
    row_b, column_b = end
    row = min(row_a, row_b)
    column = min(column_a, column_b)
    width = min(3, abs(column_b - column_a) + 1)
    height = min(3, abs(row_b - row_a) + 1)
    width = min(width, 10 - column)
    return row, column, width, height


class MapEditor(ttk.Frame):
    def __init__(
        self,
        parent,
        *,
        store: ObservationStore,
        catalog: tuple[CatalogItem, ...],
        on_change,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.catalog = {item.catalog_id: item for item in catalog}
        self.on_change = on_change
        self.selected_key: tuple[int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_existing: MapObservation | None = None
        self._drag_moved = False
        self._draft_id: int | None = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        self.quality_var = tk.StringVar(value="金")
        ttk.Label(toolbar, text="品质").pack(side=tk.LEFT)
        for quality in ("白", "蓝", "紫", "金", "彩"):
            ttk.Radiobutton(
                toolbar,
                text=quality,
                value=quality,
                variable=self.quality_var,
                command=self._refresh_identity_choices,
            ).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="规格").pack(side=tk.LEFT, padx=(12, 3))
        self.size_var = tk.StringVar(value="1x1")
        self.size_combo = ttk.Combobox(
            toolbar,
            textvariable=self.size_var,
            values=tuple(f"{w}x{h}" for w in range(1, 4) for h in range(1, 4)),
            width=5,
            state="readonly",
        )
        self.size_combo.pack(side=tk.LEFT)
        self.size_combo.bind("<<ComboboxSelected>>", self._refresh_identity_choices)
        ttk.Label(toolbar, text="身份").pack(side=tk.LEFT, padx=(12, 3))
        self.identity_var = tk.StringVar()
        self.identity_combo = ttk.Combobox(
            toolbar, textvariable=self.identity_var, width=13, state="readonly"
        )
        self.identity_combo.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="应用到选中", command=self.apply_selected).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Label(
            toolbar,
            text="空白处拖拽新增 · 拖动已有对象移动 · 右键删除",
            foreground="#64748b",
        ).pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            width=10 * CELL + 2,
        )
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._delete_at)
        self.canvas.bind(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(
                int(-event.delta / 120), "units"
            ),
        )
        self._refresh_identity_choices()

    def _size(self) -> tuple[int, int]:
        try:
            return tuple(int(value) for value in self.size_var.get().split("x", 1))
        except (TypeError, ValueError):
            return 1, 1

    def _refresh_identity_choices(self, _event=None) -> None:
        width, height = self._size()
        quality = self.quality_var.get()
        values = ("",) + tuple(
            item.catalog_id
            for item in self.catalog.values()
            if item.quality == quality
            and item.width == width
            and item.height == height
        )
        self.identity_combo.configure(values=values)
        if self.identity_var.get() not in values:
            self.identity_var.set("")

    def _grid_position(self, event) -> tuple[int, int]:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return max(0, int(y // CELL)), max(0, min(9, int(x // CELL)))

    def _item_at(self, position: tuple[int, int]) -> MapObservation | None:
        row, column = position
        for raw in reversed(self.store.snapshot().map_items):
            item = MapObservation(**raw)
            width = int(item.width or 1)
            height = int(item.height or 1)
            if (
                item.row <= row < item.row + height
                and item.column <= column < item.column + width
            ):
                return item
        return None

    def _select(self, item: MapObservation | None) -> None:
        self.selected_key = item.key if item is not None else None
        if item is not None:
            if item.quality:
                self.quality_var.set(item.quality)
            if item.width and item.height:
                self.size_var.set(f"{item.width}x{item.height}")
            self._refresh_identity_choices()
            self.identity_var.set(item.catalog_id or "")
        self.refresh()

    def _press(self, event) -> None:
        self.drag_start = self._grid_position(event)
        self.drag_existing = self._item_at(self.drag_start)
        self._drag_moved = False
        self._select(self.drag_existing)

    def _motion(self, event) -> None:
        if self.drag_start is None:
            return
        end = self._grid_position(event)
        self._drag_moved = self._drag_moved or end != self.drag_start
        if self.drag_existing is not None:
            width = int(self.drag_existing.width or 1)
            height = int(self.drag_existing.height or 1)
            row, column = end
            column = min(column, 10 - width)
        else:
            row, column, width, height = grid_rectangle(self.drag_start, end)
        coordinates = (
            column * CELL + 2,
            row * CELL + 2,
            (column + width) * CELL - 2,
            (row + height) * CELL - 2,
        )
        if self._draft_id is None:
            self._draft_id = self.canvas.create_rectangle(
                *coordinates, outline="#0f172a", width=2, dash=(5, 3)
            )
        else:
            self.canvas.coords(self._draft_id, *coordinates)

    def _release(self, event) -> None:
        if self.drag_start is None:
            return
        end = self._grid_position(event)
        existing = self.drag_existing
        if existing is not None and not self._drag_moved:
            pass
        elif existing is not None:
            width = int(existing.width or 1)
            height = int(existing.height or 1)
            row, column = end
            column = min(column, 10 - width)
            item = MapObservation(
                row=row,
                column=column,
                width=width,
                height=height,
                quality=existing.quality,
                catalog_id=existing.catalog_id,
                spatial=existing.spatial,
                first_seen_round=existing.first_seen_round,
            )
            if item.key != existing.key:
                self.store.remove_map_item(*existing.key)
                self.store.upsert_map_item(item)
                self.selected_key = item.key
                self.on_change()
        else:
            row, column, width, height = grid_rectangle(self.drag_start, end)
            quality = self.quality_var.get() or None
            catalog_id = self.identity_var.get() or None
            catalog_item = self.catalog.get(catalog_id or "")
            if catalog_item is not None and (
                catalog_item.width != width
                or catalog_item.height != height
                or catalog_item.quality != quality
            ):
                # Geometry drawn on the map is authoritative. A possibly stale
                # toolbar identity must never silently change that rectangle.
                catalog_id = None
            self.store.upsert_map_item(
                MapObservation(
                    row=row,
                    column=column,
                    width=width,
                    height=height,
                    quality=quality,
                    catalog_id=catalog_id,
                    spatial="complete" if catalog_id else "outline",
                )
            )
            self.selected_key = (row, column)
            self.on_change()
        self.drag_start = None
        self.drag_existing = None
        self._drag_moved = False
        self._draft_id = None
        self.refresh()

    def _delete_at(self, event) -> None:
        item = self._item_at(self._grid_position(event))
        if item is None:
            return
        self.store.remove_map_item(*item.key)
        if self.selected_key == item.key:
            self.selected_key = None
        self.refresh()
        self.on_change()

    def apply_selected(self) -> None:
        if self.selected_key is None:
            messagebox.showinfo("尚未选中", "先在地图上点选一个对象")
            return
        raw = next(
            (
                value
                for value in self.store.snapshot().map_items
                if (int(value["row"]), int(value["column"])) == self.selected_key
            ),
            None,
        )
        if raw is None:
            return
        width, height = self._size()
        catalog_id = self.identity_var.get() or None
        catalog_item = self.catalog.get(catalog_id or "")
        quality = self.quality_var.get() or None
        if catalog_item is not None:
            quality = catalog_item.quality
            width, height = catalog_item.width, catalog_item.height
        if self.selected_key[1] + width > 10:
            messagebox.showerror("规格超界", "该规格会超出地图右边界")
            return
        self.store.upsert_map_item(
            MapObservation(
                row=self.selected_key[0],
                column=self.selected_key[1],
                width=width,
                height=height,
                quality=quality,
                catalog_id=catalog_id,
                spatial="complete" if catalog_id else "outline",
                first_seen_round=raw.get("first_seen_round"),
            )
        )
        self.refresh()
        self.on_change()

    def refresh(self) -> None:
        snapshot = self.store.snapshot()
        max_item_row = max(
            (
                int(item["row"]) + int(item.get("height") or 1)
                for item in snapshot.map_items
            ),
            default=10,
        )
        rows = max(10, int(snapshot.map_rows or 0), max_item_row)
        self.canvas.delete("all")
        for column in range(11):
            x = column * CELL
            self.canvas.create_line(x, 0, x, rows * CELL, fill="#cbd5e1")
        for row in range(rows + 1):
            y = row * CELL
            self.canvas.create_line(0, y, 10 * CELL, y, fill="#cbd5e1")
            if row < rows:
                self.canvas.create_text(
                    4, y + 4, text=str(row), anchor=tk.NW, fill="#94a3b8"
                )
        for raw in snapshot.map_items:
            row = int(raw["row"])
            column = int(raw["column"])
            width = int(raw.get("width") or 1)
            height = int(raw.get("height") or 1)
            key = (row, column)
            outline = "#0f172a" if key == self.selected_key else "#475569"
            line_width = 3 if key == self.selected_key else 2
            dash = (4, 2) if not raw.get("width") else None
            self.canvas.create_rectangle(
                column * CELL + 3,
                row * CELL + 3,
                (column + width) * CELL - 3,
                (row + height) * CELL - 3,
                fill=QUALITY_COLORS.get(raw.get("quality"), "#e2e8f0"),
                outline=outline,
                width=line_width,
                dash=dash,
            )
            label = raw.get("catalog_id") or raw.get("quality") or "?"
            source = "人工" if raw.get("human_locked") else "OCR"
            self.canvas.create_text(
                (column + width / 2) * CELL,
                (row + height / 2) * CELL,
                text=f"{label}\n{source}",
                justify=tk.CENTER,
                fill="#0f172a",
            )
        self.canvas.configure(scrollregion=(0, 0, 10 * CELL, rows * CELL))


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
                "无出价、无截图/事件/预测/结算落盘。"
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
            right, text="v6 + world-model-v2 预测（不自动出价）", padding=8
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
        if result.p50 is None:
            self.prediction_var.set(
                f"状态：{result.status}\n尚不能运行最新模型\n"
                f"待补证据：{', '.join(result.issues) or '未知'}"
            )
            return
        model_line = ""
        if result.v6_prediction is not None and result.v2_prediction is not None:
            model_line = (
                f"v6 / v2：{result.v6_prediction:,} / {result.v2_prediction:,}"
                f"（v2 权重 {result.generation_weight:.0%}）\n"
            )
        self.prediction_var.set(
            f"状态：已完成本轮预测待机\n"
            f"{model_line}"
            f"P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}\n"
            f"范围：{result.minimum:,} - {result.maximum:,}\n"
            f"提示：{', '.join(result.issues)}\n"
            "区间未校准，actionable=false"
        )

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
