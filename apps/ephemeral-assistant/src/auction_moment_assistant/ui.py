from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

from .models import CatalogItem
from .observations import MapObservation, ObservationStore
from .predictor import PredictionResult
from .runtime import CaptureOutcome, CapturePipeline, InferenceCoordinator
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
        self.catalog = {item.catalog_id: item for item in catalog}
        self.root = tk.Tk()
        self.root.title("Auction Moment · 无持久化视觉助手")
        self.root.geometry("1180x780")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._capture_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="auction-capture"
        )
        self._capture_future: Future | None = None
        self._prediction_future: Future | None = None
        self._preview_photo = None
        self._monitoring = False
        self._build()
        self.refresh()

    def _build(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)
        self.round_var = tk.IntVar(value=1)
        self.offset_var = tk.IntVar(value=0)
        self.complete_var = tk.BooleanVar(value=False)
        ttk.Label(controls, text="轮次").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls, from_=1, to=5, width=4, textvariable=self.round_var
        ).pack(side=tk.LEFT, padx=(3, 12))
        ttk.Label(controls, text="当前视口起始行").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls, from_=0, to=30, width=5, textvariable=self.offset_var
        ).pack(side=tk.LEFT, padx=(3, 12))
        ttk.Button(controls, text="读取当前画面", command=self.capture).pack(
            side=tk.LEFT
        )
        self.monitor_button = ttk.Button(
            controls, text="开始监视", command=self.toggle_monitor
        )
        self.monitor_button.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(
            controls,
            text="本轮可见内容已核对完整",
            variable=self.complete_var,
            command=self.confirm_complete,
        ).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="清空本局内存", command=self.reset).pack(
            side=tk.RIGHT
        )

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            padding=(8, 0),
            foreground="#334155",
        ).pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.preview = ttk.Label(left, text="尚未读取画面", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)

        event_box = ttk.LabelFrame(right, text="OCR 与人工修正", padding=8)
        event_box.pack(fill=tk.X)
        self.public_var = tk.StringVar()
        self.personal_var = tk.StringVar()
        self.bankroll_var = tk.StringVar()
        for row, (label, variable) in enumerate(
            (
                ("公共事件", self.public_var),
                ("个人事件", self.personal_var),
                ("持有金额", self.bankroll_var),
            )
        ):
            ttk.Label(event_box, text=label).grid(
                row=row, column=0, sticky=tk.W, pady=3
            )
            ttk.Entry(event_box, textvariable=variable, width=43).grid(
                row=row, column=1, sticky=tk.EW, pady=3
            )
        event_box.columnconfigure(1, weight=1)
        ttk.Button(
            event_box, text="应用人工修正并重算", command=self.apply_event_edits
        ).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(7, 0))

        map_box = ttk.LabelFrame(right, text="当前内存地图证据", padding=8)
        map_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        columns = ("pos", "size", "quality", "identity", "confidence", "source")
        self.tree = ttk.Treeview(
            map_box, columns=columns, show="headings", height=12
        )
        for column, label, width in (
            ("pos", "位置", 55),
            ("size", "规格", 50),
            ("quality", "品质", 45),
            ("identity", "身份", 65),
            ("confidence", "置信度", 60),
            ("source", "来源", 55),
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)
        map_actions = ttk.Frame(map_box)
        map_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(map_actions, text="补充", command=self.add_item).pack(
            side=tk.LEFT
        )
        ttk.Button(map_actions, text="修改", command=self.edit_item).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(map_actions, text="删除误检", command=self.delete_item).pack(
            side=tk.LEFT
        )

        prediction_box = ttk.LabelFrame(
            right, text="研究估计（不自动出价）", padding=8
        )
        prediction_box.pack(fill=tk.X, pady=(8, 0))
        self.prediction_var = tk.StringVar(value="等待证据")
        ttk.Label(
            prediction_box,
            textvariable=self.prediction_var,
            justify=tk.LEFT,
            wraplength=430,
        ).pack(fill=tk.X)

    def capture(self) -> None:
        if self._capture_future is not None and not self._capture_future.done():
            return
        self.store.set_round(self.round_var.get())
        self.status_var.set("正在内存中读取并识别当前画面…")
        self._capture_future = self._capture_executor.submit(
            self.pipeline.capture_once,
            expected_round=self.round_var.get(),
            offset_rows=self.offset_var.get(),
        )
        self.root.after(60, self._poll_capture)

    def _poll_capture(self) -> None:
        future = self._capture_future
        if future is None or not future.done():
            self.root.after(60, self._poll_capture)
            return
        try:
            outcome: CaptureOutcome = future.result()
        except Exception as exc:
            self.status_var.set(f"读取失败：{exc}")
        else:
            if outcome.final_detected:
                self.status_var.set("检测到终局：本局画面和修正状态已从内存清空")
            else:
                suffix = "；".join(outcome.errors)
                self.status_var.set(
                    f"识别完成：地图候选 {outcome.map_item_count} 个"
                    + (f"；{suffix}" if suffix else "")
                )
            self.refresh()
            self.request_prediction()
        finally:
            self._capture_future = None
        if self._monitoring:
            self.root.after(1200, self.capture)

    def toggle_monitor(self) -> None:
        self._monitoring = not self._monitoring
        self.monitor_button.configure(
            text="停止监视" if self._monitoring else "开始监视"
        )
        if self._monitoring:
            self.capture()

    def apply_event_edits(self) -> None:
        round_number = int(self.round_var.get())
        for kind, text in (
            ("public", self.public_var.get()),
            ("personal", self.personal_var.get()),
        ):
            self.store.correct_event(
                round_number, kind, text, event_semantics(text)
            )
        bankroll_text = self.bankroll_var.get().replace(",", "").strip()
        try:
            bankroll = int(bankroll_text) if bankroll_text else None
        except ValueError:
            messagebox.showerror("金额无效", "持有金额必须是整数")
            return
        self.store.correct_bankroll(bankroll)
        self.refresh()
        self.request_prediction()

    def _ask_item(self, existing: MapObservation | None = None):
        row = simpledialog.askinteger(
            "位置", "行（从 0 开始）", initialvalue=existing.row if existing else 0,
            minvalue=0, maxvalue=50, parent=self.root
        )
        if row is None:
            return None
        column = simpledialog.askinteger(
            "位置", "列（0-9）", initialvalue=existing.column if existing else 0,
            minvalue=0, maxvalue=9, parent=self.root
        )
        if column is None:
            return None
        quality = simpledialog.askstring(
            "品质", "品质：白/蓝/紫/金/彩；未知可留空",
            initialvalue=existing.quality if existing and existing.quality else "",
            parent=self.root,
        )
        size = simpledialog.askstring(
            "规格", "规格，例如 2x3；未知可留空",
            initialvalue=(
                f"{existing.width}x{existing.height}"
                if existing and existing.width and existing.height else ""
            ),
            parent=self.root,
        )
        catalog_id = simpledialog.askstring(
            "身份", "目录 ID，例如 C038；未知可留空",
            initialvalue=existing.catalog_id if existing and existing.catalog_id else "",
            parent=self.root,
        )
        width = height = None
        if size:
            try:
                width, height = (int(part) for part in size.lower().split("x", 1))
            except (ValueError, TypeError):
                messagebox.showerror("规格无效", "规格必须类似 2x3")
                return None
        quality = (quality or "").strip() or None
        catalog_id = (catalog_id or "").strip() or None
        if quality not in {None, "白", "蓝", "紫", "金", "彩"}:
            messagebox.showerror("品质无效", "品质必须是白、蓝、紫、金、彩或留空")
            return None
        if catalog_id and catalog_id not in self.catalog:
            messagebox.showerror("身份无效", f"公开图鉴中没有 {catalog_id}")
            return None
        return MapObservation(
            row=row,
            column=column,
            width=width,
            height=height,
            quality=quality,
            catalog_id=catalog_id,
            spatial="complete" if catalog_id else "outline" if width else "top_left",
        )

    def add_item(self) -> None:
        item = self._ask_item()
        if item is not None:
            self.store.upsert_map_item(item)
            self.refresh()
            self.request_prediction()

    def _selected_item(self) -> MapObservation | None:
        selection = self.tree.selection()
        if not selection:
            return None
        row, column = (int(value) for value in selection[0].split(":"))
        for raw in self.store.snapshot().map_items:
            if int(raw["row"]) == row and int(raw["column"]) == column:
                return MapObservation(**raw)
        return None

    def edit_item(self) -> None:
        existing = self._selected_item()
        if existing is None:
            messagebox.showinfo("未选择", "请先选择一条地图证据")
            return
        item = self._ask_item(existing)
        if item is not None:
            if item.key != existing.key:
                self.store.remove_map_item(*existing.key)
            self.store.upsert_map_item(item)
            self.refresh()
            self.request_prediction()

    def delete_item(self) -> None:
        existing = self._selected_item()
        if existing is None:
            return
        self.store.remove_map_item(*existing.key)
        self.refresh()
        self.request_prediction()

    def confirm_complete(self) -> None:
        self.store.confirm_complete(bool(self.complete_var.get()))
        self.request_prediction()

    def request_prediction(self) -> None:
        self._prediction_future = self.inference.submit(self.store)
        self.root.after(30, self._poll_prediction)

    def _poll_prediction(self) -> None:
        future = self._prediction_future
        if future is None or not future.done():
            self.root.after(30, self._poll_prediction)
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
                f"状态：{result.status}\n兼容世界：0\n问题：{', '.join(result.issues)}"
            )
        else:
            self.prediction_var.set(
                f"状态：{result.status}\n"
                f"兼容世界：{result.compatible_worlds}\n"
                f"P10 / P50 / P90：{result.p10:,} / {result.p50:,} / {result.p90:,}\n"
                f"范围：{result.minimum:,} - {result.maximum:,}\n"
                "区间未校准，actionable=false"
            )

    def refresh(self) -> None:
        snapshot = self.store.snapshot()
        self.round_var.set(snapshot.round_number)
        self.complete_var.set(snapshot.completeness_confirmed)
        events = {
            (int(value["round_number"]), str(value["kind"])): value
            for value in snapshot.events
        }
        public = events.get((snapshot.round_number, "public"), {})
        personal = events.get((snapshot.round_number, "personal"), {})
        self.public_var.set(str(public.get("text") or ""))
        self.personal_var.set(str(personal.get("text") or ""))
        self.bankroll_var.set(
            f"{snapshot.bankroll:,}" if snapshot.bankroll is not None else ""
        )
        for row in self.tree.get_children():
            self.tree.delete(row)
        for item in snapshot.map_items:
            size = (
                f"{item['width']}x{item['height']}"
                if item.get("width") and item.get("height") else "?"
            )
            self.tree.insert(
                "",
                tk.END,
                iid=f"{item['row']}:{item['column']}",
                values=(
                    f"{item['row']},{item['column']}",
                    size,
                    item.get("quality") or "?",
                    item.get("catalog_id") or "?",
                    f"{float(item.get('confidence') or 0):.2f}",
                    item.get("source") or "?",
                ),
            )
        frame = self.store.frame_copy()
        if frame is not None:
            image = Image.fromarray(frame)
            image.thumbnail((690, 600))
            self._preview_photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self._preview_photo, text="")
        else:
            self._preview_photo = None
            self.preview.configure(image="", text="尚未读取画面")

    def reset(self) -> None:
        self.store.reset()
        self.refresh()
        self.prediction_var.set("等待证据")
        self.status_var.set("本局内存状态已清空")

    def close(self) -> None:
        self._monitoring = False
        self.store.reset()
        self.inference.close()
        self._capture_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
