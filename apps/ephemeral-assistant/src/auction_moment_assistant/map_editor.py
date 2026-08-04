from __future__ import annotations

from dataclasses import replace
from importlib import resources
from typing import Mapping
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .models import CatalogItem
from .observations import MapObservation, ObservationStore


CELL = 40
QUALITY_CHOICES = (
    ("", "无品质"),
    ("白", "白"),
    ("蓝", "蓝"),
    ("紫", "紫"),
    ("金", "金"),
    ("彩", "彩"),
)
QUALITY_CLASS_COLORS = {
    "白": "#ffffff",
    "蓝": "#1688ff",
    "紫": "#b33cff",
    "金": "#ffbd19",
    "彩": "#ff3fa4",
}
TOP_LEFT_NO_QUALITY_COLOR = "#00c8ff"
FULL_SHAPE_NO_QUALITY_COLOR = "#9b9cff"
IDENTITY_KNOWN_COLOR = "#00d46a"


def _field(item: MapObservation | Mapping, name: str, default=None):
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def evidence_state(item: MapObservation | Mapping) -> str:
    """Return one of the original reviewer's semantic evidence states."""

    spatial = str(_field(item, "spatial", "top_left") or "top_left")
    if spatial == "complete" and str(_field(item, "catalog_id", "") or ""):
        return "identity_known"
    if spatial in {"outline", "complete"}:
        return "full_shape"
    return "top_left"


def evidence_label(
    item: MapObservation | Mapping,
    catalog: Mapping[str, CatalogItem] | None = None,
) -> str:
    state = evidence_state(item)
    if state == "identity_known":
        catalog_id = str(_field(item, "catalog_id", "") or "")
        catalog_item = catalog.get(catalog_id) if catalog is not None else None
        return f"身份·{catalog_item.label if catalog_item else catalog_id}"
    quality = str(_field(item, "quality", "") or "")
    quality_label = quality or "无品质"
    prefix = "左上" if state == "top_left" else "形状"
    return f"{prefix}·{quality_label}"


def evidence_color(item: MapObservation | Mapping) -> str:
    state = evidence_state(item)
    if state == "identity_known":
        return IDENTITY_KNOWN_COLOR
    quality = str(_field(item, "quality", "") or "")
    if quality:
        return QUALITY_CLASS_COLORS.get(quality, "#e2e8f0")
    if state == "top_left":
        return TOP_LEFT_NO_QUALITY_COLOR
    return FULL_SHAPE_NO_QUALITY_COLOR


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


def display_geometry(item: MapObservation | Mapping) -> tuple[int, int]:
    if evidence_state(item) == "top_left":
        width = int(_field(item, "marker_width", 0) or 1)
        height = int(_field(item, "marker_height", 0) or 1)
    else:
        width = int(_field(item, "width", 0) or 1)
        height = int(_field(item, "height", 0) or 1)
    return max(1, min(3, width)), max(1, min(3, height))


class MapEditor(ttk.Frame):
    """13-state map reviewer plus explicit catalog identity correction."""

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
        self.catalog_items = tuple(sorted(catalog, key=lambda item: item.catalog_id))
        self.catalog = {item.catalog_id: item for item in self.catalog_items}
        self.on_change = on_change
        self.selected_key: tuple[int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_existing: MapObservation | None = None
        self._drag_moved = False
        self._draft_id: int | None = None
        self._identity_catalog_ids: list[str] = []
        self._identity_preview_photo = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        self.spatial_var = tk.StringVar(value="top_left")
        ttk.Label(toolbar, text="位置证据").pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar,
            text="左上角位置 (Q)",
            value="top_left",
            variable=self.spatial_var,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            toolbar,
            text="完整形状 (W)",
            value="outline",
            variable=self.spatial_var,
        ).pack(side=tk.LEFT, padx=2)

        attributes = ttk.Frame(self)
        attributes.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(attributes, text="品质").pack(side=tk.LEFT, padx=(0, 3))
        self.quality_var = tk.StringVar(value="")
        for value, label in QUALITY_CHOICES:
            ttk.Radiobutton(
                attributes,
                text=label,
                value=value,
                variable=self.quality_var,
                command=self._refresh_identity_choices,
            ).pack(side=tk.LEFT, padx=1)

        ttk.Label(attributes, text="规格").pack(side=tk.LEFT, padx=(12, 3))
        self.size_var = tk.StringVar(value="1x1")
        self.size_combo = ttk.Combobox(
            attributes,
            textvariable=self.size_var,
            values=tuple(f"{w}x{h}" for w in range(1, 4) for h in range(1, 4)),
            width=5,
            state="readonly",
        )
        self.size_combo.pack(side=tk.LEFT)
        self.size_combo.bind("<<ComboboxSelected>>", self._refresh_identity_choices)
        ttk.Button(
            attributes, text="应用到选中框", command=self.apply_selected
        ).pack(side=tk.LEFT, padx=8)

        help_row = ttk.Frame(self)
        help_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            help_row,
            text=(
                "空白处拖拽新增 · 拖动已有框移动 · 右键删除 · "
                "身份已知请先选框，再从右侧图鉴确认；自动 OCR 身份默认采用"
            ),
            foreground="#64748b",
        ).pack(side=tk.LEFT)
        self.selection_status = tk.StringVar(value="尚未选中地图框")
        ttk.Label(
            help_row,
            textvariable=self.selection_status,
            foreground="#0f766e",
        ).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(body)
        body.add(canvas_frame, weight=4)
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
        self.canvas.bind("q", lambda _event: self.spatial_var.set("top_left"))
        self.canvas.bind("w", lambda _event: self.spatial_var.set("outline"))
        self.canvas.bind(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(int(-event.delta / 120), "units"),
        )

        identity_panel = ttk.LabelFrame(
            body, text="完整身份（人工标注须明确确认）", padding=8
        )
        body.add(identity_panel, weight=2)
        preview = ttk.Frame(identity_panel)
        preview.pack(fill=tk.X, pady=(0, 7))
        self.identity_preview = ttk.Label(
            preview,
            text="在下方选择宝藏以预览",
            anchor=tk.CENTER,
            width=18,
        )
        self.identity_preview.pack(anchor=tk.CENTER)
        self.identity_preview_text = tk.StringVar(value="尚未选择宝藏")
        ttk.Label(
            preview,
            textvariable=self.identity_preview_text,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            foreground="#334155",
        ).pack(fill=tk.X, pady=(4, 0))
        self.identity_search_var = tk.StringVar()
        search = ttk.Entry(identity_panel, textvariable=self.identity_search_var)
        search.pack(fill=tk.X)
        self.identity_search_var.trace_add(
            "write", lambda *_args: self._refresh_identity_choices()
        )
        self.compatible_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            identity_panel,
            text="只显示与当前品质、已知尺寸兼容",
            variable=self.compatible_only_var,
            command=self._refresh_identity_choices,
        ).pack(anchor=tk.W, pady=(5, 4))

        list_frame = ttk.Frame(identity_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.identity_list = tk.Listbox(
            list_frame, exportselection=False, activestyle="dotbox", width=27
        )
        identity_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.identity_list.yview
        )
        self.identity_list.configure(yscrollcommand=identity_scroll.set)
        self.identity_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        identity_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.identity_list.bind("<Double-Button-1>", self.confirm_identity)
        self.identity_list.bind("<Return>", self.confirm_identity)
        self.identity_list.bind("<<ListboxSelect>>", self._update_identity_preview)

        self.identity_status = tk.StringVar(value="先在地图上选择一个框")
        ttk.Label(
            identity_panel,
            textvariable=self.identity_status,
            wraplength=245,
            foreground="#64748b",
        ).pack(fill=tk.X, pady=(5, 4))
        ttk.Button(
            identity_panel,
            text="确认所选身份并套用品质/尺寸",
            command=self.confirm_identity,
        ).pack(fill=tk.X)
        ttk.Button(
            identity_panel,
            text="清除精确身份（保留形状与品质）",
            command=self.clear_identity,
        ).pack(fill=tk.X, pady=(4, 0))
        self._refresh_identity_choices()

    def _size(self) -> tuple[int, int]:
        try:
            width, height = (
                int(value) for value in self.size_var.get().split("x", 1)
            )
            return width, height
        except (TypeError, ValueError):
            return 1, 1

    def _selected_item(self) -> MapObservation | None:
        if self.selected_key is None:
            return None
        for raw in self.store.snapshot().map_items:
            if (int(raw["row"]), int(raw["column"])) == self.selected_key:
                return MapObservation(**raw)
        return None

    def _refresh_identity_choices(self, _event=None) -> None:
        current = self._selected_item()
        quality = current.quality if current is not None else None
        known_size: tuple[int, int] | None = None
        if current is not None and evidence_state(current) != "top_left":
            if current.width and current.height:
                known_size = (int(current.width), int(current.height))
        search = self.identity_search_var.get().strip().casefold()
        candidate_rank = {
            str(catalog_id): index
            for index, (catalog_id, _score) in enumerate(
                current.identity_candidates if current is not None else ()
            )
        }
        matches = []
        for item in self.catalog_items:
            if self.compatible_only_var.get() and current is not None:
                if quality and item.quality != quality:
                    continue
                if known_size and (item.width, item.height) != known_size:
                    continue
            haystack = (
                f"{item.catalog_id} {item.name} {item.label} "
                f"{item.quality} {item.size}"
            ).casefold()
            if search and search not in haystack:
                continue
            matches.append(item)
        matches.sort(
            key=lambda item: (
                candidate_rank.get(item.catalog_id, 10_000), item.label
            )
        )

        self.identity_list.delete(0, tk.END)
        self._identity_catalog_ids = []
        selected_index = None
        for item in matches:
            candidate = "OCR候选 · " if item.catalog_id in candidate_rank else ""
            self.identity_list.insert(
                tk.END,
                (
                    f"{candidate}{item.label} · {item.quality} · "
                    f"{item.size} · {item.value:,}"
                ),
            )
            self._identity_catalog_ids.append(item.catalog_id)
            if current is not None and current.catalog_id == item.catalog_id:
                selected_index = len(self._identity_catalog_ids) - 1
        if selected_index is not None:
            self.identity_list.selection_set(selected_index)
            self.identity_list.see(selected_index)
        elif matches:
            self.identity_list.selection_set(0)
        if current is None:
            self.identity_status.set("先在地图上选择一个框")
        else:
            state = evidence_label(current, self.catalog)
            self.identity_status.set(
                f"当前：{state}；可选 {len(matches)} 个兼容图鉴项"
            )
        self._update_identity_preview()

    def _update_identity_preview(self, _event=None) -> None:
        item = self._selected_catalog_item()
        if item is None:
            self._identity_preview_photo = None
            self.identity_preview.configure(
                image="", text="没有兼容的宝藏可供预览"
            )
            self.identity_preview_text.set("尚未选择宝藏")
            return
        try:
            resource = resources.files(__package__).joinpath(
                "catalog_previews", f"{item.catalog_id}.png"
            )
            with resource.open("rb") as handle:
                image = Image.open(handle).convert("RGBA")
                image.thumbnail((128, 128), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
        except Exception:
            self._identity_preview_photo = None
            self.identity_preview.configure(image="", text="预览图暂不可用")
        else:
            self._identity_preview_photo = photo
            self.identity_preview.configure(image=photo, text="")
        self.identity_preview_text.set(
            f"{item.label}\n{item.quality} · {item.size} · {item.value:,}"
        )

    def _grid_position(self, event) -> tuple[int, int]:
        self.canvas.focus_set()
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return max(0, int(y // CELL)), max(0, min(9, int(x // CELL)))

    def _item_at(self, position: tuple[int, int]) -> MapObservation | None:
        row, column = position
        for raw in reversed(self.store.snapshot().map_items):
            item = MapObservation(**raw)
            width, height = display_geometry(item)
            if (
                item.row <= row < item.row + height
                and item.column <= column < item.column + width
            ):
                return item
        return None

    def _select(self, item: MapObservation | None) -> None:
        self.selected_key = item.key if item is not None else None
        if item is None:
            self.selection_status.set("尚未选中地图框")
        else:
            self.spatial_var.set(
                "top_left" if evidence_state(item) == "top_left" else "outline"
            )
            self.quality_var.set(item.quality or "")
            width, height = display_geometry(item)
            self.size_var.set(f"{width}x{height}")
            self.selection_status.set(
                (
                    f"已选 R{item.row} C{item.column} · "
                    f"{evidence_label(item, self.catalog)}"
                )
            )
        self._refresh_identity_choices()
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
            width, height = display_geometry(self.drag_existing)
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
        if existing is not None and self._drag_moved:
            width, _height = display_geometry(existing)
            row, column = end
            column = min(column, 10 - width)
            moved = replace(existing, row=row, column=column)
            if moved.key != existing.key:
                self.store.remove_map_item(*existing.key)
                self.store.upsert_map_item(moved)
                self.selected_key = moved.key
                self.on_change()
        elif existing is None:
            row, column, width, height = grid_rectangle(self.drag_start, end)
            quality = self.quality_var.get() or None
            if self.spatial_var.get() == "top_left":
                item = MapObservation(
                    row=row,
                    column=column,
                    marker_width=width,
                    marker_height=height,
                    quality=quality,
                    spatial="top_left",
                )
            else:
                item = MapObservation(
                    row=row,
                    column=column,
                    width=width,
                    height=height,
                    quality=quality,
                    spatial="outline",
                )
            self.store.upsert_map_item(item)
            self.selected_key = item.key
            self.on_change()
        self.drag_start = None
        self.drag_existing = None
        self._drag_moved = False
        self._draft_id = None
        self._select(self._selected_item())

    def _delete_at(self, event) -> None:
        item = self._item_at(self._grid_position(event))
        if item is None:
            return
        self.store.remove_map_item(*item.key)
        if self.selected_key == item.key:
            self.selected_key = None
        self._select(None)
        self.on_change()

    def apply_selected(self) -> None:
        current = self._selected_item()
        if current is None:
            messagebox.showinfo("尚未选中", "先在地图上点选一个对象")
            return
        quality = self.quality_var.get() or None
        desired_spatial = self.spatial_var.get()
        if desired_spatial == "top_left":
            marker_width, marker_height = display_geometry(current)
            updated = replace(
                current,
                width=None,
                height=None,
                marker_width=marker_width,
                marker_height=marker_height,
                quality=quality,
                catalog_id=None,
                spatial="top_left",
            )
        else:
            if current.width and current.height:
                width, height = int(current.width), int(current.height)
            elif current.marker_width and current.marker_height:
                width, height = int(current.marker_width), int(current.marker_height)
            else:
                width, height = self._size()
            if current.column + width > 10:
                messagebox.showerror("规格超界", "该规格会超出地图右边界")
                return
            updated = replace(
                current,
                width=width,
                height=height,
                marker_width=None,
                marker_height=None,
                quality=quality,
                catalog_id=None,
                spatial="outline",
            )
        self.store.upsert_map_item(updated)
        self._select(updated)
        self.on_change()

    def _selected_catalog_item(self) -> CatalogItem | None:
        selection = self.identity_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if not 0 <= index < len(self._identity_catalog_ids):
            return None
        return self.catalog.get(self._identity_catalog_ids[index])

    def confirm_identity(self, _event=None) -> None:
        current = self._selected_item()
        if current is None:
            messagebox.showinfo("尚未选中", "先在地图上点选一个对象")
            return
        catalog_item = self._selected_catalog_item()
        if catalog_item is None:
            messagebox.showinfo("尚未选择身份", "请从图鉴列表中选中一个具体身份")
            return
        if current.column + catalog_item.width > 10:
            messagebox.showerror("身份规格超界", "该图鉴规格会超出地图右边界")
            return
        snapshot = self.store.snapshot()
        if (
            snapshot.map_height_exact
            and snapshot.map_rows is not None
            and current.row + catalog_item.height > snapshot.map_rows
        ):
            messagebox.showerror("身份规格超界", "该图鉴规格会超出已确认地图底部")
            return
        updated = replace(
            current,
            width=catalog_item.width,
            height=catalog_item.height,
            marker_width=None,
            marker_height=None,
            quality=catalog_item.quality,
            catalog_id=catalog_item.catalog_id,
            spatial="complete",
        )
        self.store.upsert_map_item(updated)
        self._select(updated)
        self.on_change()

    def clear_identity(self) -> None:
        current = self._selected_item()
        if current is None:
            messagebox.showinfo("尚未选中", "先在地图上点选一个对象")
            return
        if evidence_state(current) != "identity_known":
            return
        updated = replace(current, catalog_id=None, spatial="outline")
        self.store.upsert_map_item(updated)
        self._select(updated)
        self.on_change()

    def refresh(self) -> None:
        snapshot = self.store.snapshot()
        max_item_row = max(
            (
                int(item["row"]) + display_geometry(item)[1]
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
            width, height = display_geometry(raw)
            key = (row, column)
            selected = key == self.selected_key
            color = evidence_color(raw)
            self.canvas.create_rectangle(
                column * CELL + 3,
                row * CELL + 3,
                (column + width) * CELL - 3,
                (row + height) * CELL - 3,
                fill="",
                outline=color,
                width=6 if selected else 3,
                dash=(8, 4) if selected else None,
            )
            source = "人工" if raw.get("human_locked") else "OCR"
            text_id = self.canvas.create_text(
                column * CELL + 7,
                row * CELL + 7,
                text=f"{evidence_label(raw, self.catalog)} · {source}",
                anchor=tk.NW,
                fill="#0f172a" if color == "#ffffff" else "#ffffff",
                font=("Microsoft YaHei UI", 9, "bold"),
            )
            bounds = self.canvas.bbox(text_id)
            if bounds:
                background = self.canvas.create_rectangle(
                    bounds[0] - 3,
                    bounds[1] - 2,
                    bounds[2] + 3,
                    bounds[3] + 2,
                    fill=color,
                    outline=color,
                )
                self.canvas.tag_lower(background, text_id)
        self.canvas.configure(scrollregion=(0, 0, 10 * CELL, rows * CELL))
        self._refresh_identity_choices()
