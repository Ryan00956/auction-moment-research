from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from .models import CatalogItem
from .observations import MapObservation


GRID_COLUMNS = 10
CELL_PIXELS = 60
VIEWPORT_PIXELS = 600
MAX_STITCH_SHIFT_PIXELS = VIEWPORT_PIXELS - 80
ROUND_GRID_CROP = (49, 110, 647, 705)
QUALITY_TOKENS = {
    "white": "白",
    "blue": "蓝",
    "purple": "紫",
    "gold": "金",
    "rainbow": "彩",
}
SPATIAL_RANK = {"top_left": 1, "outline": 2, "complete": 3}


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ViewportRecognition:
    items: tuple[MapObservation, ...]
    offsets_pixels: tuple[int, ...]
    estimated_rows: int
    alignment_stable: bool


class VisionBackend(Protocol):
    def detect_spatial(self, image: np.ndarray) -> Sequence[Detection]: ...

    def detect_attributes(self, image: np.ndarray) -> Sequence[Detection]: ...

    def classify(
        self, crops: Sequence[np.ndarray]
    ) -> Sequence[Sequence[tuple[str, float]]]: ...


def normalize_map_viewport(frame_rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(frame_rgb, dtype=np.uint8)
    left, top, right, bottom = ROUND_GRID_CROP
    if array.shape[0] < bottom or array.shape[1] < right:
        raise VisionError(
            f"画面尺寸 {array.shape[1]}x{array.shape[0]} 小于地图标定区域"
        )
    crop = array[top:bottom, left:right]
    rgb = cv2.resize(
        crop,
        (VIEWPORT_PIXELS, VIEWPORT_PIXELS),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_class(class_name: str) -> tuple[str, str | None]:
    if class_name == "top_left_marker":
        return "top_left", None
    if class_name in {"full_shape_marker"}:
        return "outline", None
    if class_name in {"revealed_icon", "identity_known"}:
        return "complete", None
    if class_name.startswith("top_left_"):
        return "top_left", QUALITY_TOKENS.get(
            class_name.removeprefix("top_left_")
        )
    if class_name.startswith("full_shape_"):
        return "outline", QUALITY_TOKENS.get(
            class_name.removeprefix("full_shape_")
        )
    raise VisionError(f"未知视觉类别：{class_name}")


class UltralyticsVisionBackend:
    def __init__(self, models: Mapping[str, object], *, device: str = "auto"):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionError("视觉识别需要安装 ultralytics 与 torch") from exc
        self._yolo = YOLO
        self.device = (
            "0" if device == "auto" and torch.cuda.is_available() else "cpu"
            if device == "auto"
            else str(device)
        )
        self.paths = {key: str(value) for key, value in models.items()}
        self._spatial = None
        self._attribute = None
        self._identity = None

    def _model(self, role: str):
        attribute = {
            "spatial_detector": "_spatial",
            "attribute_detector": "_attribute",
            "identity_classifier": "_identity",
        }[role]
        model = getattr(self, attribute)
        if model is None:
            path = self.paths.get(role)
            if not path:
                raise VisionError(f"缺少模型角色：{role}")
            model = self._yolo(path)
            setattr(self, attribute, model)
        return model

    def _detect(self, role: str, image: np.ndarray) -> list[Detection]:
        model = self._model(role)
        results = model.predict(
            source=[image],
            imgsz=640,
            conf=0.12,
            iou=0.45,
            max_det=300,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for index in range(len(result.boxes)):
                class_index = int(result.boxes.cls[index].detach().cpu().item())
                detections.append(
                    Detection(
                        class_name=str(model.names[class_index]),
                        confidence=float(
                            result.boxes.conf[index].detach().cpu().item()
                        ),
                        box_xyxy=tuple(
                            float(value)
                            for value in result.boxes.xyxy[index]
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                    )
                )
        return detections

    def detect_spatial(self, image: np.ndarray) -> Sequence[Detection]:
        return self._detect("spatial_detector", image)

    def detect_attributes(self, image: np.ndarray) -> Sequence[Detection]:
        return self._detect("attribute_detector", image)

    def classify(
        self, crops: Sequence[np.ndarray]
    ) -> Sequence[Sequence[tuple[str, float]]]:
        if not crops:
            return []
        model = self._model("identity_classifier")
        results = model.predict(
            source=list(crops), imgsz=224, device=self.device, verbose=False
        )
        batches = []
        for result in results:
            probabilities = result.probs.data.detach().cpu().numpy()
            indices = np.argsort(probabilities)[::-1][:10]
            batches.append(
                [
                    (str(model.names[int(index)]), float(probabilities[index]))
                    for index in indices
                ]
            )
        return batches


def _project(
    detection: Detection, *, offset_pixels: int
) -> tuple[MapObservation, tuple[int, int, int, int]] | None:
    try:
        spatial, quality = decode_class(detection.class_name)
    except VisionError:
        return None
    x1, y1, x2, y2 = detection.box_xyxy
    column = int(round(x1 / CELL_PIXELS))
    local_row = int(round(y1 / CELL_PIXELS))
    if not 0 <= column < GRID_COLUMNS or not 0 <= local_row < GRID_COLUMNS:
        return None
    x_residual = abs(x1 - column * CELL_PIXELS)
    y_residual = abs(y1 - local_row * CELL_PIXELS)
    if x_residual > 23 or y_residual > 23:
        return None
    width = max(1, int(round(x2 / CELL_PIXELS)) - column)
    height = max(1, int(round((y2 - y1) / CELL_PIXELS)))
    width = min(width, GRID_COLUMNS - column)
    height = min(height, GRID_COLUMNS - local_row)
    if spatial == "top_left":
        width_value = None
        height_value = None
    else:
        width_value = width
        height_value = height
    confidence = float(detection.confidence) * max(
        0.65, 1.0 - (x_residual + y_residual) / 80.0
    )
    crop_box = (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(VIEWPORT_PIXELS, int(round(x2))),
        min(VIEWPORT_PIXELS, int(round(y2))),
    )
    return (
        MapObservation(
            row=int(round((int(offset_pixels) + y1) / CELL_PIXELS)),
            column=column,
            width=width_value,
            height=height_value,
            quality=quality,
            confidence=round(confidence, 6),
            spatial=spatial,
        ),
        crop_box,
    )


def estimate_vertical_shift(
    previous: np.ndarray, current: np.ndarray
) -> tuple[int, str, float, int]:
    """Estimate downward canvas movement between two normalized viewports."""

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    previous_edges = cv2.Canny(previous_gray, 35, 95)
    current_edges = cv2.Canny(current_gray, 35, 95)
    previous_mask = (
        ((previous_hsv[:, :, 1] > 34) & (previous_hsv[:, :, 2] > 70))
        | (previous_edges > 0)
    ).astype(np.uint8) * 255
    current_mask = (
        ((current_hsv[:, :, 1] > 34) & (current_hsv[:, :, 2] > 70))
        | (current_edges > 0)
    ).astype(np.uint8) * 255
    for mask in (previous_mask, current_mask):
        mask[:7, :] = 0
        mask[-7:, :] = 0
        mask[:, :7] = 0
        mask[:, -7:] = 0

    orb = cv2.ORB_create(
        nfeatures=2600, edgeThreshold=5, patchSize=17, fastThreshold=5
    )
    keypoints_a, descriptors_a = orb.detectAndCompute(previous_gray, previous_mask)
    keypoints_b, descriptors_b = orb.detectAndCompute(current_gray, current_mask)
    shifts: list[float] = []
    orb_candidate: tuple[int, float, int] | None = None
    if descriptors_a is not None and descriptors_b is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance >= 0.76 * second.distance:
                continue
            point_a = keypoints_a[best.queryIdx].pt
            point_b = keypoints_b[best.trainIdx].pt
            dx = point_b[0] - point_a[0]
            dy = point_a[1] - point_b[1]
            if abs(dx) <= 2.5 and 3.0 <= dy <= MAX_STITCH_SHIFT_PIXELS:
                shifts.append(dy)
    if shifts:
        mode, _count = Counter(int(round(value)) for value in shifts).most_common(1)[0]
        cluster = [value for value in shifts if abs(value - mode) <= 2.5]
        if len(cluster) >= 3:
            shift = int(round(float(np.median(cluster))))
            orb_candidate = (shift, min(1.0, len(cluster) / 12.0), len(cluster))

    edge_a = cv2.GaussianBlur(previous_edges, (3, 3), 0).astype(np.float32)
    edge_b = cv2.GaussianBlur(current_edges, (3, 3), 0).astype(np.float32)
    scores: list[tuple[float, int]] = []
    for shift in range(MAX_STITCH_SHIFT_PIXELS + 1):
        left = edge_a[shift:, 8:-8]
        right = edge_b[: VIEWPORT_PIXELS - shift, 8:-8]
        if left.size == 0:
            break
        scores.append((float(np.mean(np.abs(left - right))), shift))
    if not scores:
        if orb_candidate is not None:
            shift, confidence, matches = orb_candidate
            return shift, "orb-cluster", confidence, matches
        return 0, "unresolved", 0.0, 0

    best_score, edge_shift = min(scores)
    separated = [score for score, shift in scores if abs(shift - edge_shift) > 8]
    runner_up = min(separated) if separated else best_score
    edge_margin = runner_up / max(best_score, 1e-6)
    if orb_candidate is not None:
        orb_shift, orb_confidence, matches = orb_candidate
        orb_score = next(score for score, shift in scores if shift == orb_shift)
        residual_ratio = orb_score / max(best_score, 1e-6)
        if (
            abs(orb_shift - edge_shift) > 8
            and residual_ratio >= 1.25
            and edge_margin >= 1.15
        ):
            confidence = min(
                1.0,
                0.45
                + 0.30 * min(1.0, edge_margin - 1.0)
                + 0.25 * min(1.0, residual_ratio - 1.0),
            )
            return edge_shift, "edge-global-validated", confidence, 0
        return orb_shift, "orb-cluster", orb_confidence, matches
    confidence = min(0.8, 0.25 + 0.5 * max(0.0, edge_margin - 1.0))
    return edge_shift, "edge-global", confidence, 0


class VisionRecognizer:
    def __init__(
        self,
        backend: VisionBackend,
        catalog: Sequence[CatalogItem],
    ) -> None:
        self.backend = backend
        self.catalog = {item.catalog_id: item for item in catalog}

    def recognize(
        self, frame_rgb: np.ndarray, *, offset_rows: int = 0
    ) -> tuple[MapObservation, ...]:
        viewport = normalize_map_viewport(frame_rgb)
        return self._recognize_viewport(
            viewport, offset_pixels=int(offset_rows) * CELL_PIXELS
        )

    def _recognize_viewport(
        self, viewport: np.ndarray, *, offset_pixels: int
    ) -> tuple[MapObservation, ...]:
        projected: dict[
            tuple[int, int], tuple[MapObservation, tuple[int, int, int, int]]
        ] = {}
        for detection in list(self.backend.detect_spatial(viewport)) + list(
            self.backend.detect_attributes(viewport)
        ):
            value = _project(detection, offset_pixels=int(offset_pixels))
            if value is None:
                continue
            item, crop_box = value
            previous = projected.get(item.key)
            if previous is None:
                projected[item.key] = (item, crop_box)
                continue
            kept, kept_box = previous
            if SPATIAL_RANK.get(item.spatial, 0) > SPATIAL_RANK.get(
                kept.spatial, 0
            ):
                kept.spatial = item.spatial
                kept.width = item.width
                kept.height = item.height
                kept_box = crop_box
            if item.quality and (
                not kept.quality or item.confidence >= kept.confidence
            ):
                kept.quality = item.quality
            kept.confidence = max(kept.confidence, item.confidence)
            projected[item.key] = (kept, kept_box)

        complete = [
            (key, item, box)
            for key, (item, box) in projected.items()
            if item.spatial == "complete"
            and box[2] > box[0]
            and box[3] > box[1]
        ]
        crops = [
            viewport[box[1] : box[3], box[0] : box[2]]
            for _key, _item, box in complete
        ]
        for (key, item, _box), candidates in zip(
            complete, self.backend.classify(crops)
        ):
            filtered = []
            for catalog_id, probability in candidates:
                catalog_item = self.catalog.get(str(catalog_id))
                if catalog_item is None:
                    continue
                if item.quality and catalog_item.quality != item.quality:
                    continue
                if (
                    item.width
                    and item.height
                    and (
                        catalog_item.width != item.width
                        or catalog_item.height != item.height
                    )
                ):
                    continue
                filtered.append((str(catalog_id), float(probability)))
            item.identity_candidates = tuple(filtered[:5])
            if filtered:
                item.catalog_id = filtered[0][0]
                item.confidence = min(item.confidence, filtered[0][1])
            projected[key] = (item, projected[key][1])
        return tuple(
            item for item, _box in (projected[key] for key in sorted(projected))
        )

    def recognize_viewports(
        self, frames_rgb: Sequence[np.ndarray]
    ) -> ViewportRecognition:
        if not frames_rgb:
            raise VisionError("地图扫描没有可识别视口")
        viewports = [normalize_map_viewport(frame) for frame in frames_rgb]
        offsets = [0]
        alignment_stable = True
        for previous, current in zip(viewports, viewports[1:]):
            shift, _method, confidence, _matches = estimate_vertical_shift(
                previous, current
            )
            if shift <= 0 or confidence < 0.25:
                alignment_stable = False
                shift = max(CELL_PIXELS, shift)
            offsets.append(offsets[-1] + int(shift))

        merged: dict[tuple[int, int], MapObservation] = {}
        for viewport, offset in zip(viewports, offsets):
            for item in self._recognize_viewport(
                viewport, offset_pixels=offset
            ):
                previous = merged.get(item.key)
                if previous is None or item.confidence >= previous.confidence:
                    if previous is not None and previous.first_seen_round is not None:
                        item.first_seen_round = previous.first_seen_round
                    merged[item.key] = item
        estimated_rows = max(
            10,
            int(round((offsets[-1] + VIEWPORT_PIXELS) / CELL_PIXELS)),
        )
        return ViewportRecognition(
            items=tuple(merged[key] for key in sorted(merged)),
            offsets_pixels=tuple(offsets),
            estimated_rows=estimated_rows,
            alignment_stable=alignment_stable,
        )
