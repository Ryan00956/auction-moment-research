from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .models import CatalogItem
from .observations import ObservationSnapshot


QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
INFORMATIONAL_EFFECTS = {
    "reveal_identity_random",
    "reveal_position_quality_random",
    "reveal_outline_by_size",
    "reveal_position_random",
    "reveal_largest_area_item_outline",
    "reveal_outline_by_quality",
    "reveal_position_by_quality",
}


class PredictionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    revision: int
    status: str
    compatible_worlds: int
    p10: int | None
    p50: int | None
    p90: int | None
    minimum: int | None
    maximum: int | None
    actionable: bool
    issues: tuple[str, ...]
    contract: str = "empirical_compatible_worlds_uncalibrated"
    v6_prediction: int | None = None
    v2_prediction: int | None = None
    generation_weight: float | None = None
    model_version: str | None = None


class EmpiricalWorldPredictor:
    """Filter frozen/public-prior worlds by decision-time evidence.

    Settlement totals are not read. Every world value is recomputed from the
    public catalog identity counts. Returned quantiles describe compatible
    frozen worlds and are intentionally never marked actionable/calibrated.
    """

    def __init__(
        self,
        treasures_csv: Path,
        catalog: Sequence[CatalogItem],
        *,
        world_model_path: Path | None = None,
        prior_samples: int = 2048,
    ) -> None:
        self.catalog = tuple(catalog)
        self.index_by_id = {
            item.catalog_id: index for index, item in enumerate(self.catalog)
        }
        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        with Path(treasures_csv).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                session_id = str(row.get("session_id") or "")
                catalog_id = str(row.get("catalog_id") or "")
                if session_id and catalog_id in self.index_by_id:
                    counts[session_id][catalog_id] += 1
        if not counts:
            raise PredictionError("公开世界样本为空")
        empirical_counts = np.asarray(
            [
                [row[item.catalog_id] for item in self.catalog]
                for _session, row in sorted(counts.items())
            ],
            dtype=np.int16,
        )
        self.empirical_world_count = len(empirical_counts)
        prior_counts = np.empty((0, len(self.catalog)), dtype=np.int16)
        self.world_model_version = "none"
        if world_model_path is not None:
            try:
                payload = json.loads(
                    Path(world_model_path).read_text(encoding="utf-8")
                )
                schema = str(payload.get("schema_version") or "")
                if schema == "world-model-v2":
                    from generative_world_model_v2 import GenerativeWorldModelV2

                    model = GenerativeWorldModelV2.load(
                        Path(world_model_path), self.catalog
                    )
                    self.world_model_version = "world_model_v2"
                else:
                    from auction_moment_research.world_model import (
                        ProbabilisticWorldModel,
                    )

                    model = ProbabilisticWorldModel.load(
                        Path(world_model_path), self.catalog
                    )
                    self.world_model_version = (
                        schema.replace("-", "_") if schema else "world_model_v1"
                    )
            except Exception as exc:
                raise PredictionError(f"世界模型加载失败：{exc}") from exc
            rng = random.Random(20260804)
            prior_counts = np.asarray(
                [
                    model.sample_prior_counts(rng)
                    for _ in range(max(0, int(prior_samples)))
                ],
                dtype=np.int16,
            )
        self.prior_world_count = len(prior_counts)
        self.world_counts = (
            np.concatenate([empirical_counts, prior_counts], axis=0)
            if len(prior_counts)
            else empirical_counts
        )
        self.values = np.asarray([item.value for item in self.catalog], dtype=np.int64)
        self.areas = np.asarray(
            [item.width * item.height for item in self.catalog], dtype=np.int16
        )
        self.total_counts = self.world_counts.sum(axis=1, dtype=np.int64)
        self.total_values = self.world_counts @ self.values
        self.total_areas = self.world_counts @ self.areas.astype(np.int64)
        self.quality_masks = {
            quality: np.asarray(
                [item.quality == quality for item in self.catalog], dtype=bool
            )
            for quality in QUALITY_ORDER
        }
        self.size_masks = {
            (width, height): np.asarray(
                [
                    item.width == width and item.height == height
                    for item in self.catalog
                ],
                dtype=bool,
            )
            for width in range(1, 4)
            for height in range(1, 4)
        }

    def _aggregate(self, mask: np.ndarray):
        selected = self.world_counts[:, mask].astype(np.int64)
        return (
            selected.sum(axis=1),
            selected @ self.values[mask],
            selected @ self.areas[mask].astype(np.int64),
        )

    @staticmethod
    def _floor_average(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        return np.divide(
            total,
            count,
            out=np.full(total.shape, -1, dtype=np.float64),
            where=count > 0,
        ).astype(np.int64)

    @staticmethod
    def _floor_area_average_hundredths(
        total: np.ndarray, count: np.ndarray
    ) -> np.ndarray:
        numerator = total * 100
        return np.divide(
            numerator,
            count,
            out=np.full(total.shape, -1, dtype=np.float64),
            where=count > 0,
        ).astype(np.int64)

    def _apply_event(
        self,
        mask: np.ndarray,
        semantics: Mapping,
    ) -> np.ndarray:
        effect = str(semantics.get("effect") or "unparsed")
        if effect in INFORMATIONAL_EFFECTS:
            return mask
        quality = str(semantics.get("quality") or "")
        width = int(semantics.get("width") or 0)
        height = int(semantics.get("height") or 0)
        if quality:
            group_mask = self.quality_masks.get(quality)
        elif width and height:
            group_mask = self.size_masks.get((width, height))
        else:
            group_mask = None
        if group_mask is not None:
            count, value, area = self._aggregate(group_mask)
        else:
            count, value, area = (
                self.total_counts,
                self.total_values,
                self.total_areas,
            )
        if effect in {"quality_item_count", "size_item_count", "total_item_count"}:
            return mask & (count == int(semantics["observed_count"]))
        if effect in {"quality_total_value", "size_total_value"}:
            return mask & (value == int(semantics["observed_value"]))
        if effect in {"quality_average_value", "size_average_value"}:
            observed = int(semantics["observed_value"])
            return mask & (self._floor_average(value, count) == observed)
        if effect in {
            "quality_total_area",
            "size_total_area",
            "total_occupied_cells",
        }:
            return mask & (area == int(semantics["observed_cells"]))
        if effect in {
            "quality_average_area",
            "size_average_area",
            "average_item_area",
        }:
            observed = int(
                math.floor(float(semantics["observed_average_cells"]) * 100 + 1e-9)
            )
            return mask & (
                self._floor_area_average_hundredths(area, count) == observed
            )
        if effect == "max_item_value":
            present = self.world_counts > 0
            maximum = np.where(present, self.values, -1).max(axis=1)
            return mask & (maximum == int(semantics["observed_value"]))
        if effect == "max_value_per_cell":
            per_cell = self.values // self.areas
            present = self.world_counts > 0
            maximum = np.where(present, per_cell, -1).max(axis=1)
            return mask & (maximum == int(semantics["observed_value"]))
        if effect == "highest_quality":
            observed_quality = str(semantics["observed_quality"])
            observed_index = QUALITY_ORDER.index(observed_quality)
            quality_present = np.column_stack(
                [
                    self.world_counts[:, quality_mask].sum(axis=1) > 0
                    for quality_mask in self.quality_masks.values()
                ]
            )
            highest = np.where(
                quality_present,
                np.arange(len(QUALITY_ORDER)),
                -1,
            ).max(axis=1)
            return mask & (highest == observed_index)
        return mask

    def _apply_visible_items(
        self, mask: np.ndarray, items: Iterable[Mapping]
    ) -> np.ndarray:
        identity_minimum: Counter[str] = Counter()
        quality_minimum: Counter[str] = Counter()
        size_minimum: Counter[tuple[int, int]] = Counter()
        joint_minimum: Counter[tuple[str, int, int]] = Counter()
        for item in items:
            human_locked = bool(item.get("human_locked"))
            confidence = float(item.get("confidence") or 0.0)
            if not human_locked and confidence < 0.85:
                continue
            catalog_id = (
                str(item.get("catalog_id") or "") if human_locked else ""
            )
            quality = str(item.get("quality") or "")
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            if catalog_id:
                identity_minimum[catalog_id] += 1
            if quality:
                quality_minimum[quality] += 1
            if width and height:
                size_minimum[(width, height)] += 1
            if quality and width and height:
                joint_minimum[(quality, width, height)] += 1
        for catalog_id, minimum in identity_minimum.items():
            index = self.index_by_id.get(catalog_id)
            if index is None:
                return np.zeros_like(mask)
            mask &= self.world_counts[:, index] >= minimum
        for quality, minimum in quality_minimum.items():
            group = self.quality_masks.get(quality)
            if group is None:
                return np.zeros_like(mask)
            mask &= self.world_counts[:, group].sum(axis=1) >= minimum
        for size, minimum in size_minimum.items():
            group = self.size_masks.get(size)
            if group is None:
                return np.zeros_like(mask)
            mask &= self.world_counts[:, group].sum(axis=1) >= minimum
        for (quality, width, height), minimum in joint_minimum.items():
            group = self.quality_masks[quality] & self.size_masks[(width, height)]
            mask &= self.world_counts[:, group].sum(axis=1) >= minimum
        return mask

    def predict(self, snapshot: ObservationSnapshot) -> PredictionResult:
        mask = np.ones(len(self.world_counts), dtype=bool)
        issues = []
        current_events = [
            event
            for event in snapshot.events
            if int(event["round_number"]) <= snapshot.round_number
        ]
        if len(current_events) < snapshot.round_number * 2:
            issues.append("missing_event")
        for event in current_events:
            semantics = event.get("semantics") or {}
            if not semantics.get("parsed"):
                issues.append(
                    f"unparsed_event:R{event['round_number']}:{event['kind']}"
                )
                continue
            mask = self._apply_event(mask, semantics)
        mask = self._apply_visible_items(mask, snapshot.map_items)
        compatible = self.total_values[mask]
        if compatible.size == 0:
            status = "constraint_conflict"
            issues.append("no_compatible_public_world")
            values = (None, None, None, None, None)
        else:
            quantiles = np.quantile(
                compatible, [0.10, 0.50, 0.90], method="nearest"
            )
            values = (
                int(quantiles[0]),
                int(quantiles[1]),
                int(quantiles[2]),
                int(compatible.min()),
                int(compatible.max()),
            )
            if issues:
                status = "provisional_event_ocr"
            elif not snapshot.completeness_confirmed:
                status = "provisional_map_review"
                issues.append("map_completeness_not_confirmed")
            elif compatible.size < 10:
                status = "provisional_low_support"
                issues.append("compatible_worlds_below_10")
            else:
                status = "ready_research_estimate"
        return PredictionResult(
            revision=snapshot.revision,
            status=status,
            compatible_worlds=int(compatible.size),
            p10=values[0],
            p50=values[1],
            p90=values[2],
            minimum=values[3],
            maximum=values[4],
            actionable=False,
            issues=tuple(dict.fromkeys(issues)),
            contract=(
                f"public_empirical_plus_world_model_prior_{self.world_model_version}_uncalibrated"
                if self.prior_world_count
                else "empirical_compatible_worlds_uncalibrated"
            ),
        )
