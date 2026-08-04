"""Leak-free, mechanism-informed value features for small-data research.

The reference video suggests that non-coloured value is largely explained by
gold/purple counts, their quality subtotals, or total occupied cells.  The raw
numbers were already present in the generic supervised feature map, but a
small dataset should not be expected to rediscover all unit conversions and
interactions from scratch.  This module exposes the video formulas and the
older 33-game local fit as explicit *candidate features*.  None of them is a
production estimate by itself, and no settlement/final-map label is read.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Mapping


QUALITY_ALIASES = {
    "金": "金",
    "金色": "金",
    "紫": "紫",
    "紫色": "紫",
    "彩": "彩",
    "彩色": "彩",
}

# Values are in the game's integer currency unit, not the video's 万 unit.
VIDEO_XY_GOLD_COEFFICIENT = 160_000.0
VIDEO_XY_PURPLE_COEFFICIENT = 70_000.0
LOCAL33_XY_GOLD_COEFFICIENT = 184_300.0
LOCAL33_XY_PURPLE_COEFFICIENT = 78_100.0
VIDEO_GOLD_COUNT_COEFFICIENT = 240_000.0
VIDEO_PURPLE_COUNT_COEFFICIENT = 320_000.0
VIDEO_GOLD_TOTAL_MULTIPLIER = 1.5
VIDEO_PURPLE_TOTAL_MULTIPLIER = 2.5
VIDEO_OCCUPIED_CELL_COEFFICIENT = 24_000.0
VIDEO_COLOURED_ITEM_MEAN = 1_060_000.0


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _quality(value: object) -> str | None:
    return QUALITY_ALIASES.get(str(value or "").strip())


def _event_confidence(event: Mapping) -> float:
    value = _number(event.get("ocr_confidence"))
    if value is None:
        return 0.5
    return min(1.0, max(0.0, value))


def _first_number(
    source: Mapping,
    keys: Iterable[str],
) -> float | None:
    for key in keys:
        value = _number(source.get(key))
        if value is not None:
            return value
    return None


def _select_measurement(
    values: list[tuple[float, float, int]],
) -> tuple[float, float, float, int] | None:
    """Return value, confidence, and disagreement for one measurement.

    A certified map count has a higher source rank than OCR.  Within the same
    rank, the highest confidence wins and the latest cumulative event breaks
    ties.  Disagreement is retained as a feature instead of being hidden.
    """

    if not values:
        return None
    selected = max(
        enumerate(values),
        key=lambda pair: (
            pair[1][2],
            pair[1][1],
            pair[0],
        ),
    )[1]
    numeric = [value for value, _, _ in values]
    disagreement = max(numeric) - min(numeric)
    return selected[0], selected[1], disagreement, selected[2]


def _measurement_features(
    features: dict[str, float],
    name: str,
    values: list[tuple[float, float, int]],
) -> float | None:
    selected = _select_measurement(values)
    if selected is None:
        features[f"structural.has.{name}"] = 0.0
        return None
    value, confidence, disagreement, source_rank = selected
    features[f"structural.has.{name}"] = 1.0
    features[f"structural.{name}"] = value
    features[f"structural.{name}.confidence"] = confidence
    features[f"structural.{name}.source_count"] = float(len(values))
    features[f"structural.{name}.source_rank"] = float(source_rank)
    features[f"structural.{name}.disagreement"] = disagreement
    return value


def _known_coloured_values(decision: Mapping) -> tuple[float, int]:
    total = 0.0
    count = 0
    for item in decision.get("visible_items") or []:
        if _quality(item.get("quality")) != "彩":
            continue
        identity = item.get("identity") or {}
        value = _number(identity.get("known_value"))
        if value is None:
            continue
        total += value
        count += 1
    return total, count


def structural_value_features(
    decision: Mapping,
) -> dict[str, float]:
    """Build decision-time structural proxies without reading outcome truth."""

    measurements: defaultdict[
        str, list[tuple[float, float, int]]
    ] = defaultdict(list)
    reveal_all_qualities = set()
    for event in decision.get("events") or []:
        semantics = event.get("semantics") or {}
        if semantics.get("parsed") is not True:
            continue
        effect = str(semantics.get("effect") or "")
        quality = _quality(
            semantics.get("quality")
            or semantics.get("observed_quality")
        )
        confidence = _event_confidence(event)
        if (
            effect
            in {
                "reveal_position_by_quality",
                "reveal_outline_by_quality",
            }
            and quality
        ):
            reveal_all_qualities.add(quality)
        if effect == "quality_item_count" and quality:
            value = _first_number(
                semantics,
                ("observed_count", "count"),
            )
            if value is not None:
                measurements[f"{quality}.count"].append(
                    (value, confidence, 3)
                )
        elif effect == "quality_total_value" and quality:
            value = _first_number(
                semantics,
                ("observed_value", "value"),
            )
            if value is not None:
                measurements[f"{quality}.total_value"].append(
                    (value, confidence, 3)
                )
        elif effect == "total_occupied_cells":
            value = _first_number(
                semantics,
                ("observed_cells", "cells"),
            )
            if value is not None:
                measurements["occupied_cells"].append(
                    (value, confidence, 3)
                )

    recognition = decision.get("map_recognition") or {}
    completeness = recognition.get("completeness") or {}
    visible_items = list(decision.get("visible_items") or [])
    for quality in reveal_all_qualities:
        matching = [
            item
            for item in visible_items
            if _quality(item.get("quality")) == quality
        ]
        confidences = [
            float(
                _number(item.get("quality_confidence"))
                or _number(item.get("evidence_confidence"))
                or 0.0
            )
            for item in matching
        ]
        confidence = min(confidences) if confidences else 0.0
        measurements[f"{quality}.count"].append(
            (float(len(matching)), confidence, 1)
        )
        features_name = {
            "金": "gold",
            "紫": "purple",
            "彩": "coloured",
        }.get(quality)
        if features_name:
            # This is a calibrated soft estimate, not an absence proof.
            # Full-data audit found 87-93% exact group counts by quality.
            # Keep it explicit so downstream models can widen intervals.
            measurements[
                f"{quality}.map_reveal_count_marker"
            ].append((float(len(matching)), confidence, 1))
    for quality, value in (
        completeness.get("exact_quality_counts") or {}
    ).items():
        normalized = _quality(quality)
        numeric = _number(value)
        if normalized and numeric is not None:
            # Current recognizer certificates are 10/11 correct in the
            # independent audit, so they remain below exact event OCR.
            measurements[f"{normalized}.count"].append(
                (numeric, 1.0, 2)
            )

    features: dict[str, float] = {}
    for quality, prefix in (
        ("金", "gold"),
        ("紫", "purple"),
        ("彩", "coloured"),
    ):
        marker = _select_measurement(
            measurements[f"{quality}.map_reveal_count_marker"]
        )
        features[
            f"structural.map_reveal.{prefix}_count.available"
        ] = float(marker is not None)
        if marker is not None:
            features[
                f"structural.map_reveal.{prefix}_count"
            ] = marker[0]
            features[
                f"structural.map_reveal.{prefix}_count.confidence"
            ] = marker[1]
    gold_count = _measurement_features(
        features,
        "gold_count",
        measurements["金.count"],
    )
    purple_count = _measurement_features(
        features,
        "purple_count",
        measurements["紫.count"],
    )
    coloured_count = _measurement_features(
        features,
        "coloured_count",
        measurements["彩.count"],
    )
    gold_total = _measurement_features(
        features,
        "gold_total_value",
        measurements["金.total_value"],
    )
    purple_total = _measurement_features(
        features,
        "purple_total_value",
        measurements["紫.total_value"],
    )
    coloured_total = _measurement_features(
        features,
        "coloured_total_value",
        measurements["彩.total_value"],
    )
    occupied_cells = _measurement_features(
        features,
        "occupied_cells",
        measurements["occupied_cells"],
    )

    proxies: dict[str, float] = {}
    if gold_count is not None:
        proxies["gold_count_video"] = (
            VIDEO_GOLD_COUNT_COEFFICIENT * gold_count
        )
    if purple_count is not None:
        proxies["purple_count_video"] = (
            VIDEO_PURPLE_COUNT_COEFFICIENT * purple_count
        )
    if gold_count is not None and purple_count is not None:
        proxies["gold_purple_video"] = (
            VIDEO_XY_GOLD_COEFFICIENT * gold_count
            + VIDEO_XY_PURPLE_COEFFICIENT * purple_count
        )
        proxies["gold_purple_local33"] = (
            LOCAL33_XY_GOLD_COEFFICIENT * gold_count
            + LOCAL33_XY_PURPLE_COEFFICIENT * purple_count
        )
    if gold_total is not None:
        proxies["gold_total_video"] = (
            VIDEO_GOLD_TOTAL_MULTIPLIER * gold_total
        )
    if purple_total is not None:
        proxies["purple_total_video"] = (
            VIDEO_PURPLE_TOTAL_MULTIPLIER * purple_total
        )
    if occupied_cells is not None:
        proxies["occupied_cells_video"] = (
            VIDEO_OCCUPIED_CELL_COEFFICIENT * occupied_cells
        )

    for name, value in proxies.items():
        features[f"structural.proxy.non_coloured.{name}"] = value
    features["structural.proxy.non_coloured.count"] = float(
        len(proxies)
    )
    if proxies:
        values = list(proxies.values())
        median = float(statistics.median(values))
        features["structural.proxy.non_coloured.median"] = median
        features["structural.proxy.non_coloured.mean"] = float(
            statistics.fmean(values)
        )
        features["structural.proxy.non_coloured.range"] = (
            max(values) - min(values)
        )
        features["structural.proxy.non_coloured.relative_range"] = (
            (max(values) - min(values)) / max(1.0, abs(median))
        )

    known_coloured, known_coloured_count = _known_coloured_values(
        decision
    )
    features["structural.visible_coloured_known_value"] = (
        known_coloured
    )
    features["structural.visible_coloured_known_count"] = float(
        known_coloured_count
    )
    if coloured_total is not None:
        coloured_component = coloured_total
        coloured_status = "exact_total"
    elif (
        coloured_count is not None
        and coloured_count >= known_coloured_count
    ):
        coloured_component = max(
            known_coloured,
            coloured_count * VIDEO_COLOURED_ITEM_MEAN,
        )
        coloured_status = "count_expectation"
    else:
        coloured_component = known_coloured
        coloured_status = "visible_lower_bound"
    for status in (
        "exact_total",
        "count_expectation",
        "visible_lower_bound",
    ):
        features[f"structural.coloured_component.{status}"] = float(
            status == coloured_status
        )
    features["structural.coloured_component.value"] = (
        coloured_component
    )
    if proxies:
        for name, value in proxies.items():
            features[f"structural.proxy.total.{name}"] = (
                value + coloured_component
            )
        features["structural.proxy.total.median"] = (
            statistics.median(proxies.values()) + coloured_component
        )
    return features
