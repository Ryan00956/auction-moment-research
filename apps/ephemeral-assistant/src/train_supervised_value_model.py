from __future__ import annotations

"""Minimal decision-time feature builder required by the released v6 model.

Training, dataset loading, settlement labels, and private repository paths are
intentionally excluded from the public runtime.
"""

import math
import statistics
from collections import Counter

from structural_value_features import structural_value_features


FEATURE_SCHEMA_VERSION = "decision-time-value-features-v2-structural"


def _safe_number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def decision_time_features(decision: dict) -> dict[str, float]:
    features: dict[str, float] = {
        "round": float(decision.get("round") or 0),
    }
    observed_map = decision.get("observed_map") or {}
    for key in (
        "columns",
        "rows",
        "mosaic_height_pixels",
        "partial_row_pixels",
        "viewport_count",
    ):
        numeric = _safe_number(observed_map.get(key))
        if numeric is not None:
            features[f"map.{key}"] = numeric
    for key in ("height_exact", "bottom_visible", "canvas_bottom_visible"):
        features[f"map.{key}"] = float(bool(observed_map.get(key)))

    effect_counts: Counter[str] = Counter()
    for event in decision.get("events") or []:
        semantics = event.get("semantics") or {}
        if semantics.get("parsed") is not True:
            continue
        effect = str(semantics.get("effect") or "unknown")
        quality = str(
            semantics.get("quality")
            or semantics.get("observed_quality")
            or "all"
        )
        width = semantics.get("width")
        height = semantics.get("height")
        size = (
            f"{int(width)}x{int(height)}"
            if _safe_number(width) is not None
            and _safe_number(height) is not None
            else "all"
        )
        scope = f"{effect}|q={quality}|s={size}"
        effect_counts[effect] += 1
        features[f"event.present.{scope}"] = 1.0
        for field in (
            "observed_count",
            "observed_average_cells",
            "observed_cells",
            "observed_value",
            "count",
            "width",
            "height",
        ):
            numeric = _safe_number(semantics.get(field))
            if numeric is not None:
                features[f"event.{scope}.{field}"] = numeric
    for effect, count in effect_counts.items():
        features[f"event.count.{effect}"] = float(count)

    visible = list(decision.get("visible_items") or [])
    features["visible.count"] = float(len(visible))
    confidence_values = []
    known_value_sum = 0.0
    known_area_sum = 0.0
    for item in visible:
        quality = str(item.get("quality") or "unknown")
        spatial = str(item.get("spatial_knowledge") or "unknown")
        features[f"visible.quality.{quality}"] = (
            features.get(f"visible.quality.{quality}", 0.0) + 1.0
        )
        features[f"visible.spatial.{spatial}"] = (
            features.get(f"visible.spatial.{spatial}", 0.0) + 1.0
        )
        confidence = _safe_number(item.get("evidence_confidence"))
        if confidence is not None:
            confidence_values.append(confidence)
            features[f"visible.confidence_weighted_quality.{quality}"] = (
                features.get(
                    f"visible.confidence_weighted_quality.{quality}",
                    0.0,
                )
                + confidence
            )
        known_size = item.get("known_size") or {}
        width = _safe_number(known_size.get("width"))
        height = _safe_number(known_size.get("height"))
        if width is not None and height is not None:
            area = width * height
            known_area_sum += area
            features[f"visible.known_size.{int(width)}x{int(height)}"] = (
                features.get(
                    f"visible.known_size.{int(width)}x{int(height)}",
                    0.0,
                )
                + 1.0
            )
        identity = item.get("identity") or {}
        known_value = _safe_number(identity.get("known_value"))
        if known_value is not None:
            known_value_sum += known_value
    features["visible.known_area_sum"] = known_area_sum
    features["visible.known_value_sum"] = known_value_sum
    features["visible.identity_known_count"] = float(
        sum(bool(item.get("identity_known")) for item in visible)
    )
    if confidence_values:
        features["visible.confidence_mean"] = float(
            statistics.fmean(confidence_values)
        )
        features["visible.confidence_min"] = min(confidence_values)

    recognition = decision.get("map_recognition") or {}
    completeness = recognition.get("completeness") or {}
    features[
        f"recognition.status.{completeness.get('status') or 'missing'}"
    ] = 1.0
    features["recognition.exact_counts_safe"] = float(
        bool(completeness.get("exact_counts_safe"))
    )
    for key, value in (completeness.get("checks") or {}).items():
        features[f"recognition.check.{key}"] = float(bool(value))
    for key, value in (
        completeness.get("exact_quality_counts") or {}
    ).items():
        numeric = _safe_number(value)
        if numeric is not None:
            features[f"recognition.exact_quality.{key}"] = numeric
    for key, value in (
        completeness.get("exact_size_counts") or {}
    ).items():
        numeric = _safe_number(value)
        if numeric is not None:
            features[f"recognition.exact_size.{key}"] = numeric
    features.update(structural_value_features(decision))
    return features
