"""Stable decision-time feature plane for the released v6 model.

No settlement, final-map annotation, bid, ranking, or post-bid field is
accepted. The public runtime accepts either the original structured pre-bid
contract or the explicit OCR/manual adapter contract. Missing fields always
mean unknown, and the frozen base model contributes diagnostics rather than
labels.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Mapping

from train_supervised_value_model import decision_time_features


SCHEMA_VERSION = "protocol-packet-residual-features-v1"
QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")


class ProtocolPacketResidualFeatureError(RuntimeError):
    """Raised when a feature input violates the live knowledge contract."""


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _quality(item: Mapping) -> str:
    return str(item.get("quality") or "unknown")


def _known_value(item: Mapping) -> float | None:
    identity = item.get("identity") or {}
    if not isinstance(identity, Mapping):
        return None
    return _number(identity.get("known_value"))


def packet_residual_features(
    decision: Mapping,
    base: Mapping,
) -> dict[str, float]:
    """Return finite pre-bid features plus frozen-base diagnostics."""

    if any(
        forbidden in decision
        for forbidden in (
            "actual",
            "label",
            "settlement",
            "settlement_total",
            "decision_outcome",
        )
    ):
        raise ProtocolPacketResidualFeatureError(
            "feature decision contains a forbidden label field"
        )
    native = decision.get("native_evidence") or {}
    mode = str(native.get("mode") or "") if isinstance(native, Mapping) else ""
    expected_ocr = mode == "ocr_visible_pre_bid_v1"
    if (
        not isinstance(native, Mapping)
        or mode not in {
            "packet_capture_pre_bid_exact",
            "ocr_visible_pre_bid_v1",
        }
        or native.get("pre_bid_proven") is not True
        or native.get("ocr_used_for_live_features") is not expected_ocr
        or native.get("settlement_or_final_truth_in_features") is not False
    ):
        raise ProtocolPacketResidualFeatureError(
            "decision is not proven pre-bid packet or OCR-visible evidence"
        )

    result = decision_time_features(dict(decision))
    visible = [
        item
        for item in decision.get("visible_items") or []
        if isinstance(item, Mapping)
    ]
    indices = sorted(
        {
            int(item["item_index"])
            for item in visible
            if item.get("item_index") is not None
            and int(item["item_index"]) > 0
        }
    )
    map_rows = float(
        (decision.get("observed_map") or {}).get("rows") or 10
    )
    round_number = float(decision.get("round") or 0)
    result["packet.index.unique_count"] = float(len(indices))
    result["packet.index.max"] = float(max(indices, default=0))
    result["packet.index.min"] = float(min(indices, default=0))
    result["packet.index.mean"] = (
        float(statistics.fmean(indices)) if indices else 0.0
    )
    result["packet.index.median"] = (
        float(statistics.median(indices)) if indices else 0.0
    )
    result["packet.index.max_per_map_row"] = (
        float(max(indices, default=0)) / max(1.0, map_rows)
    )
    result["packet.index.observed_fraction_through_max"] = (
        len(indices) / max(1.0, float(max(indices, default=0)))
    )
    result["packet.index.uniform_order_stat_nhat"] = (
        min(
            60.0,
            max(indices) * (len(indices) + 1.0) / len(indices) - 1.0,
        )
        if indices
        else 0.0
    )

    positions: list[tuple[int, int]] = []
    known_areas = []
    first_seen: Counter[int] = Counter()
    source_effects: Counter[str] = Counter()
    quality_known_values: Counter[str] = Counter()
    quality_known_counts: Counter[str] = Counter()
    all_known_values = []
    for item in visible:
        position = item.get("position") or {}
        if isinstance(position, Mapping):
            try:
                row = int(position["row"])
                column = int(position["column"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                positions.append((row, column))
        size = item.get("known_size") or {}
        if isinstance(size, Mapping):
            try:
                width = int(size["width"])
                height = int(size["height"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if width > 0 and height > 0:
                    known_areas.append(width * height)
        try:
            seen_round = int(item.get("first_seen_round") or 0)
        except (TypeError, ValueError):
            seen_round = 0
        if seen_round > 0:
            first_seen[seen_round] += 1
        for history in item.get("history") or []:
            if isinstance(history, Mapping):
                source = str(history.get("source_effect") or "")
                if source:
                    source_effects[source] += 1
        known_value = _known_value(item)
        if known_value is not None and known_value > 0:
            quality = _quality(item)
            all_known_values.append(known_value)
            quality_known_values[quality] += known_value
            quality_known_counts[quality] += 1

    result["packet.position.count"] = float(len(positions))
    result["packet.position.unique_rows"] = float(
        len({row for row, _ in positions})
    )
    result["packet.position.max_row"] = float(
        max((row for row, _ in positions), default=-1) + 1
    )
    result["packet.position.max_row_fraction"] = (
        result["packet.position.max_row"] / max(1.0, map_rows)
    )
    result["packet.position.row_span"] = float(
        (
            max(row for row, _ in positions)
            - min(row for row, _ in positions)
            + 1
        )
        if positions
        else 0
    )
    result["packet.footprint.known_count"] = float(len(known_areas))
    result["packet.footprint.area_sum"] = float(sum(known_areas))
    result["packet.footprint.area_mean"] = (
        float(statistics.fmean(known_areas)) if known_areas else 0.0
    )
    for seen_round, count in first_seen.items():
        result[f"packet.first_seen_round.{seen_round}"] = float(count)
    for source, count in source_effects.items():
        result[f"packet.source_effect.{source}"] = float(count)
    for quality in QUALITY_ORDER:
        result[f"packet.known_identity_count.{quality}"] = float(
            quality_known_counts[quality]
        )
        result[f"packet.known_identity_value_millions.{quality}"] = (
            float(quality_known_values[quality]) / 1_000_000.0
        )
    result["packet.known_identity_value.max_millions"] = (
        max(all_known_values, default=0.0) / 1_000_000.0
    )
    result["packet.known_identity_value.mean_millions"] = (
        statistics.fmean(all_known_values) / 1_000_000.0
        if all_known_values
        else 0.0
    )

    prediction = float(base["prediction"])
    p10 = float(base.get("p10") or prediction)
    p90 = float(base.get("p90") or prediction)
    result.update(
        {
            "base.prediction_millions": prediction / 1_000_000.0,
            "base.log_prediction": math.log(max(1.0, prediction)),
            "base.p10_millions": p10 / 1_000_000.0,
            "base.p90_millions": p90 / 1_000_000.0,
            "base.relative_interval_width": float(
                base.get("relative_interval_width") or 0.0
            ),
            "base.posterior_p95_ape": float(
                base.get("posterior_p95_absolute_percentage_error")
                or 0.0
            ),
            "base.selective_risk": float(
                (base.get("selective_risk") or {}).get("score") or 0.0
            ),
            "base.effective_sample_size": float(
                base.get("effective_sample_size") or 0.0
            ),
            "base.normalized_weight_entropy": float(
                base.get("normalized_weight_entropy") or 0.0
            ),
            "base.maximum_world_weight": float(
                base.get("maximum_world_weight") or 0.0
            ),
            "base.known_value_lower_bound_millions": float(
                base.get("known_value_lower_bound") or 0.0
            )
            / 1_000_000.0,
            "base.evidence_delta_millions": float(
                base.get("evidence_delta") or 0.0
            )
            / 1_000_000.0,
            "base.purple_area_delta_millions": float(
                base.get("purple_area_prediction_delta") or 0.0
            )
            / 1_000_000.0,
            "base.round_fraction": round_number / 5.0,
        }
    )
    lower_bound = float(base.get("known_value_lower_bound") or 0.0)
    result["base.lower_bound_fraction"] = lower_bound / max(
        1.0, prediction
    )
    result["packet.visible_fraction_of_row_count_proxy"] = len(
        indices
    ) / max(1.0, 2.75 * map_rows)

    invalid = [
        name
        for name, value in result.items()
        if not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ]
    if invalid:
        raise ProtocolPacketResidualFeatureError(
            "non-finite or non-numeric features: "
            + ", ".join(sorted(invalid)[:10])
        )
    return {name: float(value) for name, value in result.items()}


def event_structural_base_profile(
    features: Mapping[str, float],
) -> dict[str, float]:
    """Retain the preselected 208-feature v6 profile."""

    allowed = {"base", "event", "map", "round", "structural"}
    result = {
        name: float(value)
        for name, value in features.items()
        if name.split(".", 1)[0] in allowed
    }
    if not result:
        raise ProtocolPacketResidualFeatureError(
            "event/structural/base feature profile is empty"
        )
    return result
