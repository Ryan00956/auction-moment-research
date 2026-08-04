"""Point-only augmentation from an exact purple average-area packet event.

The guarded v3 model already combines the strongest packet evidence that
survived chronological development checks.  This module adds one deliberately
narrow mechanism hypothesis:

* ``quality_average_area`` for purple treasures is an exact, immutable
  aggregate reported before bidding;
* the same aggregate can be computed from the old reviewed training worlds
  without reading any current settlement;
* a broad Gaussian match over that aggregate reweights the existing v3
  analogue posterior; and
* only half of the posterior-median movement is applied to the point estimate.

No OCR, bid, ranking, current settlement, or final map enters prediction.  The
candidate is point-only: its interval and selective-risk fields are inherited
from v3 and explicitly marked as not revalidated for this augmentation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from protocol_additive_value_model import (
    canonical_sha256,
    sha256_file,
)
from protocol_analog_value_model import (
    _contains_forbidden_label_key,
    _weighted_quantile,
)
from protocol_position_quality_augment_model import (
    ProtocolPositionQualityAugmentModel,
)


MODEL_SCHEMA_VERSION = "protocol-purple-area-augment-v4-experimental"
MODEL_CONFIG = {
    "accepted_effect": "quality_average_area",
    "accepted_quality": "紫",
    "measurement_field": "observed_average_cells",
    "world_measurement_rounding": "floor_to_hundredths",
    "gaussian_sigma_cells": 1.0,
    "point_delta_scale": 0.5,
    "maximum_used_events": 1,
    "selection_rule": "first_chronological_exact_purple_event",
    "duplicate_measurements_must_agree": True,
    "minimum_weight_sum": 1e-12,
    "uncertainty_contract": "inherited_v3_shifted_not_revalidated",
}


class ProtocolPurpleAreaAugmentError(RuntimeError):
    """Raised when the v4 point-only evidence contract is invalid."""


@dataclass(frozen=True)
class PurpleAreaReference:
    """Purple average-area values aligned one-to-one with v3 worlds."""

    session_ids: tuple[str, ...]
    average_areas: tuple[float, ...]
    provenance: dict

    def __post_init__(self) -> None:
        if (
            not self.session_ids
            or len(self.session_ids) != len(self.average_areas)
            or len(self.session_ids) != len(set(self.session_ids))
            or any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in self.average_areas
            )
        ):
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格参考为空、未对齐、重复或含非法值"
            )


def _purple_average_area(record: Mapping) -> float:
    annotations = [
        item
        for item in record.get("annotations") or []
        if isinstance(item, Mapping)
        and str(item.get("quality") or "") == "紫"
    ]
    if not annotations:
        raise ProtocolPurpleAreaAugmentError(
            f"历史对局缺少紫色宝藏：{record.get('session_id')}"
        )
    try:
        areas = [
            int(item["width"]) * int(item["height"])
            for item in annotations
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolPurpleAreaAugmentError(
            f"历史紫色宝藏几何无效：{record.get('session_id')}"
        ) from exc
    if any(area <= 0 for area in areas):
        raise ProtocolPurpleAreaAugmentError(
            f"历史紫色宝藏面积非正：{record.get('session_id')}"
        )
    return float(100 * sum(areas) // len(areas)) / 100.0


def load_purple_area_reference(
    final_annotations_path: Path,
    *,
    expected_session_ids: Sequence[str],
) -> PurpleAreaReference:
    """Load only purple geometry and align it to the frozen v3 worlds."""

    path = Path(final_annotations_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolPurpleAreaAugmentError(
            f"无法读取历史训练金标：{path}: {exc}"
        ) from exc
    images = payload.get("images") if isinstance(payload, Mapping) else None
    if not isinstance(images, list):
        raise ProtocolPurpleAreaAugmentError(
            "历史训练金标缺少 images 数组"
        )
    records = {
        str(record.get("session_id") or ""): record
        for record in images
        if isinstance(record, Mapping)
        and record.get("status") == "reviewed"
        and record.get("session_id")
    }
    expected = tuple(str(value) for value in expected_session_ids)
    if (
        not expected
        or len(expected) != len(set(expected))
        or set(records) != set(expected)
    ):
        raise ProtocolPurpleAreaAugmentError(
            "历史紫色几何会话与冻结 v3 世界没有严格一一对应"
        )
    average_areas = tuple(
        _purple_average_area(records[session_id])
        for session_id in expected
    )
    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "sessions": len(expected),
        "session_fingerprint": canonical_sha256(list(expected)),
        "reference_fingerprint": canonical_sha256(
            [
                {
                    "session_id": session_id,
                    "purple_average_area": area,
                }
                for session_id, area in zip(expected, average_areas)
            ]
        ),
        "fields_extracted": [
            "session_id",
            "annotations.quality",
            "annotations.width",
            "annotations.height",
        ],
        "item_values_extracted": False,
        "current_settlement_used": False,
    }
    return PurpleAreaReference(
        session_ids=expected,
        average_areas=average_areas,
        provenance=provenance,
    )


def _purple_area_observation(decision: Mapping) -> dict:
    accepted: list[tuple[int, float]] = []
    for event in decision.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        semantics = event.get("semantics") or {}
        if not isinstance(semantics, Mapping):
            continue
        if (
            str(semantics.get("effect") or "")
            != MODEL_CONFIG["accepted_effect"]
            or str(semantics.get("quality") or "")
            != MODEL_CONFIG["accepted_quality"]
        ):
            continue
        if (
            semantics.get("parsed") is not True
            or semantics.get("protocol_source")
            not in {
                "packet_capture",
                "ocr_exact",
                "ocr_thresholded",
                "human_confirmed",
            }
        ):
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格事件不是已确认的精确语义"
            )
        try:
            event_round = int(event.get("round") or 0)
            value = float(
                semantics[MODEL_CONFIG["measurement_field"]]
            )
            decision_round = int(decision.get("round") or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格事件字段无效"
            ) from exc
        if (
            event_round <= 0
            or event_round > decision_round
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格事件轮次或数值无效"
            )
        accepted.append((event_round, value))
    if not accepted:
        return {
            "status": "no_exact_purple_average_area",
            "accepted_event_count": 0,
            "used_event_count": 0,
            "ignored_duplicate_event_count": 0,
            "observed_average_cells": None,
            "used_round": None,
        }
    distinct = {round(value, 8) for _, value in accepted}
    if len(distinct) != 1:
        raise ProtocolPurpleAreaAugmentError(
            "同一决策内紫色平均占格精确事件互相矛盾"
        )
    selected = min(accepted, key=lambda row: row[0])
    return {
        "status": "exact_first_purple_average_area_used",
        "accepted_event_count": len(accepted),
        "used_event_count": 1,
        "ignored_duplicate_event_count": len(accepted) - 1,
        "observed_average_cells": selected[1],
        "used_round": selected[0],
    }


@dataclass
class ProtocolPurpleAreaAugmentModel:
    """Conservative point-only wrapper around one guarded v3 model."""

    guarded_v3: ProtocolPositionQualityAugmentModel
    reference: PurpleAreaReference
    config: dict

    def __post_init__(self) -> None:
        if not isinstance(
            self.guarded_v3,
            ProtocolPositionQualityAugmentModel,
        ):
            raise ProtocolPurpleAreaAugmentError(
                "guarded_v3 类型错误"
            )
        if self.config != MODEL_CONFIG:
            raise ProtocolPurpleAreaAugmentError(
                "v4 配置必须与固定实验配置完全一致"
            )
        world_session_ids = tuple(
            world.session_id
            for world in self.guarded_v3.frozen_v2.worlds
        )
        if self.reference.session_ids != world_session_ids:
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格参考顺序与 v3 世界不一致"
            )

    @classmethod
    def fit(
        cls,
        *,
        guarded_v3: ProtocolPositionQualityAugmentModel,
        final_annotations_path: Path,
    ) -> "ProtocolPurpleAreaAugmentModel":
        if not isinstance(
            guarded_v3,
            ProtocolPositionQualityAugmentModel,
        ):
            raise ProtocolPurpleAreaAugmentError(
                "guarded_v3 类型错误"
            )
        reference = load_purple_area_reference(
            final_annotations_path,
            expected_session_ids=[
                world.session_id
                for world in guarded_v3.frozen_v2.worlds
            ],
        )
        return cls(
            guarded_v3=guarded_v3,
            reference=reference,
            config=json.loads(
                json.dumps(MODEL_CONFIG, ensure_ascii=False)
            ),
        )

    def predict(self, decision: Mapping) -> dict:
        if _contains_forbidden_label_key(decision):
            raise ProtocolPurpleAreaAugmentError(
                "预测输入禁止包含结算或标签字段"
            )
        base = self.guarded_v3.predict(decision)
        observation = _purple_area_observation(decision)
        if observation["used_event_count"] == 0:
            return {
                **base,
                "schema_version": MODEL_SCHEMA_VERSION,
                "base_v3_prediction": int(base["prediction"]),
                "purple_area_prediction_delta": 0,
                "purple_area_raw_posterior_delta": 0,
                "purple_area_evidence": observation,
                "uncertainty_status": self.config[
                    "uncertainty_contract"
                ],
                "selective_risk": {
                    **dict(base["selective_risk"]),
                    "status": "inherited_v3_not_revalidated_for_v4",
                },
            }
        (
            posterior_base,
            _,
            _,
            totals,
            current_weights,
            _,
            current_median,
        ) = self.guarded_v3._posterior(decision)
        reference_values = np.asarray(
            self.reference.average_areas,
            dtype=float,
        )
        observed = float(observation["observed_average_cells"])
        sigma = float(self.config["gaussian_sigma_cells"])
        likelihood = np.exp(
            -0.5 * ((reference_values - observed) / sigma) ** 2
        )
        updated_weights = current_weights * likelihood
        weight_sum = float(updated_weights.sum())
        if weight_sum < float(self.config["minimum_weight_sum"]):
            raise ProtocolPurpleAreaAugmentError(
                "紫色平均占格后验权重质量不足"
            )
        updated_weights /= weight_sum
        conditioned_median = _weighted_quantile(
            totals,
            updated_weights,
            0.5,
        )
        raw_delta = conditioned_median - current_median
        requested_delta = round(
            float(self.config["point_delta_scale"]) * raw_delta
        )
        prediction = max(
            1,
            int(posterior_base["known_value_lower_bound"]),
            int(base["prediction"]) + requested_delta,
        )
        applied_delta = prediction - int(base["prediction"])
        shifted_support = [
            {
                **dict(row),
                "total_value": max(
                    int(posterior_base["known_value_lower_bound"]),
                    int(row["total_value"]) + applied_delta,
                ),
            }
            for row in base.get("distribution_support") or []
        ]
        p10 = max(
            int(posterior_base["known_value_lower_bound"]),
            int(base["p10"]) + applied_delta,
        )
        p90 = max(p10, int(base["p90"]) + applied_delta)
        return {
            **base,
            "schema_version": MODEL_SCHEMA_VERSION,
            "prediction": prediction,
            "p10": p10,
            "p90": p90,
            "distribution_support": shifted_support,
            "base_v3_prediction": int(base["prediction"]),
            "purple_area_prediction_delta": applied_delta,
            "purple_area_raw_posterior_delta": round(raw_delta),
            "purple_area_conditioned_analog_median": round(
                conditioned_median
            ),
            "purple_area_evidence": observation,
            "uncertainty_status": self.config[
                "uncertainty_contract"
            ],
            "selective_risk": {
                **dict(base["selective_risk"]),
                "status": "inherited_v3_not_revalidated_for_v4",
            },
        }

    def identity(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "config": self.config,
            "guarded_v3_identity": self.guarded_v3.identity(),
            "purple_area_reference": {
                key: value
                for key, value in self.reference.provenance.items()
                if key != "path"
            },
        }
