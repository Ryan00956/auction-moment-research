"""Conservative v3 augmentation for exact position-quality packet samples.

The frozen v2 model already conditions its historical analogue worlds on two
audited packet facts:

* an exact count of coloured treasures; and
* the pooled quality composition of exact random-identity reveals.

Packet events that reveal random positions *and* their exact qualities are a
separate sampling channel.  This module starts from the frozen v2 posterior,
then applies only the first complete ``reveal_position_quality_random`` event
with a conservative likelihood temperature of 0.5.  Later events are retained
for diagnostics but do not add likelihood.  This avoids treating correlated
multi-round observations as unlimited independent evidence.

No settlement, final map, OCR field, bid, or ranking enters prediction.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from protocol_additive_value_model import canonical_sha256
from protocol_analog_value_model import (
    ProtocolAnalogEvidenceModel,
    ProtocolAnalogValueModelError,
    _contains_forbidden_label_key,
    _exact_colour_count,
    _multivariate_hypergeometric_log_likelihood,
    _random_identity_quality_sample,
    _weighted_quantile,
)
from protocol_repeated_quality_sampling import (
    EventQualitySample,
    ProtocolRepeatedQualityError,
    extract_event_quality_samples,
)


MODEL_SCHEMA_VERSION = "protocol-position-quality-augment-v3-experimental"
MODEL_CONFIG = {
    "position_quality_likelihood_temperature": 0.5,
    "maximum_position_quality_events": 1,
    "accepted_effect": "reveal_position_quality_random",
    "selection_rule": "first_chronological_exact_event",
    "require_no_additive_adjustments": True,
    "dependence_guard": (
        "withhold_position_quality_likelihood_when_an_exact_additive_"
        "scalar_adjustment_is_already_active"
    ),
    "minimum_weight_sum": 1e-12,
    "selective_risk_posterior_quantile": 0.95,
    "selective_risk_interval_weight": 0.5,
    "selective_risk_posterior_weight": 0.5,
    "selective_risk_minimum_reference_sessions": 20,
}


class ProtocolPositionQualityAugmentError(RuntimeError):
    """Raised when v3 packet evidence violates its fail-closed contract."""


def _empirical_cdf(
    sorted_reference: Sequence[float],
    value: float,
) -> float:
    if (
        not sorted_reference
        or not math.isfinite(float(value))
        or any(
            not math.isfinite(float(reference))
            for reference in sorted_reference
        )
        or any(
            float(left) > float(right)
            for left, right in zip(
                sorted_reference,
                sorted_reference[1:],
            )
        )
    ):
        raise ProtocolPositionQualityAugmentError(
            "经验风险分位数参考无效"
        )
    return bisect.bisect_right(
        sorted_reference,
        float(value),
    ) / len(sorted_reference)


def _position_quality_samples(decision: Mapping) -> dict:
    try:
        extracted = extract_event_quality_samples(decision)
    except ProtocolRepeatedQualityError as exc:
        raise ProtocolPositionQualityAugmentError(str(exc)) from exc
    accepted = [
        sample
        for sample in extracted["accepted"]
        if sample.effect == MODEL_CONFIG["accepted_effect"]
    ]
    rejected = [
        dict(row)
        for row in extracted["rejected"]
        if str(row.get("effect") or "")
        == MODEL_CONFIG["accepted_effect"]
    ]
    limit = int(MODEL_CONFIG["maximum_position_quality_events"])
    used = accepted[:limit]
    return {
        "status": (
            "exact_first_event_used"
            if used
            else "no_usable_position_quality_event"
        ),
        "accepted_event_count": len(accepted),
        "used_event_count": len(used),
        "ignored_later_event_count": max(0, len(accepted) - len(used)),
        "used": used,
        "rejected": rejected,
    }


@dataclass
class ProtocolPositionQualityAugmentModel:
    """Feature-only posterior update around one frozen v2 model."""

    frozen_v2: ProtocolAnalogEvidenceModel
    config: dict = field(
        default_factory=lambda: json.loads(
            json.dumps(MODEL_CONFIG, ensure_ascii=False)
        )
    )
    selective_risk_reference: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_v2, ProtocolAnalogEvidenceModel):
            raise ProtocolPositionQualityAugmentError(
                "frozen_v2 类型错误"
            )
        if self.config != MODEL_CONFIG:
            raise ProtocolPositionQualityAugmentError(
                "v3 配置必须与冻结实验配置完全一致"
            )

    def _posterior(
        self,
        decision: Mapping,
    ) -> tuple[dict, dict, dict, np.ndarray, np.ndarray, float, float]:
        try:
            base = self.frozen_v2.base_model.predict(decision)
            colour_count = _exact_colour_count(decision)
            identity_sample = _random_identity_quality_sample(decision)
        except ProtocolAnalogValueModelError as exc:
            raise ProtocolPositionQualityAugmentError(str(exc)) from exc
        position_sample = _position_quality_samples(decision)
        if (
            self.config["require_no_additive_adjustments"]
            and base.get("adjustments")
            and position_sample["used"]
        ):
            position_sample = {
                **position_sample,
                "status": (
                    "withheld_due_to_nonindependent_additive_adjustment"
                ),
                "withheld_event_count": len(
                    position_sample["used"]
                ),
                "used_event_count": 0,
                "ignored_later_event_count": len(
                    position_sample["used"]
                )
                + int(
                    position_sample["ignored_later_event_count"]
                ),
                "used": [],
            }
        else:
            position_sample = {
                **position_sample,
                "withheld_event_count": 0,
            }
        identity_counts = (
            identity_sample["quality_counts"]
            if identity_sample["status"]
            == "exact_random_identity_quality_sample"
            else {}
        )
        map_rows = int(base["map_rows"])
        evidence_available = bool(
            colour_count is not None
            or identity_counts
            or position_sample["used"]
        )
        prior = self.frozen_v2._weights(
            map_rows=map_rows,
            colour_count=None,
            sample_quality_counts={},
            conditioned=False,
        )
        conditioned = self.frozen_v2._weights(
            map_rows=map_rows,
            colour_count=colour_count,
            sample_quality_counts=identity_counts,
            conditioned=bool(
                colour_count is not None or identity_counts
            ),
        )
        if position_sample["used"]:
            log_weights = np.log(np.maximum(conditioned, 1e-300))
            temperature = float(
                self.config[
                    "position_quality_likelihood_temperature"
                ]
            )
            for index, world in enumerate(self.frozen_v2.worlds):
                log_weights[index] += temperature * sum(
                    _multivariate_hypergeometric_log_likelihood(
                        world,
                        sample.counts(),
                    )
                    for sample in position_sample["used"]
                )
            finite = log_weights[np.isfinite(log_weights)]
            if len(finite) == 0:
                raise ProtocolPositionQualityAugmentError(
                    "位置品质后验没有有限权重"
                )
            maximum = float(np.max(finite))
            conditioned = np.asarray(
                [
                    math.exp(float(value) - maximum)
                    if math.isfinite(float(value))
                    else 0.0
                    for value in log_weights
                ],
                dtype=float,
            )
            total = float(conditioned.sum())
            if total < float(self.config["minimum_weight_sum"]):
                raise ProtocolPositionQualityAugmentError(
                    "位置品质后验权重质量不足"
                )
            conditioned /= total
        totals = np.asarray(
            [world.total_value for world in self.frozen_v2.worlds],
            dtype=float,
        )
        quantile = float(self.frozen_v2.config["point_quantile"])
        prior_median = _weighted_quantile(totals, prior, quantile)
        conditioned_median = _weighted_quantile(
            totals,
            conditioned,
            quantile,
        )
        metadata = {
            "map_rows": map_rows,
            "colour_count": colour_count,
            "identity_sample": identity_sample,
            "position_sample": position_sample,
            "evidence_available": evidence_available,
        }
        return (
            base,
            metadata,
            position_sample,
            totals,
            conditioned,
            prior_median,
            conditioned_median,
        )

    def predict(self, decision: Mapping) -> dict:
        (
            base,
            metadata,
            position_sample,
            totals,
            conditioned_weights,
            prior_median,
            conditioned_median,
        ) = self._posterior(decision)
        evidence_delta = conditioned_median - prior_median
        prediction = max(
            1.0,
            float(base["prediction"]) + evidence_delta,
        )
        interval_scale = float(
            self.frozen_v2.config["interval_scale"]
        )
        support = prediction + interval_scale * (
            totals - conditioned_median
        )
        support = np.maximum(
            support,
            float(base["known_value_lower_bound"]),
        )
        p10 = _weighted_quantile(
            support,
            conditioned_weights,
            float(
                self.frozen_v2.config[
                    "interval_lower_quantile"
                ]
            ),
        )
        p90 = _weighted_quantile(
            support,
            conditioned_weights,
            float(
                self.frozen_v2.config[
                    "interval_upper_quantile"
                ]
            ),
        )
        effective_sample_size = 1.0 / float(
            np.sum(conditioned_weights**2)
        )
        positive = conditioned_weights[conditioned_weights > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        normalized_entropy = (
            entropy / math.log(len(self.frozen_v2.worlds))
            if len(self.frozen_v2.worlds) > 1
            else 0.0
        )
        posterior_apes = (
            np.abs(support - prediction) / np.maximum(support, 1.0)
        )
        posterior_p95_ape = _weighted_quantile(
            posterior_apes,
            conditioned_weights,
            float(
                self.config[
                    "selective_risk_posterior_quantile"
                ]
            ),
        )
        relative_interval_width = round(
            (p90 - p10) / max(prediction, 1.0),
            8,
        )
        rounded_tail = round(posterior_p95_ape, 8)
        selective_risk = {
            "schema_version": "protocol-selective-risk-v1",
            "status": "uncalibrated_feature_components_only",
            "score": None,
            "relative_interval_width_percentile": None,
            "posterior_p95_ape_percentile": None,
            "reference_fingerprint": None,
        }
        if self.selective_risk_reference:
            width_percentile = _empirical_cdf(
                self.selective_risk_reference[
                    "relative_interval_widths"
                ],
                relative_interval_width,
            )
            tail_percentile = _empirical_cdf(
                self.selective_risk_reference[
                    "posterior_p95_absolute_percentage_errors"
                ],
                rounded_tail,
            )
            interval_weight = float(
                self.config["selective_risk_interval_weight"]
            )
            tail_weight = float(
                self.config["selective_risk_posterior_weight"]
            )
            total_weight = interval_weight + tail_weight
            if total_weight <= 0:
                raise ProtocolPositionQualityAugmentError(
                    "选择性风险权重无效"
                )
            selective_risk = {
                "schema_version": "protocol-selective-risk-v1",
                "status": "calibrated_feature_only",
                "score": round(
                    (
                        interval_weight * width_percentile
                        + tail_weight * tail_percentile
                    )
                    / total_weight,
                    8,
                ),
                "relative_interval_width_percentile": round(
                    width_percentile,
                    8,
                ),
                "posterior_p95_ape_percentile": round(
                    tail_percentile,
                    8,
                ),
                "reference_fingerprint": (
                    self.selective_risk_reference[
                        "reference_fingerprint"
                    ]
                ),
            }
        support_rows = [
            {
                "historical_session_id": world.session_id,
                "total_value": round(float(value)),
                "weight": round(float(weight), 12),
            }
            for world, value, weight in zip(
                self.frozen_v2.worlds,
                support,
                conditioned_weights,
            )
            if float(weight) > 0.0
        ]
        used_samples: list[EventQualitySample] = position_sample["used"]
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "decision_id": str(decision.get("decision_id") or ""),
            "session_id": str(decision.get("session_id") or ""),
            "round": int(decision.get("round") or 0),
            "map_rows": int(metadata["map_rows"]),
            "prediction": round(prediction),
            "p10": round(max(0.0, p10)),
            "p90": round(max(p10, p90)),
            "relative_interval_width": relative_interval_width,
            "posterior_p95_absolute_percentage_error": rounded_tail,
            "selective_risk": selective_risk,
            "base_prediction": int(base["prediction"]),
            "base_row_prediction": int(base["base_row_prediction"]),
            "base_active_adjustments": list(base["adjustments"]),
            "known_value_lower_bound": int(
                base["known_value_lower_bound"]
            ),
            "prior_analog_median": round(prior_median),
            "conditioned_analog_median": round(
                conditioned_median
            ),
            "evidence_delta": round(evidence_delta),
            "evidence_available": bool(
                metadata["evidence_available"]
            ),
            "exact_colour_count": metadata["colour_count"],
            "random_identity_sample": metadata["identity_sample"],
            "position_quality_sample": {
                key: value
                for key, value in position_sample.items()
                if key != "used"
            }
            | {
                "used_events": [
                    {
                        "round": sample.round,
                        "effect": sample.effect,
                        "expected_count": sample.expected_count,
                        "quality_counts": sample.counts(),
                        "track_ids": list(sample.track_ids),
                    }
                    for sample in used_samples
                ]
            },
            "effective_sample_size": round(
                effective_sample_size,
                8,
            ),
            "normalized_weight_entropy": round(
                normalized_entropy,
                8,
            ),
            "maximum_world_weight": round(
                float(conditioned_weights.max()),
                8,
            ),
            "distribution_support": support_rows,
        }
    def calibrate_selective_risk(
        self,
        decisions: Sequence[Mapping],
        *,
        provenance: Mapping,
    ) -> None:
        if self.selective_risk_reference:
            raise ProtocolPositionQualityAugmentError(
                "选择性风险参考已经冻结，禁止重复校准"
            )
        normalized = [dict(decision) for decision in decisions]
        decision_ids = [
            str(decision.get("decision_id") or "")
            for decision in normalized
        ]
        session_ids = [
            str(decision.get("session_id") or "")
            for decision in normalized
        ]
        if (
            not normalized
            or not all(decision_ids)
            or not all(session_ids)
            or len(decision_ids) != len(set(decision_ids))
            or any(
                str(decision.get("cohort") or "")
                != "historical_shadow_train_only"
                for decision in normalized
            )
            or any(
                _contains_forbidden_label_key(decision)
                for decision in normalized
            )
        ):
            raise ProtocolPositionQualityAugmentError(
                "风险参考必须是无标签且主键唯一的 shadow 决策"
            )
        unique_sessions = sorted(set(session_ids))
        if len(unique_sessions) < int(
            self.config[
                "selective_risk_minimum_reference_sessions"
            ]
        ):
            raise ProtocolPositionQualityAugmentError(
                "选择性风险参考的 shadow 会话不足"
            )
        components = [self.predict(decision) for decision in normalized]
        reference = {
            "schema_version": "protocol-selective-risk-reference-v1",
            "calibration_source": (
                "feature_only_historical_shadow_v3"
            ),
            "labels_used": False,
            "decisions": len(normalized),
            "sessions": len(unique_sessions),
            "decision_fingerprint": canonical_sha256(
                sorted(decision_ids)
            ),
            "session_fingerprint": canonical_sha256(unique_sessions),
            "relative_interval_widths": sorted(
                float(row["relative_interval_width"])
                for row in components
            ),
            "posterior_p95_absolute_percentage_errors": sorted(
                float(
                    row[
                        "posterior_p95_absolute_percentage_error"
                    ]
                )
                for row in components
            ),
            "provenance": dict(provenance),
        }
        reference["reference_fingerprint"] = canonical_sha256(
            reference
        )
        self.selective_risk_reference = reference

    def identity(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "config": self.config,
            "frozen_v2_identity": self.frozen_v2.identity(),
            "selective_risk_reference": (
                {
                    key: value
                    for key, value in self.selective_risk_reference.items()
                    if key
                    not in {
                        "relative_interval_widths",
                        "posterior_p95_absolute_percentage_errors",
                    }
                }
                if self.selective_risk_reference
                else None
            ),
        }
