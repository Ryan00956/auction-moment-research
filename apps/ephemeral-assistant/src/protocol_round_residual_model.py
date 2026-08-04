"""Frozen-v4 hierarchical round residual challenger.

The model is intentionally small-data conservative: one global Ridge model
and one Ridge model per auction round predict a bounded log correction around
the frozen v4 estimate.  Runtime features are packet-only and pre-bid.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from protocol_packet_residual_features import (
    SCHEMA_VERSION as FEATURE_SCHEMA_VERSION,
    event_structural_base_profile,
    packet_residual_features,
)
from protocol_purple_area_augment_model import (
    ProtocolPurpleAreaAugmentModel,
)


SCHEMA_VERSION = "protocol-round-residual-model-v1"


class ProtocolRoundResidualModelError(RuntimeError):
    """Raised when the v6 residual model contract is violated."""


def _model(alpha: float):
    return make_pipeline(
        DictVectorizer(sparse=False, sort=True),
        StandardScaler(),
        Ridge(alpha=float(alpha)),
    )


def _session_weights(session_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(str(value) for value in session_ids)
    return np.asarray(
        [1.0 / counts[str(value)] for value in session_ids],
        dtype=float,
    )


def _validate_decisions(decisions: Sequence[Mapping]) -> None:
    if not decisions:
        raise ProtocolRoundResidualModelError(
            "training decisions are empty"
        )
    decision_ids = [
        str(row.get("decision_id") or "") for row in decisions
    ]
    if (
        not all(decision_ids)
        or len(decision_ids) != len(set(decision_ids))
    ):
        raise ProtocolRoundResidualModelError(
            "training decision ids are empty or duplicated"
        )
    for decision in decisions:
        try:
            round_number = int(decision["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolRoundResidualModelError(
                "training round is invalid"
            ) from exc
        if round_number not in range(1, 6):
            raise ProtocolRoundResidualModelError(
                f"training round is outside 1..5: {round_number}"
            )


class ProtocolRoundResidualModel:
    """Bounded hierarchical point correction around frozen v4."""

    def __init__(
        self,
        *,
        candidate_id: str,
        frozen_v4: ProtocolPurpleAreaAugmentModel,
        global_estimator,
        round_estimators: Mapping[int, object],
        alpha: float,
        round_mix: float,
        correction_scale: float,
        maximum_absolute_log_correction: float,
        minimum_round_training_rows: int,
        training_session_ids: Sequence[str],
        training_decision_count: int,
        feature_names: Sequence[str],
    ) -> None:
        if not candidate_id:
            raise ProtocolRoundResidualModelError(
                "candidate id is empty"
            )
        if not isinstance(
            frozen_v4, ProtocolPurpleAreaAugmentModel
        ):
            raise ProtocolRoundResidualModelError(
                "base is not frozen ProtocolPurpleAreaAugmentModel"
            )
        if (
            float(alpha) <= 0
            or not 0.0 <= float(round_mix) <= 1.0
            or not 0.0 < float(correction_scale) <= 1.0
            or float(maximum_absolute_log_correction) <= 0.0
            or int(minimum_round_training_rows) <= 0
        ):
            raise ProtocolRoundResidualModelError(
                "residual model hyperparameters are invalid"
            )
        normalized_rounds = {
            int(round_number): estimator
            for round_number, estimator in round_estimators.items()
        }
        if not normalized_rounds or any(
            round_number not in range(1, 6)
            for round_number in normalized_rounds
        ):
            raise ProtocolRoundResidualModelError(
                "round estimators are missing or invalid"
            )
        sessions = tuple(str(value) for value in training_session_ids)
        if (
            not sessions
            or tuple(sorted(set(sessions))) != sessions
            or int(training_decision_count) <= 0
            or not feature_names
        ):
            raise ProtocolRoundResidualModelError(
                "training provenance is invalid"
            )
        self.candidate_id = candidate_id
        self.frozen_v4 = frozen_v4
        self.global_estimator = global_estimator
        self.round_estimators = normalized_rounds
        self.alpha = float(alpha)
        self.round_mix = float(round_mix)
        self.correction_scale = float(correction_scale)
        self.maximum_absolute_log_correction = float(
            maximum_absolute_log_correction
        )
        self.minimum_round_training_rows = int(
            minimum_round_training_rows
        )
        self.training_session_ids = sessions
        self.training_decision_count = int(training_decision_count)
        self.feature_names = tuple(sorted(set(feature_names)))

    def _features(
        self,
        decision: Mapping,
        base: Mapping,
    ) -> dict[str, float]:
        return event_structural_base_profile(
            packet_residual_features(decision, base)
        )

    def predict(self, decision: Mapping) -> dict:
        base = self.frozen_v4.predict(decision)
        features = self._features(decision, base)
        try:
            round_number = int(decision["round"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolRoundResidualModelError(
                "decision round is invalid"
            ) from exc
        if round_number not in range(1, 6):
            raise ProtocolRoundResidualModelError(
                "decision round is outside 1..5"
            )
        global_raw = float(
            self.global_estimator.predict([features])[0]
        )
        round_estimator = self.round_estimators.get(round_number)
        if round_estimator is None:
            round_raw = global_raw
            round_status = "global_fallback"
        else:
            round_raw = float(round_estimator.predict([features])[0])
            round_status = "round_specific"
        if not math.isfinite(global_raw) or not math.isfinite(round_raw):
            raise ProtocolRoundResidualModelError(
                "residual estimator produced non-finite output"
            )
        mixed_raw = (
            (1.0 - self.round_mix) * global_raw
            + self.round_mix * round_raw
        )
        unclipped = self.correction_scale * mixed_raw
        correction = max(
            -self.maximum_absolute_log_correction,
            min(self.maximum_absolute_log_correction, unclipped),
        )
        base_prediction = float(base["prediction"])
        lower_bound = float(base.get("known_value_lower_bound") or 0.0)
        prediction = max(
            1.0,
            lower_bound,
            base_prediction * math.exp(correction),
        )
        shift = prediction / max(1.0, base_prediction)
        base_p10 = float(base.get("p10") or base_prediction)
        base_p90 = float(base.get("p90") or base_prediction)
        p10 = max(0.0, lower_bound, base_p10 * shift)
        p90 = max(p10, lower_bound, base_p90 * shift)
        result = dict(base)
        result.update(
            {
                "prediction": round(prediction),
                "p10": round(p10),
                "p90": round(p90),
                "relative_interval_width": round(
                    (p90 - p10) / max(1.0, prediction),
                    8,
                ),
                "model_version": self.candidate_id,
                "base_model_version": str(
                    base.get("model_version") or ""
                ),
                "frozen_v4_prediction": round(base_prediction),
                "round_residual": {
                    "schema_version": SCHEMA_VERSION,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_profile": "event_structural_base",
                    "round_model_status": round_status,
                    "round": round_number,
                    "global_raw_log_correction": round(global_raw, 8),
                    "round_raw_log_correction": round(round_raw, 8),
                    "model_disagreement": round(
                        abs(round_raw - global_raw), 8
                    ),
                    "mixed_raw_log_correction": round(mixed_raw, 8),
                    "unclipped_log_correction": round(unclipped, 8),
                    "applied_log_correction": round(correction, 8),
                    "correction_clipped": bool(
                        abs(unclipped - correction) > 1e-12
                    ),
                    "prediction_multiplier": round(shift, 8),
                    "settlement_or_final_truth_consumed": False,
                    "ocr_used_for_features": bool(
                        (decision.get("native_evidence") or {}).get(
                            "ocr_used_for_live_features"
                        )
                    ),
                },
            }
        )
        return result


def fit_protocol_round_residual_model(
    *,
    candidate_id: str,
    frozen_v4: ProtocolPurpleAreaAugmentModel,
    decisions: Sequence[Mapping],
    labels: Mapping[str, int | float] | None = None,
    label_loader: (
        Callable[[], Mapping[str, int | float]] | None
    ) = None,
    alpha: float = 600.0,
    round_mix: float = 0.5,
    correction_scale: float = 0.5,
    maximum_absolute_log_correction: float = 0.1,
    minimum_round_training_rows: int = 10,
) -> ProtocolRoundResidualModel:
    """Fit the preselected v6 configuration on revealed development games."""

    _validate_decisions(decisions)
    if not isinstance(
        frozen_v4, ProtocolPurpleAreaAugmentModel
    ):
        raise ProtocolRoundResidualModelError(
            "base is not frozen ProtocolPurpleAreaAugmentModel"
        )
    ordered = sorted(
        (dict(row) for row in decisions),
        key=lambda row: (
            str(row["session_id"]),
            int(row["round"]),
            str(row["decision_id"]),
        ),
    )
    base_rows = [frozen_v4.predict(row) for row in ordered]
    features = [
        event_structural_base_profile(
            packet_residual_features(decision, base)
        )
        for decision, base in zip(ordered, base_rows)
    ]
    # Labels are touched only after every live feature and base prediction is
    # materialized.
    if (labels is None) == (label_loader is None):
        raise ProtocolRoundResidualModelError(
            "provide exactly one of labels or label_loader"
        )
    resolved_labels = labels if labels is not None else label_loader()
    try:
        actual = np.asarray(
            [
                float(resolved_labels[str(row["decision_id"])])
                for row in ordered
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolRoundResidualModelError(
            "training labels do not align with decisions"
        ) from exc
    if (
        len(resolved_labels) != len(ordered)
        or np.any(~np.isfinite(actual))
        or np.any(actual <= 0)
    ):
        raise ProtocolRoundResidualModelError(
            "training labels are invalid or contain extra rows"
        )
    baseline = np.asarray(
        [float(row["prediction"]) for row in base_rows],
        dtype=float,
    )
    target = np.log(actual / np.maximum(baseline, 1.0))
    session_ids = np.asarray(
        [str(row["session_id"]) for row in ordered]
    )
    rounds = np.asarray([int(row["round"]) for row in ordered])
    weights = _session_weights(session_ids.tolist())

    global_estimator = _model(alpha)
    global_estimator.fit(
        features,
        target,
        ridge__sample_weight=weights,
    )
    round_estimators = {}
    for round_number in range(1, 6):
        indices = np.flatnonzero(rounds == round_number)
        if len(indices) < int(minimum_round_training_rows):
            continue
        estimator = _model(alpha)
        estimator.fit(
            [features[index] for index in indices],
            target[indices],
            ridge__sample_weight=_session_weights(
                session_ids[indices].tolist()
            ),
        )
        round_estimators[round_number] = estimator
    return ProtocolRoundResidualModel(
        candidate_id=candidate_id,
        frozen_v4=frozen_v4,
        global_estimator=global_estimator,
        round_estimators=round_estimators,
        alpha=alpha,
        round_mix=round_mix,
        correction_scale=correction_scale,
        maximum_absolute_log_correction=(
            maximum_absolute_log_correction
        ),
        minimum_round_training_rows=minimum_round_training_rows,
        training_session_ids=tuple(sorted(set(session_ids.tolist()))),
        training_decision_count=len(ordered),
        feature_names=tuple(
            sorted(set().union(*(row.keys() for row in features)))
        ),
    )
