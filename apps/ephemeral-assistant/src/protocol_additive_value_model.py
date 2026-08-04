"""Interpretable total-value model using exact packet event adjustments.

The baseline is a robust map-row regressor trained only on reviewed historical
games that predate packet-capture development.  Four exact packet events then
move the estimate by a pre-frozen fraction of the difference between the
observed statistic and its historical row-conditional expectation.

No settlement, final item count, process-map truth, current ranking, or OCR
field enters prediction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


MODEL_SCHEMA_VERSION = "protocol-additive-exact-events-v1"
QUALITY_ORDER = ("白", "蓝", "紫", "金", "彩")
MODEL_CONFIG = {
    "estimator": "GradientBoostingRegressor",
    "loss": "absolute_error",
    "n_estimators": 120,
    "learning_rate": 0.04,
    "max_depth": 2,
    "min_samples_leaf": 7,
    "random_state": 7,
    "adjustment_weights": {
        "gold_total_value": 1.0,
        "max_item_value": 0.75,
        "max_value_per_cell": 1.0,
        "highest_quality": {
            "downgrade": 1.0,
            "upgrade_or_equal": 0.25,
        },
    },
    "target_scale": 1_000_000,
}


class ProtocolAdditiveValueModelError(RuntimeError):
    """Raised when model training or decision evidence is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolAdditiveValueModelError(
            f"无法读取文件：{path}: {exc}"
        ) from exc
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _training_game(record: Mapping) -> dict:
    annotations = [
        item
        for item in record.get("annotations") or []
        if isinstance(item, Mapping)
    ]
    if not annotations:
        raise ProtocolAdditiveValueModelError(
            f"训练对局缺少 annotations：{record.get('session_id')}"
        )
    try:
        rows = max(
            int(item["row"]) + int(item["height"])
            for item in annotations
        )
        values = [int(item["value"]) for item in annotations]
        gold_total = sum(
            int(item["value"])
            for item in annotations
            if str(item.get("quality") or "") == "金"
        )
        max_item_value = max(values)
        max_value_per_cell = max(
            int(item["value"])
            / (int(item["width"]) * int(item["height"]))
            for item in annotations
        )
        highest_quality = max(
            QUALITY_ORDER.index(str(item["quality"])) + 1
            for item in annotations
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProtocolAdditiveValueModelError(
            f"训练对局字段无效：{record.get('session_id')}"
        ) from exc
    return {
        "session_id": str(record.get("session_id") or ""),
        "map_rows": rows,
        "total_value": sum(values),
        "gold_total_value": gold_total,
        "max_item_value": max_item_value,
        "max_value_per_cell": max_value_per_cell,
        "highest_quality": highest_quality,
    }


def load_training_games(final_annotations_path: Path) -> tuple[list[dict], dict]:
    try:
        payload = json.loads(
            Path(final_annotations_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolAdditiveValueModelError(
            f"无法读取历史训练金标：{final_annotations_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtocolAdditiveValueModelError(
            "历史训练金标顶层必须是对象"
        )
    images = payload.get("images")
    if not isinstance(images, list):
        raise ProtocolAdditiveValueModelError(
            "历史训练金标缺少 images"
        )
    records = [
        record
        for record in images
        if isinstance(record, Mapping)
        and record.get("status") == "reviewed"
        and record.get("session_id")
    ]
    games = [_training_game(record) for record in records]
    session_ids = [game["session_id"] for game in games]
    if (
        not games
        or len(session_ids) != len(set(session_ids))
        or session_ids != sorted(session_ids)
    ):
        raise ProtocolAdditiveValueModelError(
            "历史训练会话为空、重复或未按时间排序"
        )
    provenance = {
        "path": str(Path(final_annotations_path).resolve()),
        "sha256": sha256_file(final_annotations_path),
        "sessions": len(games),
        "first_session_id": session_ids[0],
        "last_session_id": session_ids[-1],
        "session_fingerprint": canonical_sha256(session_ids),
    }
    return games, provenance


def _new_estimator() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss=str(MODEL_CONFIG["loss"]),
        n_estimators=int(MODEL_CONFIG["n_estimators"]),
        learning_rate=float(MODEL_CONFIG["learning_rate"]),
        max_depth=int(MODEL_CONFIG["max_depth"]),
        min_samples_leaf=int(MODEL_CONFIG["min_samples_leaf"]),
        random_state=int(MODEL_CONFIG["random_state"]),
    )


def _event_observations(decision: Mapping) -> dict[str, float]:
    result: dict[str, float] = {}
    for event in decision.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        semantics = event.get("semantics") or {}
        if not isinstance(semantics, Mapping):
            continue
        if semantics.get("protocol_source") not in {
            "packet_capture",
            "ocr_exact",
            "ocr_thresholded",
            "human_confirmed",
        }:
            raise ProtocolAdditiveValueModelError(
                "候选模型只接受已确认的决策时事件语义"
            )
        effect = str(semantics.get("effect") or "")
        if (
            effect == "quality_total_value"
            and semantics.get("quality") == "金"
        ):
            result["gold_total_value"] = float(
                semantics["observed_value"]
            )
        elif effect == "max_item_value":
            result["max_item_value"] = float(
                semantics["observed_value"]
            )
        elif effect == "max_value_per_cell":
            result["max_value_per_cell"] = float(
                semantics["observed_value"]
            )
        elif effect == "highest_quality":
            quality = str(semantics.get("observed_quality") or "")
            if quality not in QUALITY_ORDER:
                raise ProtocolAdditiveValueModelError(
                    f"最高品质无效：{quality!r}"
                )
            result["highest_quality"] = float(
                QUALITY_ORDER.index(quality) + 1
            )
    return result


def _known_value_lower_bound(
    decision: Mapping,
    observations: Mapping[str, float],
) -> float:
    identity_sum = sum(
        int((item.get("identity") or {}).get("known_value") or 0)
        for item in decision.get("visible_items") or []
        if isinstance(item, Mapping)
    )
    quality_totals: dict[str, int] = {}
    for event in decision.get("events") or []:
        semantics = (
            event.get("semantics") or {}
            if isinstance(event, Mapping)
            else {}
        )
        if semantics.get("effect") == "quality_total_value":
            quality = str(semantics.get("quality") or "")
            quality_totals[quality] = int(
                semantics.get("observed_value") or 0
            )
    lower_bounds = [
        0,
        identity_sum,
        sum(quality_totals.values()),
        int(observations.get("max_item_value") or 0),
        int(observations.get("max_value_per_cell") or 0),
    ]
    return float(max(lower_bounds))


@dataclass
class ProtocolAdditiveExactEventModel:
    estimators: dict[str, GradientBoostingRegressor]
    training_provenance: dict
    config: dict

    @classmethod
    def fit(
        cls,
        games: Sequence[Mapping],
        *,
        training_provenance: Mapping,
    ) -> "ProtocolAdditiveExactEventModel":
        if not games:
            raise ProtocolAdditiveValueModelError(
                "没有可训练的历史对局"
            )
        scale = float(MODEL_CONFIG["target_scale"])
        rows = np.asarray(
            [[float(game["map_rows"])] for game in games],
            dtype=float,
        )
        targets = {
            "total_value": np.asarray(
                [float(game["total_value"]) / scale for game in games]
            ),
            "gold_total_value": np.asarray(
                [
                    float(game["gold_total_value"]) / scale
                    for game in games
                ]
            ),
            "max_item_value": np.asarray(
                [
                    float(game["max_item_value"]) / scale
                    for game in games
                ]
            ),
            "max_value_per_cell": np.asarray(
                [
                    float(game["max_value_per_cell"]) / scale
                    for game in games
                ]
            ),
            "highest_quality": np.asarray(
                [float(game["highest_quality"]) for game in games]
            ),
        }
        estimators = {}
        for name, target in targets.items():
            estimator = _new_estimator()
            estimator.fit(rows, target)
            estimators[name] = estimator
        return cls(
            estimators=estimators,
            training_provenance=dict(training_provenance),
            config=json.loads(
                json.dumps(MODEL_CONFIG, ensure_ascii=False)
            ),
        )

    def _row_expectation(self, name: str, map_rows: int) -> float:
        estimator = self.estimators.get(name)
        if estimator is None:
            raise ProtocolAdditiveValueModelError(
                f"模型缺少估计器：{name}"
            )
        raw = float(estimator.predict([[float(map_rows)]])[0])
        if name == "highest_quality":
            return raw
        return raw * float(self.config["target_scale"])

    def predict(self, decision: Mapping) -> dict:
        observed_map = decision.get("observed_map") or {}
        proof = observed_map.get("height_proof") or {}
        try:
            map_rows = int(observed_map["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolAdditiveValueModelError(
                "决策缺少有效 map_rows"
            ) from exc
        exact_height_proof = (
            observed_map.get("height_exact") is True
            and proof.get("schema_version")
            == "hidden-map-bottom-proof-v1"
            and proof.get("status") == "proven"
            and int(proof.get("rows") or 0) == map_rows
            and str(observed_map.get("source") or "")
            in {
                "packet_capture_match_opened",
                "ocr_visible_map_confirmed",
            }
        )
        estimated_scroll_proof = (
            observed_map.get("height_exact") is False
            and proof.get("schema_version")
            == "hidden-map-bottom-proof-v1"
            and proof.get("status") == "estimated"
            and int(proof.get("rows") or 0) == map_rows
            and str(observed_map.get("source") or "")
            == "ocr_visible_map_estimated"
            and str(proof.get("proof_source") or "")
            == "automatic_scroll_alignment"
        )
        if not (exact_height_proof or estimated_scroll_proof):
            raise ProtocolAdditiveValueModelError(
                "候选模型要求人工精确行数或显式标记的自动滚动估计"
            )
        observations = _event_observations(decision)
        base = self._row_expectation("total_value", map_rows)
        prediction = base
        adjustments: list[dict] = []
        weights = self.config["adjustment_weights"]
        for name in (
            "gold_total_value",
            "max_item_value",
            "max_value_per_cell",
            "highest_quality",
        ):
            observed = observations.get(name)
            if observed is None:
                continue
            expected = self._row_expectation(name, map_rows)
            difference = float(observed) - expected
            if name == "highest_quality":
                weight = float(
                    weights[name][
                        "downgrade"
                        if difference < 0
                        else "upgrade_or_equal"
                    ]
                )
                difference *= float(self.config["target_scale"])
            else:
                weight = float(weights[name])
            delta = weight * difference
            prediction += delta
            adjustments.append(
                {
                    "feature": name,
                    "observed": round(float(observed), 8),
                    "row_expected": round(float(expected), 8),
                    "weight": weight,
                    "delta": round(float(delta)),
                }
            )
        lower_bound = _known_value_lower_bound(
            decision,
            observations,
        )
        unclamped = prediction
        prediction = max(prediction, lower_bound, 1.0)
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "decision_id": str(decision.get("decision_id") or ""),
            "session_id": str(decision.get("session_id") or ""),
            "round": int(decision.get("round") or 0),
            "map_rows": map_rows,
            "base_row_prediction": round(base),
            "adjustments": adjustments,
            "active_adjustment_count": len(adjustments),
            "unclamped_prediction": round(unclamped),
            "known_value_lower_bound": round(lower_bound),
            "lower_bound_clamped": prediction != unclamped,
            "prediction": round(prediction),
        }

    def identity(self) -> dict:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "config": self.config,
            "training_provenance": self.training_provenance,
        }
