from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .models import CatalogItem
from .observations import ObservationSnapshot
from .predictor import EmpiricalWorldPredictor, PredictionResult


V6_CANDIDATE_ID = "protocol-round-residual-v6-72a08074c3d8af81"
ADAPTER_SCHEMA = "ocr-visible-pre-bid-adapter-v1"
BASE_GENERATION_WEIGHT = 0.40
MAX_GENERATION_WEIGHT = 0.80
GENERATION_WEIGHT_STEP = 0.10
STRONG_CONDITIONING_EFFECTS = frozenset(
    {
        "quality_total_value",
        "quality_item_count",
        "total_occupied_cells",
        "max_item_value",
        "max_value_per_cell",
    }
)


class LatestModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptedDecision:
    decision: dict
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    strong_condition_count: int

    @property
    def accepted(self) -> bool:
        return not self.issues


class OcrDecisionAdapter:
    """Translate reviewed screen evidence into the frozen v6 input plane.

    This adapter never invents packet indexes, event codes, exact identities,
    settlement fields, or post-bid outcomes. OCR observations remain explicitly
    marked as thresholded OCR; only user-locked values are human-confirmed.
    """

    def __init__(
        self,
        catalog: Sequence[CatalogItem],
        *,
        minimum_ocr_confidence: float = 0.85,
    ) -> None:
        threshold = float(minimum_ocr_confidence)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_ocr_confidence must be in [0, 1]")
        self.minimum_ocr_confidence = threshold
        self.catalog = {item.catalog_id: item for item in catalog}

    def adapt(self, snapshot: ObservationSnapshot) -> AdaptedDecision:
        issues: list[str] = []
        warnings: list[str] = []
        if not snapshot.pre_bid_confirmed:
            issues.append("pre_bid_not_confirmed")
        elif snapshot.pre_bid_source == "visual_state_machine":
            warnings.append("pre_bid_visual_state_machine")
        if snapshot.map_rows is None:
            issues.append("exact_map_rows_not_confirmed")
        elif not snapshot.map_height_exact:
            if snapshot.map_rows_source == "automatic_scroll_estimate":
                warnings.append("map_rows_auto_estimated")
            else:
                issues.append("exact_map_rows_not_confirmed")
        if not snapshot.completeness_confirmed:
            issues.append("visible_map_not_reviewed")
        elif snapshot.completeness_source == "automatic_scroll_scan":
            warnings.append("visible_map_auto_scanned_not_human_reviewed")

        events: list[dict] = []
        seen_event_keys: set[tuple[int, str]] = set()
        for raw in snapshot.events:
            try:
                round_number = int(raw["round_number"])
            except (KeyError, TypeError, ValueError):
                issues.append("invalid_event_round")
                continue
            if round_number > snapshot.round_number:
                continue
            kind = str(raw.get("kind") or "")
            key = (round_number, kind)
            if kind not in {"public", "personal"} or key in seen_event_keys:
                issues.append(f"invalid_event_key:R{round_number}:{kind or 'missing'}")
                continue
            seen_event_keys.add(key)
            semantics = copy.deepcopy(raw.get("semantics") or {})
            if semantics.get("parsed") is not True:
                issues.append(f"unparsed_event:R{round_number}:{kind}")
                continue
            human_locked = bool(raw.get("human_locked"))
            confidence = float(raw.get("confidence") or 0.0)
            if not human_locked and confidence < self.minimum_ocr_confidence:
                issues.append(f"low_confidence_event:R{round_number}:{kind}")
                continue
            semantics["protocol_source"] = (
                "human_confirmed" if human_locked else "ocr_thresholded"
            )
            events.append(
                {
                    "round": round_number,
                    "kind": kind,
                    "ocr_confidence": 1.0 if human_locked else confidence,
                    "semantics": semantics,
                }
            )

        expected_keys = {
            (round_number, kind)
            for round_number in range(1, snapshot.round_number + 1)
            for kind in ("public", "personal")
        }
        missing = sorted(expected_keys - seen_event_keys)
        issues.extend(
            f"missing_event:R{round_number}:{kind}"
            for round_number, kind in missing
        )

        visible_items: list[dict] = []
        occupied_cells: set[tuple[int, int]] = set()
        for index, raw in enumerate(snapshot.map_items, start=1):
            confidence = float(raw.get("confidence") or 0.0)
            human_locked = bool(raw.get("human_locked"))
            if not human_locked and confidence < self.minimum_ocr_confidence:
                issues.append(
                    f"low_confidence_map_item:{raw.get('row')}:{raw.get('column')}"
                )
                continue
            try:
                row = int(raw["row"])
                column = int(raw["column"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"invalid_map_position:{index}")
                continue
            if row < 0 or column not in range(10):
                issues.append(f"invalid_map_position:{row}:{column}")
                continue
            width = int(raw.get("width") or 0)
            height = int(raw.get("height") or 0)
            spatial = str(raw.get("spatial") or "top_left")
            if spatial not in {"top_left", "outline", "complete"}:
                issues.append(f"invalid_map_spatial:{row}:{column}")
                continue
            if (width > 0) != (height > 0):
                issues.append(f"invalid_map_size:{row}:{column}")
                continue
            known_size = None
            if spatial in {"outline", "complete"} and width > 0 and height > 0:
                if width > 3 or height > 3:
                    issues.append(f"invalid_map_size:{row}:{column}")
                    continue
                if column + width > 10 or (
                    snapshot.map_rows is not None
                    and row + height > snapshot.map_rows
                ):
                    issues.append(f"map_item_out_of_bounds:{row}:{column}")
                    continue
                cells = {
                    (item_row, item_column)
                    for item_row in range(row, row + height)
                    for item_column in range(column, column + width)
                }
                if occupied_cells & cells:
                    issues.append(f"overlapping_map_item:{row}:{column}")
                    continue
                occupied_cells.update(cells)
                known_size = {"width": width, "height": height}

            catalog_id = str(raw.get("catalog_id") or "")
            catalog_item = self.catalog.get(catalog_id)
            # A detector "complete" class includes identity classification.
            # Thresholded OCR is accepted by default; manual promotion still
            # requires the UI to attach an explicit catalog id.
            identity_known = bool(
                spatial == "complete" and catalog_item is not None
            )
            quality = str(raw.get("quality") or "") or None
            if quality not in {None, "白", "蓝", "紫", "金", "彩"}:
                issues.append(f"invalid_map_quality:{row}:{column}")
                continue
            if identity_known:
                if quality is not None and quality != catalog_item.quality:
                    issues.append(f"identity_quality_conflict:{row}:{column}")
                    continue
                if known_size is not None and (
                    width != catalog_item.width or height != catalog_item.height
                ):
                    issues.append(f"identity_size_conflict:{row}:{column}")
                    continue
                quality = catalog_item.quality
                if known_size is None:
                    width = catalog_item.width
                    height = catalog_item.height
                    if column + width > 10 or (
                        snapshot.map_rows is not None
                        and row + height > snapshot.map_rows
                    ):
                        issues.append(f"map_item_out_of_bounds:{row}:{column}")
                        continue
                    cells = {
                        (item_row, item_column)
                        for item_row in range(row, row + height)
                        for item_column in range(column, column + width)
                    }
                    if occupied_cells & cells:
                        issues.append(f"overlapping_map_item:{row}:{column}")
                        continue
                    occupied_cells.update(cells)
                    known_size = {"width": width, "height": height}
            item = {
                "track_id": f"screen-r{row}-c{column}",
                "position": {"row": row, "column": column},
                "known_size": known_size,
                "quality": quality,
                "quality_validation": (
                    "human_confirmed"
                    if human_locked and quality
                    else "ocr_thresholded"
                ),
                "spatial_knowledge": (
                    "complete"
                    if identity_known
                    else "outline"
                    if spatial in {"outline", "complete"}
                    else "top_left"
                ),
                "evidence_confidence": 1.0 if human_locked else confidence,
                "quality_confidence": 1.0 if human_locked else confidence,
                "identity_known": identity_known,
                "identity": (
                    {
                        "catalog_id": catalog_id,
                        "known_value": int(catalog_item.value),
                        "match_confidence_status": (
                            "human_confirmed"
                            if human_locked
                            else "ocr_thresholded"
                        ),
                    }
                    if identity_known
                    else {}
                ),
                "first_seen_round": int(
                    raw.get("first_seen_round") or snapshot.round_number
                ),
                "history": [],
            }
            visible_items.append(item)

        map_rows = int(snapshot.map_rows or 0)
        automatic_rows = bool(
            not snapshot.map_height_exact
            and snapshot.map_rows_source == "automatic_scroll_estimate"
        )
        decision = {
            "adapter_schema_version": ADAPTER_SCHEMA,
            "round": int(snapshot.round_number),
            "events": events,
            "visible_items": visible_items,
            "observed_map": {
                "rows": map_rows,
                "columns": 10,
                "height_exact": bool(snapshot.map_height_exact),
                "bottom_visible": bool(snapshot.map_height_exact),
                "canvas_bottom_visible": bool(snapshot.map_height_exact),
                "source": (
                    "ocr_visible_map_estimated"
                    if automatic_rows
                    else "ocr_visible_map_confirmed"
                ),
                "height_proof": {
                    "schema_version": "hidden-map-bottom-proof-v1",
                    "status": (
                        "estimated"
                        if automatic_rows
                        else "proven" if snapshot.map_height_exact else "missing"
                    ),
                    "rows": map_rows,
                    "proof_source": (
                        "automatic_scroll_alignment"
                        if automatic_rows
                        else "human_screen_confirmation"
                    ),
                },
            },
            "map_recognition": {
                "completeness": {
                    "status": (
                        "automatic_scroll_scan"
                        if snapshot.completeness_source
                        == "automatic_scroll_scan"
                        else "human_reviewed_visible_screen"
                    ),
                    "exact_counts_safe": False,
                    "checks": {
                        "visible_map_reviewed": bool(
                            snapshot.completeness_confirmed
                        ),
                        "map_height_human_confirmed": bool(
                            snapshot.map_height_exact
                        ),
                        "pre_bid_human_confirmed": bool(
                            snapshot.pre_bid_source == "human_confirmed"
                        ),
                        "pre_bid_visual_state_machine": bool(
                            snapshot.pre_bid_source == "visual_state_machine"
                        ),
                    },
                    "exact_quality_counts": {},
                    "exact_size_counts": {},
                }
            },
            "native_evidence": {
                "mode": "ocr_visible_pre_bid_v1",
                "pre_bid_proven": bool(snapshot.pre_bid_confirmed),
                "ocr_used_for_live_features": True,
                "settlement_or_final_truth_in_features": False,
            },
        }
        strong_count = len(
            {
                str((event.get("semantics") or {}).get("effect") or "")
                for event in events
            }
            & STRONG_CONDITIONING_EFFECTS
        )
        return AdaptedDecision(
            decision=decision,
            issues=tuple(dict.fromkeys(issues)),
            warnings=tuple(dict.fromkeys(warnings)),
            strong_condition_count=strong_count,
        )


def _geometric_blend(first: float, second: float, weight: float) -> int:
    if first <= 0 or second <= 0:
        raise LatestModelError("model predictions must be positive")
    return round(
        math.exp(
            (1.0 - float(weight)) * math.log(first)
            + float(weight) * math.log(second)
        )
    )


class LatestV6V2Predictor:
    """Non-actionable OCR-adapted fusion of frozen v6 and world-model-v2."""

    def __init__(
        self,
        *,
        v6_model_path: Path,
        world_model_v2_path: Path,
        treasures_csv: Path,
        catalog: Sequence[CatalogItem],
        minimum_ocr_confidence: float = 0.85,
        prior_samples: int = 4096,
    ) -> None:
        try:
            import joblib

            self.v6_model = joblib.load(Path(v6_model_path))
        except Exception as exc:
            raise LatestModelError(f"v6 model load failed: {exc}") from exc
        if str(getattr(self.v6_model, "candidate_id", "")) != V6_CANDIDATE_ID:
            raise LatestModelError("v6 candidate id mismatch")
        self.adapter = OcrDecisionAdapter(
            catalog,
            minimum_ocr_confidence=minimum_ocr_confidence,
        )
        self.world_predictor = EmpiricalWorldPredictor(
            treasures_csv,
            catalog,
            world_model_path=world_model_v2_path,
            prior_samples=prior_samples,
        )
        if self.world_predictor.world_model_version != "world_model_v2":
            raise LatestModelError("world-model-v2 artifact required")

    def predict(self, snapshot: ObservationSnapshot) -> PredictionResult:
        adapted = self.adapter.adapt(snapshot)
        if not adapted.accepted:
            return PredictionResult(
                revision=snapshot.revision,
                status="ocr_adapter_blocked",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=adapted.issues,
                contract="ocr_adapted_v6_v2_uncalibrated",
                model_version=V6_CANDIDATE_ID,
            )
        try:
            v6 = self.v6_model.predict(adapted.decision)
        except Exception as exc:
            return PredictionResult(
                revision=snapshot.revision,
                status="v6_inference_failed",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=(
                    *adapted.warnings,
                    f"v6_error:{type(exc).__name__}:{exc}",
                ),
                contract="ocr_adapted_v6_v2_uncalibrated",
                model_version=V6_CANDIDATE_ID,
            )
        world = self.world_predictor.predict(snapshot)
        if world.p50 is None:
            return PredictionResult(
                revision=snapshot.revision,
                status="v2_conditioning_conflict",
                compatible_worlds=0,
                p10=None,
                p50=None,
                p90=None,
                minimum=None,
                maximum=None,
                actionable=False,
                issues=tuple(dict.fromkeys((*adapted.warnings, *world.issues))),
                contract="ocr_adapted_v6_v2_uncalibrated",
                v6_prediction=int(v6["prediction"]),
                model_version=V6_CANDIDATE_ID,
            )
        weight = min(
            MAX_GENERATION_WEIGHT,
            BASE_GENERATION_WEIGHT
            + GENERATION_WEIGHT_STEP * adapted.strong_condition_count,
        )
        v6_p50 = int(v6["prediction"])
        v6_p10 = int(v6.get("p10") or v6_p50)
        v6_p90 = int(v6.get("p90") or v6_p50)
        p10 = _geometric_blend(v6_p10, int(world.p10), weight)
        p50 = _geometric_blend(v6_p50, int(world.p50), weight)
        p90 = max(p10, _geometric_blend(v6_p90, int(world.p90), weight))
        issues = tuple(
            dict.fromkeys(
                (
                    *adapted.warnings,
                    *world.issues,
                    "ocr_adapter_not_packet_equivalent",
                    "interval_uncalibrated",
                )
            )
        )
        return PredictionResult(
            revision=snapshot.revision,
            status="ocr_adapted_latest_model_estimate",
            compatible_worlds=world.compatible_worlds,
            p10=p10,
            p50=p50,
            p90=p90,
            minimum=world.minimum,
            maximum=world.maximum,
            actionable=False,
            issues=issues,
            contract="ocr_adapted_v6_plus_world_model_v2_uncalibrated",
            v6_prediction=v6_p50,
            v2_prediction=int(world.p50),
            generation_weight=weight,
            model_version=V6_CANDIDATE_ID,
        )
