from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from auction_moment_assistant.models import load_catalog
from auction_moment_assistant.observations import (
    EventObservation,
    ObservationStore,
)
from auction_moment_assistant.predictor import EmpiricalWorldPredictor


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TREASURES = REPOSITORY_ROOT / "data" / "v1" / "core" / "treasures.csv"


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(TREASURES)
        cls.predictor = EmpiricalWorldPredictor(TREASURES, cls.catalog)

    def test_missing_events_is_provisional_and_never_actionable(self) -> None:
        store = ObservationStore()
        result = self.predictor.predict(store.snapshot())
        self.assertEqual(result.status, "provisional_event_ocr")
        self.assertFalse(result.actionable)
        self.assertGreater(result.compatible_worlds, 1000)

    def test_exact_total_count_filters_public_worlds(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[
                EventObservation(
                    1,
                    "public",
                    "显示本局藏品总数量20",
                    1.0,
                    {
                        "parsed": True,
                        "effect": "total_item_count",
                        "observed_count": 20,
                    },
                    source="human",
                    human_locked=True,
                ),
                EventObservation(
                    1,
                    "personal",
                    "随机显示4件藏品的位置",
                    1.0,
                    {
                        "parsed": True,
                        "effect": "reveal_position_random",
                        "count": 4,
                    },
                    source="human",
                    human_locked=True,
                ),
            ],
            bankroll=None,
            map_items=[],
        )
        result = self.predictor.predict(store.snapshot())
        self.assertGreater(result.compatible_worlds, 0)
        self.assertLess(result.compatible_worlds, 1794)
        self.assertIsNotNone(result.p50)

    def test_release_world_model_is_consumed_when_configured(self) -> None:
        from auction_moment_research.train_world_model import train

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "world-model.json"
            train(TREASURES, model_path)
            predictor = EmpiricalWorldPredictor(
                TREASURES,
                self.catalog,
                world_model_path=model_path,
                prior_samples=16,
            )
        result = predictor.predict(ObservationStore().snapshot())
        self.assertEqual(predictor.prior_world_count, 16)
        self.assertIn("world_model_prior", result.contract)

    def test_top_left_unknown_quality_still_constrains_visible_count(self) -> None:
        mask = np.ones(len(self.predictor.world_counts), dtype=bool)
        conditioned = self.predictor._apply_visible_items(
            mask,
            [
                {
                    "row": 0,
                    "column": 0,
                    "confidence": 0.95,
                    "spatial": "top_left",
                }
            ],
        )
        np.testing.assert_array_equal(
            conditioned, self.predictor.total_counts >= 1
        )

    def test_top_left_marker_geometry_is_not_treated_as_known_size(self) -> None:
        base = {
            "row": 0,
            "column": 0,
            "confidence": 0.95,
            "width": 3,
            "height": 3,
        }
        top_left = self.predictor._apply_visible_items(
            np.ones(len(self.predictor.world_counts), dtype=bool),
            [{**base, "spatial": "top_left"}],
        )
        outline = self.predictor._apply_visible_items(
            np.ones(len(self.predictor.world_counts), dtype=bool),
            [{**base, "spatial": "outline"}],
        )
        expected_top_left = self.predictor.total_counts >= 1
        size_mask = self.predictor.size_masks[(3, 3)]
        expected_outline = expected_top_left & (
            self.predictor.world_counts[:, size_mask].sum(axis=1) >= 1
        )
        np.testing.assert_array_equal(top_left, expected_top_left)
        np.testing.assert_array_equal(outline, expected_outline)

    def test_thresholded_ocr_complete_identity_is_an_exact_minimum(self) -> None:
        catalog_id = self.catalog[0].catalog_id
        index = self.predictor.index_by_id[catalog_id]
        conditioned = self.predictor._apply_visible_items(
            np.ones(len(self.predictor.world_counts), dtype=bool),
            [
                {
                    "row": 0,
                    "column": 0,
                    "confidence": 0.95,
                    "spatial": "complete",
                    "catalog_id": catalog_id,
                }
            ],
        )
        expected = (self.predictor.total_counts >= 1) & (
            self.predictor.world_counts[:, index] >= 1
        )
        np.testing.assert_array_equal(conditioned, expected)

    def test_reports_the_event_that_eliminates_all_worlds(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[
                EventObservation(
                    1,
                    "public",
                    "显示本局藏品总数量999",
                    1.0,
                    {
                        "parsed": True,
                        "effect": "total_item_count",
                        "observed_count": 999,
                    },
                    source="human",
                    human_locked=True,
                )
            ],
            bankroll=None,
            map_items=[],
        )
        result = self.predictor.predict(store.snapshot())
        self.assertIsNone(result.p50)
        self.assertIn(
            "event_conflict:R1:public:total_item_count:999",
            result.diagnostics,
        )

    def test_reports_visible_count_constraint_that_eliminates_worlds(self) -> None:
        maximum = int(self.predictor.total_counts.max())
        diagnostics: list[str] = []
        conditioned = self.predictor._apply_visible_items(
            np.ones(len(self.predictor.world_counts), dtype=bool),
            [
                {
                    "row": index,
                    "column": 0,
                    "confidence": 0.95,
                    "spatial": "top_left",
                }
                for index in range(maximum + 1)
            ],
            diagnostics,
        )
        self.assertFalse(np.any(conditioned))
        self.assertEqual(
            diagnostics,
            [f"map_conflict:visible_count:{maximum + 1}"],
        )


if __name__ == "__main__":
    unittest.main()
