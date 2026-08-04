from __future__ import annotations

import unittest

import numpy as np

from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.vision import (
    Detection,
    VisionRecognizer,
    estimate_vertical_shift,
)


class FakeBackend:
    def detect_spatial(self, image):
        return [Detection("revealed_icon", 0.95, (60, 120, 180, 240))]

    def detect_attributes(self, image):
        return [Detection("full_shape_gold", 0.90, (60, 120, 180, 240))]

    def classify(self, crops):
        return [[("C001", 0.97), ("C002", 0.02)] for _ in crops]


class VisionTests(unittest.TestCase):
    def test_fuses_spatial_quality_and_identity_in_memory(self) -> None:
        catalog = (
            CatalogItem("C001", "catalog_C001", "金", 2, 2, 100),
            CatalogItem("C002", "catalog_C002", "蓝", 2, 2, 90),
        )
        recognizer = VisionRecognizer(FakeBackend(), catalog)
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
        items = recognizer.recognize(frame, offset_rows=5)
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.key, (7, 1))
        self.assertEqual((item.width, item.height), (2, 2))
        self.assertEqual(item.quality, "金")
        self.assertEqual(item.catalog_id, "C001")
        self.assertEqual(item.identity_candidates[0][0], "C001")

    def test_estimates_vertical_overlap_without_writing_a_mosaic(self) -> None:
        rng = np.random.default_rng(20260804)
        mosaic = rng.integers(0, 256, (760, 600, 3), dtype=np.uint8)
        previous = mosaic[:600]
        current = mosaic[120:720]
        shift, _method, confidence, _matches = estimate_vertical_shift(
            previous, current
        )
        self.assertLessEqual(abs(shift - 120), 2)
        self.assertGreaterEqual(confidence, 0.25)


if __name__ == "__main__":
    unittest.main()
