from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from auction_moment_assistant.capture import StaticFrameSource
from auction_moment_assistant.observations import (
    EventObservation,
    MapObservation,
    ObservationStore,
)
from auction_moment_assistant.ocr import OcrCapture
from auction_moment_assistant.runtime import CapturePipeline
from auction_moment_assistant.runtime import is_final_result


class FakeOCR:
    def recognize(self, image, *, expected_round):
        return OcrCapture(
            round_number=expected_round,
            round_verified=True,
            events=(
                EventObservation(
                    expected_round,
                    "public",
                    "显示本局藏品总数量20",
                    0.99,
                    {
                        "parsed": True,
                        "effect": "total_item_count",
                        "observed_count": 20,
                    },
                ),
            ),
            bankroll=123456,
            timing_ms=1.0,
        )


class FakeVision:
    def recognize(self, image, *, offset_rows):
        return (MapObservation(row=offset_rows, column=0, confidence=0.9),)


class RuntimePrivacyTests(unittest.TestCase):
    def test_capture_pipeline_creates_no_session_files(self) -> None:
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        store = ObservationStore()
        pipeline = CapturePipeline(
            source=StaticFrameSource(frame),
            store=store,
            ocr=FakeOCR(),
            vision=FakeVision(),
        )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                before = set(Path(".").rglob("*"))
                outcome = pipeline.capture_once(expected_round=2, offset_rows=4)
                after = set(Path(".").rglob("*"))
            finally:
                os.chdir(previous)
        self.assertEqual(before, after)
        self.assertFalse(outcome.final_detected)
        self.assertEqual(store.snapshot().bankroll, 123456)
        self.assertEqual(store.snapshot().map_items[0]["row"], 4)

    def test_final_signature_clears_without_running_ocr_or_vision(self) -> None:
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
        frame[15:80, 485:665] = 0
        frame[585:700, 835:1270] = (20, 100, 245)
        self.assertTrue(is_final_result(frame))
        store = ObservationStore()
        store.correct_bankroll(9999)
        pipeline = CapturePipeline(
            source=StaticFrameSource(frame),
            store=store,
            ocr=FakeOCR(),
            vision=FakeVision(),
        )
        outcome = pipeline.capture_once(expected_round=5, offset_rows=0)
        self.assertTrue(outcome.final_detected)
        self.assertIsNone(store.snapshot().bankroll)
        self.assertIsNone(store.frame_copy())


if __name__ == "__main__":
    unittest.main()
