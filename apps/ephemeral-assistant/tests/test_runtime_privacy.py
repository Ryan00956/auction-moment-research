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
from auction_moment_assistant.vision import ViewportRecognition


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


class FailingOCR:
    def recognize(self, image, *, expected_round):
        raise RuntimeError(f"diagnostic failure at R{expected_round}")


class FakeScrollSource:
    def __init__(self, frames):
        self.frames = list(frames)
        self.swipes = []

    def capture(self):
        return self.frames.pop(0).copy()

    def swipe_map(self, start, end, duration_ms):
        self.swipes.append((start, end, duration_ms))


class FakeMultiVision(FakeVision):
    def recognize_viewports(self, frames):
        return ViewportRecognition(
            items=(MapObservation(row=0, column=0, confidence=0.9),),
            offsets_pixels=tuple(0 for _ in frames),
            estimated_rows=12,
            alignment_stable=True,
        )


class TestableScrollPipeline(CapturePipeline):
    def _round_frame(self):
        return self.source.capture()


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

    def test_capture_error_keeps_full_in_memory_traceback(self) -> None:
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        pipeline = CapturePipeline(
            source=StaticFrameSource(frame),
            store=ObservationStore(),
            ocr=FailingOCR(),
            vision=None,
        )
        outcome = pipeline.capture_once(expected_round=3, offset_rows=0)
        self.assertIn("OCR: diagnostic failure at R3", outcome.errors)
        self.assertTrue(outcome.debug)
        self.assertIn("Traceback (most recent call last)", outcome.debug[0])
        self.assertIn("RuntimeError: diagnostic failure at R3", outcome.debug[0])

    def test_round_scan_only_uses_map_swipes_and_creates_no_files(self) -> None:
        first = np.full((720, 1280, 3), 80, dtype=np.uint8)
        second = first.copy()
        second[110:705, 49:647] = 180
        # initial, top-same, down-new, down-same, restore-new, restore-same
        source = FakeScrollSource(
            [first, first, second, second, first, first]
        )
        store = ObservationStore()
        pipeline = TestableScrollPipeline(
            source=source,
            store=store,
            ocr=FakeOCR(),
            vision=FakeMultiVision(),
            scroll_wait=0.05,
        )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                before = set(Path(".").rglob("*"))
                outcome = pipeline.scan_round(expected_round=1)
                after = set(Path(".").rglob("*"))
            finally:
                os.chdir(previous)
        self.assertEqual(before, after)
        self.assertTrue(outcome.success, outcome.errors)
        self.assertEqual(outcome.viewport_count, 2)
        self.assertTrue(source.swipes)
        for start, end, _duration in source.swipes:
            self.assertEqual(start[0], 340)
            self.assertEqual(end[0], 340)
        self.assertEqual(store.snapshot().map_rows_source, "automatic_scroll_estimate")


if __name__ == "__main__":
    unittest.main()
