from __future__ import annotations

import unittest

import numpy as np

from auction_moment_assistant.observations import (
    EventObservation,
    MapObservation,
    ObservationStore,
)


class ObservationStoreTests(unittest.TestCase):
    def test_human_event_is_not_overwritten_by_later_ocr(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[
                EventObservation(
                    1, "public", "OCR-A", 0.9, {"parsed": False}
                )
            ],
            bankroll=1000,
            map_items=[],
        )
        store.correct_event(
            1,
            "public",
            "人工值",
            {"parsed": True, "effect": "total_item_count", "observed_count": 20},
        )
        store.apply_capture(
            round_number=1,
            events=[
                EventObservation(
                    1, "public", "OCR-B", 0.99, {"parsed": False}
                )
            ],
            bankroll=2000,
            map_items=[],
        )
        event = store.snapshot().events[0]
        self.assertEqual(event["text"], "人工值")
        self.assertEqual(event["suggested_text"], "OCR-B")
        self.assertTrue(event["human_locked"])

    def test_manual_delete_suppresses_later_vision_candidate(self) -> None:
        store = ObservationStore()
        item = MapObservation(row=2, column=3, confidence=0.9)
        store.apply_capture(
            round_number=1, events=[], bankroll=None, map_items=[item]
        )
        store.remove_map_item(2, 3)
        store.apply_capture(
            round_number=1, events=[], bankroll=None, map_items=[item]
        )
        self.assertEqual(store.snapshot().map_items, ())

    def test_reset_releases_owned_frame_and_session_state(self) -> None:
        store = ObservationStore()
        frame = np.full((10, 10, 3), 42, dtype=np.uint8)
        store.set_frame(frame)
        store.correct_bankroll(1234)
        store.upsert_map_item(MapObservation(row=0, column=0))
        store.reset()
        self.assertIsNone(store.frame_copy())
        snapshot = store.snapshot()
        self.assertIsNone(snapshot.bankroll)
        self.assertEqual(snapshot.map_items, ())
        self.assertTrue(np.all(frame == 42), "store must own its frame copy")


if __name__ == "__main__":
    unittest.main()
