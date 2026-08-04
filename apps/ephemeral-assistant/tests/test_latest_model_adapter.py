from __future__ import annotations

import unittest

from auction_moment_assistant.latest_model import OcrDecisionAdapter
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import (
    EventObservation,
    MapObservation,
    ObservationStore,
)


CATALOG = (
    CatalogItem("C001", "catalog_C001", "金", 2, 2, 123456),
)


def event(round_number: int, kind: str, confidence: float = 0.95):
    return EventObservation(
        round_number,
        kind,
        "显示本局藏品总数量24",
        confidence,
        {
            "parsed": True,
            "effect": "total_item_count",
            "observed_count": 24,
        },
    )


class OcrDecisionAdapterTests(unittest.TestCase):
    def reviewed_store(self) -> ObservationStore:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[
                MapObservation(
                    row=0,
                    column=0,
                    width=2,
                    height=2,
                    quality="金",
                    catalog_id="C001",
                    confidence=0.96,
                )
            ],
        )
        store.confirm_complete(True)
        store.confirm_pre_bid(True)
        store.correct_map_extent(10, True)
        return store

    def test_translates_reviewed_ocr_without_inventing_packet_fields(self) -> None:
        adapted = OcrDecisionAdapter(CATALOG).adapt(
            self.reviewed_store().snapshot()
        )
        self.assertTrue(adapted.accepted, adapted.issues)
        decision = adapted.decision
        self.assertEqual(
            decision["native_evidence"]["mode"], "ocr_visible_pre_bid_v1"
        )
        self.assertNotIn("packet", decision)
        self.assertNotIn("settlement", decision)
        self.assertNotIn("item_index", decision["visible_items"][0])
        self.assertEqual(
            decision["events"][0]["semantics"]["protocol_source"],
            "ocr_thresholded",
        )
        self.assertFalse(decision["visible_items"][0]["identity_known"])
        self.assertEqual(decision["visible_items"][0]["identity"], {})

    def test_manual_identity_becomes_exact_human_evidence(self) -> None:
        store = self.reviewed_store()
        store.upsert_map_item(
            MapObservation(
                row=0,
                column=0,
                width=2,
                height=2,
                quality="金",
                catalog_id="C001",
            )
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        item = adapted.decision["visible_items"][0]
        self.assertTrue(item["identity_known"])
        self.assertEqual(item["identity"]["known_value"], 123456)
        self.assertEqual(
            item["identity"]["match_confidence_status"], "human_confirmed"
        )

    def test_blocks_missing_proofs_and_low_confidence_event(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public", 0.5), event(1, "personal")],
            bankroll=None,
            map_items=[],
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertFalse(adapted.accepted)
        self.assertIn("pre_bid_not_confirmed", adapted.issues)
        self.assertIn("exact_map_rows_not_confirmed", adapted.issues)
        self.assertIn("visible_map_not_reviewed", adapted.issues)
        self.assertIn("low_confidence_event:R1:public", adapted.issues)

    def test_auto_scroll_estimate_runs_with_explicit_warnings(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[],
        )
        store.apply_automatic_scan_proof(
            round_number=1, map_rows=12, complete=True, pre_bid=True
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertTrue(adapted.accepted, adapted.issues)
        self.assertIn("map_rows_auto_estimated", adapted.warnings)
        self.assertEqual(
            adapted.decision["observed_map"]["source"],
            "ocr_visible_map_estimated",
        )
        self.assertEqual(
            adapted.decision["observed_map"]["height_proof"]["status"],
            "estimated",
        )


if __name__ == "__main__":
    unittest.main()
