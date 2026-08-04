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
                spatial="complete",
            )
        )
        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        item = adapted.decision["visible_items"][0]
        self.assertTrue(item["identity_known"])
        self.assertEqual(item["identity"]["known_value"], 123456)
        self.assertEqual(
            item["identity"]["match_confidence_status"], "human_confirmed"
        )

    def test_preserves_all_five_evidence_states_without_inventing_size(self) -> None:
        store = ObservationStore()
        store.apply_capture(
            round_number=1,
            events=[event(1, "public"), event(1, "personal")],
            bankroll=999,
            map_items=[
                MapObservation(
                    row=0,
                    column=0,
                    marker_width=3,
                    marker_height=2,
                    confidence=0.96,
                    spatial="top_left",
                ),
                MapObservation(
                    row=0,
                    column=1,
                    marker_width=2,
                    marker_height=2,
                    quality="金",
                    confidence=0.96,
                    spatial="top_left",
                ),
                MapObservation(
                    row=0,
                    column=2,
                    width=1,
                    height=1,
                    confidence=0.96,
                    spatial="outline",
                ),
                MapObservation(
                    row=0,
                    column=3,
                    width=1,
                    height=1,
                    quality="金",
                    confidence=0.96,
                    spatial="outline",
                ),
                MapObservation(
                    row=0,
                    column=4,
                    width=2,
                    height=2,
                    quality="金",
                    catalog_id="C001",
                    confidence=0.96,
                    spatial="complete",
                ),
            ],
        )
        store.confirm_complete(True)
        store.confirm_pre_bid(True)
        store.correct_map_extent(10, True)

        adapted = OcrDecisionAdapter(CATALOG).adapt(store.snapshot())
        self.assertTrue(adapted.accepted, adapted.issues)
        items = {
            value["position"]["column"]: value
            for value in adapted.decision["visible_items"]
        }
        self.assertIsNone(items[0]["known_size"])
        self.assertEqual(items[0]["spatial_knowledge"], "top_left")
        self.assertIsNone(items[0]["quality"])
        self.assertIsNone(items[1]["known_size"])
        self.assertEqual(items[1]["quality"], "金")
        self.assertEqual(items[2]["known_size"], {"width": 1, "height": 1})
        self.assertEqual(items[2]["spatial_knowledge"], "outline")
        self.assertIsNone(items[2]["quality"])
        self.assertEqual(items[3]["quality"], "金")
        self.assertTrue(items[4]["identity_known"])
        self.assertEqual(
            items[4]["identity"]["match_confidence_status"],
            "ocr_thresholded",
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
