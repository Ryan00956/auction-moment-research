from __future__ import annotations

import unittest

from auction_moment_assistant.map_editor import (
    FULL_SHAPE_NO_QUALITY_COLOR,
    IDENTITY_KNOWN_COLOR,
    TOP_LEFT_NO_QUALITY_COLOR,
    display_geometry,
    evidence_color,
    evidence_label,
    evidence_state,
)
from auction_moment_assistant.observations import MapObservation


class MapEditorStateTests(unittest.TestCase):
    def test_original_thirteen_state_semantics(self) -> None:
        top_left = MapObservation(
            0, 0, marker_width=3, marker_height=2, spatial="top_left"
        )
        self.assertEqual(evidence_state(top_left), "top_left")
        self.assertEqual(evidence_label(top_left), "左上·无品质")
        self.assertEqual(evidence_color(top_left), TOP_LEFT_NO_QUALITY_COLOR)
        self.assertEqual(display_geometry(top_left), (3, 2))
        self.assertIsNone(top_left.width)

        full_shape = MapObservation(
            0, 0, width=2, height=3, spatial="outline"
        )
        self.assertEqual(evidence_state(full_shape), "full_shape")
        self.assertEqual(evidence_label(full_shape), "形状·无品质")
        self.assertEqual(evidence_color(full_shape), FULL_SHAPE_NO_QUALITY_COLOR)

        for quality in ("白", "蓝", "紫", "金", "彩"):
            top_left.quality = quality
            full_shape.quality = quality
            self.assertEqual(evidence_label(top_left), f"左上·{quality}")
            self.assertEqual(evidence_label(full_shape), f"形状·{quality}")

        identity = MapObservation(
            0,
            0,
            width=2,
            height=2,
            quality="金",
            catalog_id="C001",
            spatial="complete",
        )
        self.assertEqual(evidence_state(identity), "identity_known")
        self.assertEqual(evidence_label(identity), "身份·C001")
        self.assertEqual(evidence_color(identity), IDENTITY_KNOWN_COLOR)


if __name__ == "__main__":
    unittest.main()
