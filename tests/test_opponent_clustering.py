from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auction_moment_research.opponent_clustering import (
    StudyConfig,
    StudyError,
    _input_data_quality,
    full_trajectory_features,
    prefix_features,
    run_study,
)


PATTERNS = (
    (0.0, 0.0, 0.1, 0.0, 0.0),
    (0.2, 0.3, 0.2, 0.4, 0.3),
    (0.85, 0.9, 0.9, 0.85, 0.9),
    (0.9, 0.8, 0.5, 0.2, 0.1),
    (0.2, 0.3, 0.5, 0.8, 0.9),
)


def make_rows(date: str, count: int) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        pattern = PATTERNS[index % len(PATTERNS)]
        adjustment = (index % 3) * 0.01
        session_id = f"{date.replace('-', '')}-{index:03d}"
        player_key = f"synthetic-player-{index % 9:02d}"
        for round_number, base in enumerate(pattern, start=1):
            relative = 0.0 if base == 0 else min(1.0, base + adjustment)
            bid = 0 if relative == 0 else round(relative * 1_000_000)
            rows.append({
                "player_key": player_key,
                "session_id": session_id,
                "round": round_number,
                "observed_at": f"{date}T12:{index % 60:02d}:00+08:00",
                "source": "trusted_protocol_confirmed",
                "label_bid": bid,
                "label_bid_state": "explicit_zero" if bid == 0 else "positive",
                "label_relative_to_round_max": relative,
            })
    return rows


class OpponentBehaviorClusteringStudyTests(unittest.TestCase):
    def _research_root(self, root: Path) -> Path:
        research_root = root / "research"
        research_root.mkdir()
        (research_root / "manifest.json").write_text(
            json.dumps({"research_only": True, "production_enabled": False}),
            encoding="utf-8",
        )
        rows = make_rows("2026-07-30", 50) + make_rows("2026-08-02", 30)
        (research_root / "player_rounds.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return research_root

    def test_features_have_the_declared_information_boundary(self) -> None:
        group = make_rows("2026-07-30", 1)
        self.assertEqual(prefix_features(group), [0.0, 0.0])
        self.assertEqual(len(full_trajectory_features(group)), 7)

    def test_rejects_inconsistent_explicit_zero_semantics(self) -> None:
        row = make_rows("2026-07-30", 1)[0]
        row["label_bid_state"] = "positive"
        with self.assertRaises(StudyError):
            _input_data_quality([row])

    def test_generates_research_only_time_ordered_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            manifest = run_study(StudyConfig(
                research_root=self._research_root(root),
                output_root=output_root,
                bootstrap_replicates=8,
                stability_replicates=3,
            ))
            summary = manifest["summary"]
            self.assertFalse(manifest["production_enabled"])
            self.assertFalse(manifest["advisory_enabled"])
            self.assertFalse(manifest["auto_bid_enabled"])
            self.assertEqual(summary["cohort"]["train_five_round_trajectories"], 50)
            self.assertEqual(summary["cohort"]["holdout_five_round_trajectories"], 30)
            self.assertIn(
                summary["full_trajectory_clustering"]["model_selection"]["selected_cluster_count"],
                range(2, 7),
            )
            self.assertIn(
                summary["two_round_prefix_clustering"]["model_selection"]["selected_cluster_count"],
                range(2, 7),
            )
            self.assertEqual(
                summary["two_round_prefix_clustering"]["feature_contract"]
                ["future_rounds_used_as_features"],
                False,
            )
            self.assertTrue((output_root / "data_quality.json").is_file())
            scores = (output_root / "holdout_round5_scores.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("player_key", scores)
            self.assertNotIn("synthetic-player", scores)


if __name__ == "__main__":
    unittest.main()
