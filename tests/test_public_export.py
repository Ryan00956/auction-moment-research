from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from auction_moment_research.public_export import export_public_dataset


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class PublicExportTests(unittest.TestCase):
    def test_export_removes_names_paths_and_exact_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "core"
            opponents = root / "opponents"
            destination = root / "public"
            session_id = "20260804T120000"

            write_csv(
                core / "sessions_master.csv",
                ["session_id", "auction_tier", "date", "hour", "raw_session_path"],
                [{
                    "session_id": session_id,
                    "auction_tier": "S",
                    "date": "2026-08-04",
                    "hour": "12",
                    "raw_session_path": r"C:\private\session",
                }],
            )
            write_csv(
                core / "rounds.csv",
                ["session_id", "auction_tier", "round"],
                [{"session_id": session_id, "auction_tier": "S", "round": "1"}],
            )
            write_csv(
                core / "opponent_bids.csv",
                ["session_id", "auction_tier", "round", "status", "name", "player_name", "bid"],
                [{
                    "session_id": session_id,
                    "auction_tier": "S",
                    "round": "1",
                    "status": "complete",
                    "name": "private-opponent",
                    "player_name": "private-self",
                    "bid": "10",
                }],
            )
            write_csv(
                core / "treasures.csv",
                ["session_id", "auction_tier", "catalog_id", "name", "quality", "width", "height", "value"],
                [{
                    "session_id": session_id,
                    "auction_tier": "S",
                    "catalog_id": "1",
                    "name": "private-item-name",
                    "quality": "白",
                    "width": "1",
                    "height": "1",
                    "value": "100",
                }],
            )
            write_csv(core / "event_catalog.csv", ["event_code"], [{"event_code": "101"}])
            write_csv(
                core / "event_instances.csv",
                ["session_id", "event_code", "observed_at", "source_file"],
                [{
                    "session_id": session_id,
                    "event_code": "101",
                    "observed_at": "2026-08-04T12:00:00+08:00",
                    "source_file": r"C:\private\events.jsonl",
                }],
            )
            write_csv(
                core / "event_offers.csv",
                ["session_id", "event_code", "observed_at", "source_file"],
                [{
                    "session_id": session_id,
                    "event_code": "101",
                    "observed_at": "2026-08-04T12:00:00+08:00",
                    "source_file": r"C:\private\events.jsonl",
                }],
            )
            (core / "dataset_manifest.json").write_text("{}", encoding="utf-8")

            opponents.mkdir()
            (opponents / "player_rounds.jsonl").write_text(
                json.dumps({
                    "session_id": session_id,
                    "player_key": "old-player-key",
                    "observed_at": "2026-08-04T12:01:02+08:00",
                    "round": 1,
                    "source": "trusted_protocol_confirmed",
                    "label_bid": 10,
                    "label_relative_to_round_max": 1.0,
                }) + "\n",
                encoding="utf-8",
            )
            (opponents / "player_profiles.jsonl").write_text(
                json.dumps({
                    "player_key": "old-player-key",
                    "first_observed_at": "2026-08-04T12:01:02+08:00",
                    "last_observed_at": "2026-08-04T12:01:02+08:00",
                }) + "\n",
                encoding="utf-8",
            )
            (opponents / "quality.json").write_text(
                json.dumps({"capture_owner_inference": {"owner_player_key": "owner"}}),
                encoding="utf-8",
            )

            report = root / "report.md"
            report.write_text(
                f"session {session_id} at C:\\private\\evidence",
                encoding="utf-8",
            )
            export_public_dataset(
                core_root=core,
                opponent_root=opponents,
                destination=destination,
                secret=b"a-public-export-test-key-with-32-bytes",
                reports=[(report, "report.md")],
                reports_destination=root / "reports",
            )

            published = "\n".join(
                path.read_text(encoding="utf-8")
                for path in destination.rglob("*")
                if path.is_file()
            )
            self.assertNotIn(session_id, published)
            self.assertNotIn("private-opponent", published)
            self.assertNotIn("private-self", published)
            self.assertNotIn("private-item-name", published)
            self.assertNotIn(r"C:\private", published)
            opponent_header = (destination / "core" / "opponent_bids.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("player_key", opponent_header)
            self.assertNotIn("player_name", opponent_header)


if __name__ == "__main__":
    unittest.main()
