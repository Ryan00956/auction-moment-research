from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from auction_moment_research.train_world_model import load_training_rows, train


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class WorldModelTrainingTests(unittest.TestCase):
    def test_public_dataset_rebuilds_expected_model(self) -> None:
        input_path = REPOSITORY_ROOT / "data" / "v1" / "core" / "treasures.csv"
        catalog, games, tier = load_training_rows(input_path)
        self.assertEqual(len(catalog), 120)
        self.assertEqual(len(games), 1794)
        self.assertEqual(tier, "S")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "world-model-v1.json"
            manifest = train(input_path, output)
            expected = json.loads(
                (REPOSITORY_ROOT / "results" / "world-model-training-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["input"], expected["input"])
            self.assertEqual(manifest["output"]["sha256"], expected["output"]["sha256"])
            self.assertEqual(manifest["output"]["bytes"], expected["output"]["bytes"])


if __name__ == "__main__":
    unittest.main()
