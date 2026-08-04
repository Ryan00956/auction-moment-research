from __future__ import annotations

from importlib import resources
import unittest
from pathlib import Path

from auction_moment_assistant.models import load_catalog


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TREASURES = REPOSITORY_ROOT / "data" / "v1" / "core" / "treasures.csv"


class CatalogPresentationTests(unittest.TestCase):
    def test_all_public_catalog_items_have_names_and_previews(self) -> None:
        catalog = load_catalog(TREASURES)
        self.assertEqual(len(catalog), 120)
        self.assertEqual(catalog[0].catalog_id, "C000")
        self.assertEqual(catalog[0].label, "打印机墨盒")
        self.assertEqual(catalog[0].name, "catalog_C000")

        preview_root = resources.files("auction_moment_assistant").joinpath(
            "catalog_previews"
        )
        missing = [
            item.catalog_id
            for item in catalog
            if not preview_root.joinpath(f"{item.catalog_id}.png").is_file()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
