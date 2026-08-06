from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_public_archive",
    REPOSITORY_ROOT / "scripts" / "build_public_archive.py",
)
verifier = load_module(
    "verify_archive_release",
    REPOSITORY_ROOT / "scripts" / "verify_archive_release.py",
)


class ArchiveReleaseTests(unittest.TestCase):
    def test_deterministic_zip_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("alpha\n", encoding="utf-8")
            second.write_text("beta\n", encoding="utf-8")
            entries = [(second, "b/second.txt"), (first, "a/first.txt")]
            one = root / "one.zip"
            two = root / "two.zip"
            builder.write_deterministic_zip(one, entries)
            builder.write_deterministic_zip(two, reversed(entries))
            self.assertEqual(one.read_bytes(), two.read_bytes())
            with zipfile.ZipFile(one) as archive:
                self.assertEqual(archive.namelist(), ["a/first.txt", "b/second.txt"])

    def test_verifier_rejects_promoted_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in verifier.REQUIRED_FILES - {
                verifier.RELEASE_MANIFEST,
                verifier.CHECKSUMS,
            }:
                (root / filename).write_bytes(filename.encode("utf-8"))
            payload_path = root / "STATUS.md"
            manifest = {
                "archive_id": verifier.ARCHIVE_ID,
                "project_status": "hibernating",
                "source_commit": "0" * 40,
                "capability_boundary": {
                    "formal_full_match_end_to_end_completed": True,
                    "fresh_prospective_validation_completed": False,
                    "production_enabled": False,
                    "automatic_bid_execution_in_public_project": False,
                    "all_assistant_outputs_actionable": False,
                },
                "payloads": [
                    {
                        "filename": payload_path.name,
                        "bytes": payload_path.stat().st_size,
                        "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                    }
                ],
            }
            (root / verifier.RELEASE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            checksummed = [
                path for path in root.iterdir() if path.name != verifier.CHECKSUMS
            ]
            (root / verifier.CHECKSUMS).write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in sorted(checksummed)
                ),
                encoding="utf-8",
            )
            errors = verifier.verify_bundle(root)
            self.assertIn(
                "capability boundary must keep formal_full_match_end_to_end_completed=false",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
