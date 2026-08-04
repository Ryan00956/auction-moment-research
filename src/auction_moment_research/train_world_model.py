from __future__ import annotations

"""Train the reference hierarchical world model from public treasure rows."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .world_model import ProbabilisticWorldModel


@dataclass(frozen=True)
class CatalogItem:
    catalog_id: str
    name: str
    quality: str
    width: int
    height: int
    value: int


@dataclass(frozen=True)
class TrainingGame:
    session_id: str
    total_count: int
    rows: int
    item_counts: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: str, field: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def load_training_rows(path: Path) -> tuple[list[CatalogItem], list[TrainingGame], str]:
    catalog_rows: dict[str, CatalogItem] = {}
    counts_by_session: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rows_by_session: defaultdict[str, int] = defaultdict(int)
    tiers: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = str(row.get("session_id") or "")
            catalog_id = str(row.get("catalog_id") or "")
            if not session_id or not catalog_id:
                continue
            item = CatalogItem(
                catalog_id=catalog_id,
                name=f"catalog_{catalog_id}",
                quality=str(row["quality"]),
                width=_integer(row["width"], "width"),
                height=_integer(row["height"], "height"),
                value=_integer(row["value"], "value"),
            )
            existing = catalog_rows.get(catalog_id)
            if existing is not None and existing != item:
                raise ValueError(f"catalog metadata conflict: {catalog_id}")
            catalog_rows[catalog_id] = item
            counts_by_session[session_id][catalog_id] += 1
            row_top = _integer(row.get("row", "0"), "row") + item.height
            rows_by_session[session_id] = max(rows_by_session[session_id], row_top)
            if row.get("auction_tier"):
                tiers.add(str(row["auction_tier"]))

    if not catalog_rows or not counts_by_session:
        raise ValueError("public treasure dataset is empty")
    if len(tiers) != 1:
        raise ValueError(f"expected exactly one auction tier, found: {sorted(tiers)}")

    catalog = sorted(catalog_rows.values(), key=lambda item: item.catalog_id)
    games = [
        TrainingGame(
            session_id=session_id,
            total_count=sum(counts_by_session[session_id].values()),
            rows=rows_by_session[session_id],
            item_counts=tuple(counts_by_session[session_id][item.catalog_id] for item in catalog),
        )
        for session_id in sorted(counts_by_session)
    ]
    return catalog, games, next(iter(tiers))


def train(input_path: Path, output_path: Path) -> dict:
    catalog, games, auction_tier = load_training_rows(input_path)
    source_sha256 = sha256_file(input_path)
    model = ProbabilisticWorldModel.fit(
        catalog,
        games,
        auction_tier=auction_tier,
        training_fingerprint=source_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return {
        "schema_version": "public-world-model-training-v1",
        "research_only": True,
        "production_enabled": False,
        "input": {
            "path": input_path.name,
            "sha256": source_sha256,
            "catalog_items": len(catalog),
            "sessions": len(games),
            "auction_tier": auction_tier,
        },
        "output": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the public hierarchical world model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/v1/core/treasures.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/generated/world-model-v1.json"),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = train(args.input.resolve(), args.output.resolve())
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
