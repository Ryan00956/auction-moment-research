from __future__ import annotations

"""Build the text-only public dataset without copying private evidence."""

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path


SESSION_PATTERN = re.compile(r"\b\d{8}T\d{6}\b")
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\\?\\)[^`\r\n]+")
PUBLIC_DATASET_VERSION = "auction-moment-public-v1"

SESSION_COLUMNS = (
    "session_id",
    "auction_tier",
    "date",
    "hour",
    "continuous_game_number",
    "mode",
    "schema_version",
    "duration_seconds",
    "termination_reason",
    "round_count_observed",
    "parse_quality_status",
    "parse_limitation_count",
    "protocol_status",
    "protocol_parser_version",
    "protocol_event_count",
    "protocol_valid_event_count",
    "protocol_unknown_event_count",
    "protocol_schema_mismatch_count",
    "protocol_settlement_valid",
    "settlement_total",
    "settlement_confidence",
    "treasure_count",
    "detected_value_sum",
    "value_sum_difference",
    "value_sum_matches",
    "catalog_assignment_found",
    "catalog_assignment_mode",
    "catalog_independent_of_settlement_total",
    "geometry_source",
    "geometry_settlement_independent",
    "final_occupied_cells",
    "identity_review_item_count",
    "clue_quality_conflict_count",
    "treasure_blue_count",
    "treasure_purple_count",
    "treasure_gold_count",
    "treasure_other_quality_count",
    "treasure_1x1_count",
    "treasure_1x2_count",
    "treasure_2x1_count",
    "treasure_2x2_count",
    "treasure_other_size_count",
    "treasure_value_min",
    "treasure_value_median",
    "treasure_value_mean",
    "treasure_value_max",
    "postgame_review_status",
    "postgame_value_gate_status",
    "postgame_value_gate_matches",
    "final_treasure_review_status",
)

ROUND_COLUMNS = (
    "session_id",
    "auction_tier",
    "round",
    "event_choice",
    "own_bid",
    "bankroll",
    "initial_bankroll",
    "rounds_remaining",
    "opponent_bid_count",
    "opponent_bid_min",
    "opponent_bid_median",
    "opponent_bid_mean",
    "opponent_bid_max",
    "opponent_bid_sum",
    "imputed_zero_row_count",
    "alignment_status",
    "alignment_confidence",
    "alignment_observed_round",
    "map_status",
    "map_exact_counts_safe",
    "map_track_count",
    "map_rows",
    "map_height_exact",
    "map_canvas_height_exact",
    "map_event_hint_count",
    "map_protocol_matched_item_count",
    "map_protocol_unmatched_item_count",
    "map_protocol_conflict_count",
    "map_protocol_usable_for_live_inference",
)

OPPONENT_COLUMNS = (
    "session_id",
    "auction_tier",
    "round",
    "status",
    "player_key",
    "rank",
    "bid",
    "name_confidence",
    "bid_confidence",
    "is_self",
    "is_imputed_zero",
)

TREASURE_DROPPED_COLUMNS = {
    "created_at",
    "name",
    *(f"round_{number}_identity" for number in range(1, 6)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_key(secret: bytes, namespace: str, value: str) -> str:
    digest = hmac.new(
        secret,
        f"{namespace}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{namespace}_{digest[:16]}"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def collect_session_ids(core_root: Path, opponent_root: Path) -> set[str]:
    result: set[str] = set()
    for path in core_root.glob("*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "session_id" not in (reader.fieldnames or []):
                continue
            for row in reader:
                value = str(row.get("session_id") or "")
                if value:
                    result.add(value)
    rounds_path = opponent_root / "player_rounds.jsonl"
    if rounds_path.is_file():
        for row in read_jsonl(rounds_path):
            value = str(row.get("session_id") or "")
            if value:
                result.add(value)
    return result


def csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def export_csv(
    source: Path,
    destination: Path,
    columns: Iterable[str],
    transform: Callable[[dict], dict],
) -> dict:
    source_columns = set(csv_fieldnames(source))
    selected = [column for column in columns if column in source_columns or column == "player_key"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as reader_handle:
        reader = csv.DictReader(reader_handle)
        with destination.open("w", encoding="utf-8", newline="") as writer_handle:
            writer = csv.DictWriter(writer_handle, fieldnames=selected, lineterminator="\n")
            writer.writeheader()
            for source_row in reader:
                transformed = transform(dict(source_row))
                writer.writerow({column: transformed.get(column, "") for column in selected})
                rows += 1
    return {
        "rows": rows,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "columns": selected,
    }


def sanitize_markdown(
    text: str,
    session_ids: Mapping[str, str],
) -> str:
    def replace_session(match: re.Match[str]) -> str:
        return session_ids.get(match.group(0), "session_redacted")

    text = SESSION_PATTERN.sub(replace_session, text)
    text = text.replace("data/raw/sessions/", "<private-source>/sessions/")
    text = text.replace("data\\raw\\sessions\\", "<private-source>\\sessions\\")
    text = ABSOLUTE_WINDOWS_PATH.sub("<private-absolute-path>", text)
    return text


def export_public_dataset(
    *,
    core_root: Path,
    opponent_root: Path,
    destination: Path,
    secret: bytes,
    reports: Iterable[tuple[Path, str]] = (),
    reports_destination: Path | None = None,
    overwrite: bool = False,
) -> dict:
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise RuntimeError(f"destination is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    session_ids = {
        session_id: public_key(secret, "session", session_id)
        for session_id in collect_session_ids(core_root, opponent_root)
    }

    def base_transform(row: dict) -> dict:
        session_id = str(row.get("session_id") or "")
        if session_id:
            row["session_id"] = session_ids[session_id]
        return row

    files: dict[str, dict] = {}
    core_destination = destination / "core"
    files["core/sessions.csv"] = export_csv(
        core_root / "sessions_master.csv",
        core_destination / "sessions.csv",
        SESSION_COLUMNS,
        base_transform,
    )
    files["core/rounds.csv"] = export_csv(
        core_root / "rounds.csv",
        core_destination / "rounds.csv",
        ROUND_COLUMNS,
        base_transform,
    )

    def opponent_bid_transform(row: dict) -> dict:
        base_transform(row)
        name = str(row.pop("name", "") or "")
        row["player_key"] = public_key(secret, "player", name) if name else ""
        return row

    files["core/opponent_bids.csv"] = export_csv(
        core_root / "opponent_bids.csv",
        core_destination / "opponent_bids.csv",
        OPPONENT_COLUMNS,
        opponent_bid_transform,
    )

    treasure_source = core_root / "treasures.csv"
    treasure_columns = [
        column
        for column in csv_fieldnames(treasure_source)
        if column not in TREASURE_DROPPED_COLUMNS
    ]
    files["core/treasures.csv"] = export_csv(
        treasure_source,
        core_destination / "treasures.csv",
        treasure_columns,
        base_transform,
    )

    for source_name, destination_name, dropped in (
        ("event_catalog.csv", "event_catalog.csv", set()),
        ("event_instances.csv", "event_instances.csv", {"observed_at", "source_file"}),
        ("event_offers.csv", "event_offers.csv", {"observed_at", "source_file"}),
    ):
        source = core_root / source_name
        columns = [column for column in csv_fieldnames(source) if column not in dropped]
        files[f"core/{destination_name}"] = export_csv(
            source,
            core_destination / destination_name,
            columns,
            base_transform,
        )

    public_opponent_root = destination / "opponents"
    player_rounds = []
    for row in read_jsonl(opponent_root / "player_rounds.jsonl"):
        row["session_id"] = session_ids[str(row["session_id"])]
        row["player_key"] = public_key(secret, "player", str(row["player_key"]))
        row["observed_at"] = str(row.get("observed_at") or "")[:10]
        player_rounds.append(row)
    player_round_path = public_opponent_root / "player_rounds.jsonl"
    player_round_count = write_jsonl(player_round_path, player_rounds)

    profiles = []
    for row in read_jsonl(opponent_root / "player_profiles.jsonl"):
        row["player_key"] = public_key(secret, "player", str(row["player_key"]))
        row["first_observed_at"] = str(row.get("first_observed_at") or "")[:10]
        row["last_observed_at"] = str(row.get("last_observed_at") or "")[:10]
        profiles.append(row)
    profile_path = public_opponent_root / "player_profiles.jsonl"
    profile_count = write_jsonl(profile_path, profiles)

    quality = json.loads((opponent_root / "quality.json").read_text(encoding="utf-8"))
    owner = quality.get("capture_owner_inference", {}).get("owner_player_key")
    if owner:
        quality["capture_owner_inference"]["owner_player_key"] = "excluded_capture_owner"
    quality_path = public_opponent_root / "quality.json"
    write_json(quality_path, quality)

    opponent_manifest = {
        "schema_version": "opponent-behavior-public-v1",
        "research_only": True,
        "production_enabled": False,
        "advisory_enabled": False,
        "auto_bid_enabled": False,
        "privacy": {
            "session_ids": "HMAC-SHA256 pseudonyms; secret not published",
            "player_keys": "re-keyed HMAC-SHA256 pseudonyms; original IDs and keys not published",
            "timestamps": "calendar date only",
        },
        "quality": quality,
        "outputs": {
            "player_rounds.jsonl": {
                "rows": player_round_count,
                "bytes": player_round_path.stat().st_size,
                "sha256": sha256_file(player_round_path),
            },
            "player_profiles.jsonl": {
                "rows": profile_count,
                "bytes": profile_path.stat().st_size,
                "sha256": sha256_file(profile_path),
            },
            "quality.json": {
                "bytes": quality_path.stat().st_size,
                "sha256": sha256_file(quality_path),
            },
        },
    }
    opponent_manifest_path = public_opponent_root / "manifest.json"
    write_json(opponent_manifest_path, opponent_manifest)

    for relative_path, metadata in (
        ("opponents/player_rounds.jsonl", opponent_manifest["outputs"]["player_rounds.jsonl"]),
        ("opponents/player_profiles.jsonl", opponent_manifest["outputs"]["player_profiles.jsonl"]),
        ("opponents/quality.json", opponent_manifest["outputs"]["quality.json"]),
        (
            "opponents/manifest.json",
            {
                "bytes": opponent_manifest_path.stat().st_size,
                "sha256": sha256_file(opponent_manifest_path),
            },
        ),
    ):
        files[relative_path] = metadata

    report_files: dict[str, dict] = {}
    if reports_destination is not None:
        reports_destination.mkdir(parents=True, exist_ok=True)
        for source, name in reports:
            destination_path = reports_destination / name
            cleaned = sanitize_markdown(source.read_text(encoding="utf-8"), session_ids)
            destination_path.write_text(cleaned, encoding="utf-8", newline="\n")
            report_files[name] = {
                "bytes": destination_path.stat().st_size,
                "sha256": sha256_file(destination_path),
            }

    source_manifest = core_root / "dataset_manifest.json"
    manifest = {
        "schema_version": PUBLIC_DATASET_VERSION,
        "research_only": True,
        "source_dataset_manifest_sha256": sha256_file(source_manifest),
        "privacy": {
            "session_ids": "HMAC-SHA256 pseudonyms; secret not published",
            "player_names": "removed and replaced with re-keyed pseudonyms",
            "timestamps": "exact timestamps removed; session date/hour retained in sessions.csv and date retained in opponent rows",
            "paths": "absolute paths and source-file paths removed",
            "excluded": [
                "screenshots and video",
                "packet captures",
                "input action telemetry",
                "UI state telemetry",
                "model weights",
                "third-party binaries and decrypted scripts",
            ],
        },
        "session_count": len(session_ids),
        "files": files,
        "reports": report_files,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the text-only public research dataset")
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--opponent-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reports-destination", type=Path)
    parser.add_argument(
        "--report",
        action="append",
        nargs=2,
        metavar=("SOURCE", "PUBLIC_NAME"),
        default=[],
    )
    parser.add_argument("--key-env", default="AUCTION_PUBLIC_EXPORT_KEY")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    secret_text = os.environ.get(args.key_env, "")
    if len(secret_text) < 32:
        raise SystemExit(f"{args.key_env} must contain at least 32 characters")
    manifest = export_public_dataset(
        core_root=args.core_root.resolve(),
        opponent_root=args.opponent_root.resolve(),
        destination=args.destination.resolve(),
        secret=secret_text.encode("utf-8"),
        reports=[(Path(source).resolve(), name) for source, name in args.report],
        reports_destination=(
            args.reports_destination.resolve() if args.reports_destination else None
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "schema_version": manifest["schema_version"],
        "session_count": manifest["session_count"],
        "file_count": len(manifest["files"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
