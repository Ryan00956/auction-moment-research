from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = REPOSITORY_ROOT / "apps" / "ephemeral-assistant" / "src"
ROOT_SOURCE = REPOSITORY_ROOT / "src"
RAW_SESSION_ID = re.compile(r"\b\d{8}T\d{6}\b")
WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\\?\\)")
V6_CANDIDATE_ID = "protocol-round-residual-v6-72a08074c3d8af81"
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


class ReleaseBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_session(value: str, mapping: dict[str, str]) -> str:
    if value not in mapping:
        mapping[value] = f"public-reference-{len(mapping) + 1:04d}"
    return mapping[value]


def _sanitize_string(value: str, sessions: dict[str, str]) -> str:
    result = RAW_SESSION_ID.sub(
        lambda match: _public_session(match.group(0), sessions),
        value,
    )
    if WINDOWS_PATH.search(result):
        return "redacted-private-path"
    return result


def sanitize_object(
    value: Any,
    sessions: dict[str, str],
    memo: dict[int, Any] | None = None,
) -> Any:
    """Remove raw session ids and paths while preserving numeric state."""

    if memo is None:
        memo = {}
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return value
    if isinstance(value, Path):
        return "redacted-private-path"
    if isinstance(value, str):
        return _sanitize_string(value, sessions)
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        memo[identity] = result
        for key, child in value.items():
            result[sanitize_object(key, sessions, memo)] = sanitize_object(
                child, sessions, memo
            )
        return result
    if isinstance(value, list):
        result: list[Any] = []
        memo[identity] = result
        result.extend(sanitize_object(child, sessions, memo) for child in value)
        return result
    if isinstance(value, tuple):
        result = tuple(sanitize_object(child, sessions, memo) for child in value)
        memo[identity] = result
        return result
    if isinstance(value, set):
        result = {sanitize_object(child, sessions, memo) for child in value}
        memo[identity] = result
        return result

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        replacements = {
            field.name: sanitize_object(
                getattr(value, field.name), sessions, memo
            )
            for field in dataclasses.fields(value)
        }
        result = dataclasses.replace(value, **replacements)
        memo[identity] = result
        return result

    module = type(value).__module__
    if module.startswith(("numpy", "scipy", "sklearn", "joblib")):
        memo[identity] = value
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        memo[identity] = value
        for name, child in tuple(attributes.items()):
            setattr(value, name, sanitize_object(child, sessions, memo))
    return value


def _walk_strings(value: Any, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if isinstance(value, (str, Path)):
        yield str(value)
        return
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(key, seen)
            yield from _walk_strings(child, seen)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_strings(child, seen)
    else:
        module = type(value).__module__
        if not module.startswith(("numpy", "scipy", "sklearn", "joblib")):
            for child in (getattr(value, "__dict__", {}) or {}).values():
                yield from _walk_strings(child, seen)


def assert_public_metadata(value: Any) -> None:
    findings = []
    for text in _walk_strings(value):
        if RAW_SESSION_ID.search(text):
            findings.append(f"raw_session_id:{text}")
        if WINDOWS_PATH.search(text):
            findings.append(f"windows_path:{text}")
    if findings:
        raise ReleaseBuildError(
            "sanitized artifact still contains private metadata: "
            + "; ".join(findings[:10])
        )


def canonicalize_tree_padding(value: Any, seen: set[int] | None = None) -> None:
    """Zero non-semantic padding bytes in sklearn tree node arrays."""

    import numpy as np

    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    tree = getattr(value, "tree_", None)
    if tree is not None and hasattr(tree, "__getstate__"):
        state = tree.__getstate__()
        nodes = state.get("nodes")
        if isinstance(nodes, np.ndarray) and nodes.dtype.names:
            canonical = np.zeros(nodes.shape, dtype=nodes.dtype)
            for name in nodes.dtype.names:
                canonical[name] = nodes[name]
            state["nodes"] = canonical
            tree.__setstate__(state)
    if isinstance(value, dict):
        children = (*value.keys(), *value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif isinstance(value, np.ndarray):
        children = value.flat if value.dtype == object else ()
    else:
        children = (getattr(value, "__dict__", {}) or {}).values()
    for child in children:
        canonicalize_tree_padding(child, seen)


def synthetic_decision(round_number: int, map_rows: int) -> dict:
    events = []
    for event_round in range(1, round_number + 1):
        for kind in ("public", "personal"):
            events.append(
                {
                    "round": event_round,
                    "kind": kind,
                    "ocr_confidence": 1.0,
                    "semantics": {
                        "parsed": True,
                        "effect": "total_item_count",
                        "observed_count": 24,
                        "protocol_source": "human_confirmed",
                    },
                }
            )
    return {
        "round": round_number,
        "events": events,
        "visible_items": [],
        "observed_map": {
            "rows": map_rows,
            "columns": 10,
            "height_exact": True,
            "bottom_visible": True,
            "canvas_bottom_visible": True,
            "source": "ocr_visible_map_confirmed",
            "height_proof": {
                "schema_version": "hidden-map-bottom-proof-v1",
                "status": "proven",
                "rows": map_rows,
            },
        },
        "map_recognition": {
            "completeness": {
                "status": "human_reviewed_visible_screen",
                "exact_counts_safe": False,
                "checks": {},
                "exact_quality_counts": {},
                "exact_size_counts": {},
            }
        },
        "native_evidence": {
            "mode": "ocr_visible_pre_bid_v1",
            "pre_bid_proven": True,
            "ocr_used_for_live_features": True,
            "settlement_or_final_truth_in_features": False,
        },
    }


def _numeric_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _numeric_projection(child)
            for key, child in value.items()
            if isinstance(child, (Mapping, list, tuple, int, float, bool))
        }
    if isinstance(value, (list, tuple)):
        return [_numeric_projection(child) for child in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ReleaseBuildError("model returned a non-finite number")
        return round(float(value), 10)
    return None


def sanitize_v6(source: Path, target: Path) -> dict:
    sys.path.insert(0, str(ROOT_SOURCE))
    sys.path.insert(0, str(APP_SOURCE))
    try:
        import joblib

        original = joblib.load(source)
    except Exception as exc:
        raise ReleaseBuildError(f"cannot load source v6 model: {exc}") from exc
    if str(getattr(original, "candidate_id", "")) != V6_CANDIDATE_ID:
        raise ReleaseBuildError("source v6 candidate id mismatch")
    sanitized = copy.deepcopy(original)
    session_mapping: dict[str, str] = {}
    sanitized = sanitize_object(sanitized, session_mapping)
    canonicalize_tree_padding(sanitized)
    assert_public_metadata(sanitized)

    probes = []
    for round_number, rows in ((1, 10), (3, 13), (5, 16)):
        decision = synthetic_decision(round_number, rows)
        before = original.predict(decision)
        after = sanitized.predict(decision)
        if _numeric_projection(before) != _numeric_projection(after):
            raise ReleaseBuildError(
                f"v6 numerical equivalence failed for round {round_number}"
            )
        probes.append(
            {
                "round": round_number,
                "map_rows": rows,
                "prediction": int(after["prediction"]),
                "p10": int(after["p10"]),
                "p90": int(after["p90"]),
            }
        )
    # Uncompressed joblib avoids compressor-level byte drift and remains small.
    joblib.dump(sanitized, target, compress=0)
    reloaded = joblib.load(target)
    assert_public_metadata(reloaded)
    if str(getattr(reloaded, "candidate_id", "")) != V6_CANDIDATE_ID:
        raise ReleaseBuildError("released v6 candidate id mismatch")
    return {
        "filename": target.name,
        "role": "latest_value_model_v6",
        "format": "joblib",
        "candidate_id": V6_CANDIDATE_ID,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "privacy_strings_replaced": len(session_mapping),
        "numerical_equivalence_probes": probes,
        "license": "Apache-2.0",
    }


def sanitize_world_model_v2(
    source: Path,
    target: Path,
    treasures_csv: Path,
) -> dict:
    sys.path.insert(0, str(ROOT_SOURCE))
    sys.path.insert(0, str(APP_SOURCE))
    from auction_moment_assistant.models import load_catalog
    from auction_moment_research.world_model import catalog_semantic_hash
    from generative_world_model_v2 import GenerativeWorldModelV2

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("artifact_sha256", None)
    sessions: dict[str, str] = {}
    payload = sanitize_object(payload, sessions)
    catalog = load_catalog(treasures_csv)
    payload["catalog_semantic_sha256"] = catalog_semantic_hash(catalog)
    provenance = dict(payload.get("training_provenance") or {})
    payload["training_provenance"] = {
        "source_kind": provenance.get("source_kind"),
        "session_count": int(provenance.get("session_count") or 0),
        "privacy_note": "paths and raw session ids removed for public release",
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    assert_public_metadata(payload)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model = GenerativeWorldModelV2.load(target, catalog)
    return {
        "filename": target.name,
        "role": "generative_world_model_v2",
        "format": "json",
        "schema_version": "world-model-v2",
        "artifact_sha256": model.artifact["artifact_sha256"],
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "license": "Apache-2.0",
    }


def deterministic_zip(target: Path, files: list[Path], root: Path) -> None:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            info = zipfile.ZipInfo(
                path.relative_to(root).as_posix(),
                date_time=ZIP_TIMESTAMP,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_release(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ReleaseBuildError(f"output already exists: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="auction-latest-model-") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(args.source_bundle.resolve()) as archive:
            archive.extractall(extracted)
        v6_record = sanitize_v6(
            extracted / "model.joblib",
            output / "latest-value-model-v6.joblib",
        )
        v2_record = sanitize_world_model_v2(
            extracted / "world-model-v2.json",
            output / "world-model-v2-S.json",
            args.treasures.resolve(),
        )

    shutil.copy2(args.model_card.resolve(), output / "LATEST_MODEL_CARD.md")
    shutil.copy2(REPOSITORY_ROOT / "LICENSE", output / "Apache-2.0.txt")
    manifest = {
        "schema_version": "auction-latest-model-release-v1",
        "release": str(args.release),
        "privacy_contract": {
            "contains_raw_images": False,
            "contains_session_records": False,
            "contains_packet_capture_or_protocol_code": False,
            "contains_absolute_training_paths": False,
            "contains_raw_session_ids": False,
        },
        "runtime_contract": {
            "adapter": "ocr-visible-pre-bid-adapter-v1",
            "automatic_bidding": False,
            "actionable": False,
            "packet_equivalent": False,
        },
        "models": [v6_record, v2_record],
    }
    manifest_path = output / "latest-model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        [path for path in output.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    archive = output.parent / f"auction-latest-model-{args.release}.zip"
    deterministic_zip(
        archive,
        [path for path in output.iterdir() if path.is_file()],
        output,
    )
    return {
        "status": "ok",
        "output": str(output),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "models": [v6_record, v2_record],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanitize and package the frozen v6 + world-model-v2 release"
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--treasures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default="latest-model-v6-v2.0.0-beta.4")
    parser.add_argument(
        "--model-card",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "LATEST_MODEL_CARD.md",
    )
    return parser


def main() -> int:
    try:
        result = build_release(build_parser().parse_args())
    except (ReleaseBuildError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
