from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


SCHEMA_VERSION = "auction-vision-model-release-v1"
EXPECTED_ROLES = {
    "spatial_detector",
    "attribute_detector",
    "identity_classifier",
    "probabilistic_world_model",
}
WINDOWS_PATH = re.compile(
    rb"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?![\\/])|\\\\\?\\)"
)
RAW_SESSION_ID = re.compile(rb"\b\d{8}T\d{6}\b")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_release(path: Path) -> tuple[dict[str, bytes], str]:
    path = Path(path).resolve()
    if path.is_dir():
        files = {
            child.name: child.read_bytes()
            for child in path.iterdir()
            if child.is_file()
        }
        return files, str(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            files = {
                Path(name).name: archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/")
            }
        return files, str(path)
    raise ValueError("--release 必须是 Release 目录或 ZIP")


def verify(path: Path, expected_manifest: Path | None = None) -> dict:
    files, source = read_release(path)
    errors = []
    try:
        manifest = json.loads(files["model-manifest.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"模型清单无效：{exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    roles = {str(record.get("role")) for record in manifest.get("models") or []}
    if roles != EXPECTED_ROLES:
        errors.append(f"model roles mismatch: {sorted(roles)}")
    for record in manifest.get("models") or []:
        filename = str(record.get("filename") or "")
        payload = files.get(filename)
        if payload is None:
            errors.append(f"missing: {filename}")
            continue
        if len(payload) != int(record.get("bytes") or -1):
            errors.append(f"byte count mismatch: {filename}")
        if sha256_bytes(payload) != str(record.get("sha256") or ""):
            errors.append(f"sha256 mismatch: {filename}")
    for required in (
        "MODEL_CARD.md",
        "THIRD_PARTY_NOTICES.md",
        "AGPL-3.0.txt",
        "SHA256SUMS",
    ):
        if required not in files:
            errors.append(f"missing: {required}")
    for filename, payload in files.items():
        if Path(filename).suffix.lower() not in {
            ".md",
            ".json",
            ".txt",
            "",
        } and filename != "SHA256SUMS":
            continue
        if WINDOWS_PATH.search(payload):
            errors.append(f"absolute Windows path: {filename}")
        if RAW_SESSION_ID.search(payload):
            errors.append(f"raw session id: {filename}")
    if expected_manifest is not None:
        expected = json.loads(
            Path(expected_manifest).read_text(encoding="utf-8")
        )
        expected_by_role = {
            str(record["role"]): record for record in expected.get("models") or []
        }
        actual_by_role = {
            str(record["role"]): record for record in manifest.get("models") or []
        }
        for role in EXPECTED_ROLES:
            for field in ("filename", "bytes", "sha256"):
                if actual_by_role.get(role, {}).get(field) != expected_by_role.get(
                    role, {}
                ).get(field):
                    errors.append(f"committed manifest mismatch: {role}.{field}")
    if errors:
        raise ValueError("；".join(errors))
    return {
        "status": "ok",
        "source": source,
        "release": manifest.get("release"),
        "files": len(files),
        "models": len(manifest.get("models") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="独立验证视觉模型 Release")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "vision-release-v1.json",
    )
    args = parser.parse_args()
    try:
        result = verify(args.release, args.expected_manifest)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
