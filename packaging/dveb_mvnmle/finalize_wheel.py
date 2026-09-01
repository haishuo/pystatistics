#!/usr/bin/env python3
"""Refresh the DVEB artifact manifest after auditwheel repairs the ELF."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ARTIFACT = Path("pystatistics/mvnmle/_dveb/artifacts/mvnmle_cpu_abi_v1.so")
MANIFEST = Path("pystatistics/mvnmle/_dveb/artifacts/manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_hash(path: Path) -> str:
    raw = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def write_record(root: Path, record: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != record):
        rows.append(
            (path.relative_to(root).as_posix(), record_hash(path), str(path.stat().st_size))
        )
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dveb-wheel-finalize-") as scratch:
        root = Path(scratch)
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(root)

        artifact = root / ARTIFACT
        manifest_path = root / MANIFEST
        if not artifact.is_file() or not manifest_path.is_file():
            raise SystemExit("repaired wheel is missing the DVEB artifact or manifest")

        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "artifact_sha256": digest(artifact),
                "artifact_stage": "auditwheel-repaired-final-wheel",
                "auditwheel_input": args.wheel.name,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        records = list(root.glob("*.dist-info/RECORD"))
        if len(records) != 1:
            raise SystemExit(f"expected exactly one wheel RECORD, found {len(records)}")
        write_record(root, records[0])

        destination = args.output_dir / args.wheel.name
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(root).as_posix())

    print(
        json.dumps(
            {
                "wheel": str(destination),
                "wheel_sha256": digest(destination),
                "artifact_sha256": manifest["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
