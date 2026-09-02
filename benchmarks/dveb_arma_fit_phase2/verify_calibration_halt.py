#!/usr/bin/env python3
"""Offline verifier for the Phase-II calibration correctness halt."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    selection = json.loads((HERE / "calibration-selection.json").read_text())
    valid_path = HERE / selection["valid_correctness_record"]
    assert sha256(valid_path) == selection["valid_correctness_record_gzip_sha256"]
    with gzip.open(valid_path, "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == selection[
            "valid_correctness_record_uncompressed_sha256"
        ]
    with gzip.open(valid_path, "rt") as stream:
        valid = json.load(stream)
    assert valid["pass"] is False
    assert valid["evaluation_cells_observed"] is False
    assert valid["timing_results_exist"] is False
    assert valid["gradient_admitted"] == selection["gradient_admission"]
    for route, passed_cells in selection["complete_fit_cells_passed"].items():
        actual = [
            cell_id for cell_id, cell in valid["cells"].items()
            if cell["routes"][route]["pass"]
        ]
        assert actual == passed_cells, (route, actual, passed_cells)
    assert all(value is None for value in selection["selection"].values())
    assert selection["calibration_timing_performed"] is False
    assert selection["evaluation_cells_observed"] is False
    assert selection["admitted_campaign_manifest_created"] is False
    assert selection["comparative_timing_performed"] is False
    invalid = selection["invalid_attempt_preserved"]
    assert sha256(HERE / invalid["record"]) == invalid["gzip_sha256"]
    print("Phase-II calibration halt evidence: VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
