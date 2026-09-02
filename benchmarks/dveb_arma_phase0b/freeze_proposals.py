#!/usr/bin/env python3
"""Freeze deterministic L3 proposal identities before calibration or timing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from common import CELLS, sha256_array
from proposal_trace import PROPOSALS, SERIES_BUCKETS, proposal_trace, trace_seed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "proposal-freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cells = {}
    for cell_id in CELLS:
        phi, loading, minimum_root = proposal_trace(cell_id)
        cells[cell_id] = {
            "seed": trace_seed(cell_id),
            "shape": list(phi.shape),
            "phi_sha256": sha256_array(phi),
            "loading_sha256": sha256_array(loading),
            "minimum_ar_root_modulus": minimum_root,
        }
        print(cell_id, cells[cell_id])
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.proposal-freeze.v1",
        "status": "frozen-before-calibration-or-timing",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "proposal_count": PROPOSALS,
        "series_buckets": SERIES_BUCKETS,
        "generator_sha256": sha256(HERE / "proposal_trace.py"),
        "cells": cells,
        "decision_observations": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
