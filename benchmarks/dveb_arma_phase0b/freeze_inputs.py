#!/usr/bin/env python3
"""Freeze exact coefficients, deterministic inputs, and source identities."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from common import BURN_IN, CELLS, FAMILIES, MASTER_SEED, input_record

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "input-freeze.json"
PROTOCOL_COMMIT = "bb8b84cc0b79dc339190779e33044d82a60429cd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def main() -> int:
    families = {}
    for family_id, family in FAMILIES.items():
        record = {
            "phi": list(family.phi),
            "noise_loading": list(family.noise_loading),
            "r": family.r,
        }
        matching_cell = next(
            cell_id for cell_id, cell in CELLS.items() if cell.family_id == family_id
        )
        minimum = input_record(matching_cell)["minimum_ar_root_modulus"]
        if minimum < 1.05:
            raise SystemExit(f"{family_id} violates root floor: {minimum}")
        record["minimum_ar_root_modulus"] = minimum
        families[family_id] = record

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.input-freeze.v1",
        "status": "frozen-before-comparator-execution",
        "master_seed": MASTER_SEED,
        "generator": "numpy.random.PCG64DXSM",
        "burn_in": BURN_IN,
        "source": {
            "branch": git_value("branch", "--show-current"),
            "protocol_commit": PROTOCOL_COMMIT,
            "generation_parent_commit": git_value("rev-parse", "HEAD"),
            "protocol_sha256": sha256(ROOT / "docs/research/DVEB_ARMA_PHASE0B_PROTOCOL.md"),
            "common_sha256": sha256(HERE / "common.py"),
            "freeze_script_sha256": sha256(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "families": families,
        "cells": {cell_id: input_record(cell_id) for cell_id in CELLS},
        "decision_observations": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "cells": len(CELLS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
