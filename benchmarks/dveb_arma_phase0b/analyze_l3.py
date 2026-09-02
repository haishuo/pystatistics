#!/usr/bin/env python3
"""Apply the frozen Phase-0B native-headroom decision to L3 evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 0x41524D415F424F4F
BOOTSTRAP_RESAMPLES = 10_000
REQUIRED_WINS = 6
WIN_THRESHOLD = 1.5
FLOOR = 0.80


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if raw["status"] != "pass" or raw["device"] != "cuda":
        raise SystemExit("CUDA L3 campaign is incomplete or failed")
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    cells = {}
    wins = 0
    ratios = []
    floor_pass = True
    for cell_id, cell in raw["cells"].items():
        existing = np.asarray(
            [row["seconds"] for row in cell["observations"]["t3-cuda"]],
            dtype=np.float64,
        )
        native = np.asarray(
            [row["seconds"] for row in cell["observations"]["ng-auto"]],
            dtype=np.float64,
        )
        if existing.size != 30 or native.size != 30:
            raise SystemExit(f"incomplete observations: {cell_id}")
        ratio = statistics.median(existing) / statistics.median(native)
        indices = rng.integers(0, existing.size, size=(BOOTSTRAP_RESAMPLES, existing.size))
        samples = np.median(existing[indices], axis=1) / np.median(native[indices], axis=1)
        low, high = np.quantile(samples, (0.025, 0.975))
        qualifies = bool(ratio >= WIN_THRESHOLD and low > 1.0)
        wins += int(qualifies)
        floor_pass = bool(floor_pass and ratio >= FLOOR)
        ratios.append(ratio)
        cells[cell_id] = {
            "shape": cell["shape"],
            "existing_median_seconds": statistics.median(existing),
            "native_median_seconds": statistics.median(native),
            "existing_over_native": ratio,
            "paired_bootstrap_95": [float(low), float(high)],
            "qualifying_1_5x_win": qualifies,
            "floor_pass": bool(ratio >= FLOOR),
            "existing_p05_p95": [float(value) for value in np.quantile(existing, (0.05, 0.95))],
            "native_p05_p95": [float(value) for value in np.quantile(native, (0.05, 0.95))],
        }
    case_2 = bool(wins >= REQUIRED_WINS and floor_pass)
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.l3-analysis.v1",
        "status": "complete",
        "decision": "CASE 2" if case_2 else "CASE 1 / NO-GO",
        "case_2": case_2,
        "case_1": not case_2,
        "decision_rule": {
            "ratio": "best ordinary existing median / native median",
            "win_threshold": WIN_THRESHOLD,
            "required_wins": REQUIRED_WINS,
            "floor": FLOOR,
            "wins": wins,
            "floor_pass": floor_pass,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "generator": "numpy.random.PCG64DXSM",
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "geometric_mean_existing_over_native": math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        ),
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "wins": wins,
                "floor_pass": floor_pass,
                "geometric_mean": result["geometric_mean_existing_over_native"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
