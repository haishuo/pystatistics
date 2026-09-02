#!/usr/bin/env python3
"""Summarize descriptive CUDA L1/L2 ancillary evidence mechanically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "median_seconds_per_likelihood": median,
        "p05_p95": [float(value) for value in np.quantile(array, (0.05, 0.95))],
        "mad": float(np.median(np.abs(array - median))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if raw["status"] != "pass" or len(raw["cells"]) != 12:
        raise SystemExit("ancillary CUDA campaign is incomplete or failed")

    cells = {}
    for cell_id, cell in raw["cells"].items():
        endpoints = {}
        for endpoint, record in cell["endpoints"].items():
            torch_values = [
                row["seconds_per_likelihood"] for row in record["observations"]["t3-cuda"]
            ]
            native_values = [
                row["seconds_per_likelihood"] for row in record["observations"]["ng-auto"]
            ]
            torch_summary = summary(torch_values)
            native_summary = summary(native_values)
            endpoints[endpoint] = {
                "t3-cuda": torch_summary,
                "ng-auto": native_summary,
                "existing_over_native": (
                    torch_summary["median_seconds_per_likelihood"]
                    / native_summary["median_seconds_per_likelihood"]
                ),
            }
        cells[cell_id] = {
            "shape": cell["shape"],
            "block": cell["block"],
            "repeats": cell["repeats"],
            "torch_peak_allocated_bytes": cell["torch_peak_allocated_bytes"],
            "native_payload_bytes": cell["native_payload_bytes"],
            "endpoints": endpoints,
        }

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.cuda-ancillary-analysis.v1",
        "status": "complete",
        "role": "descriptive only; does not revise the completed L3 Case-2 decision",
        "repetition_policy": raw["repetition_policy"],
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
