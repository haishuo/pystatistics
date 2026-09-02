#!/usr/bin/env python3
"""Summarize the descriptive E07 fresh-process L4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    seconds = np.asarray([row["seconds"] for row in rows], dtype=np.float64)
    rss = np.asarray([row["peak_rss_kib"] for row in rows], dtype=np.float64)
    median = float(np.median(seconds))
    return {
        "median_seconds": median,
        "p05_p95_seconds": [float(value) for value in np.quantile(seconds, (0.05, 0.95))],
        "mad_seconds": float(np.median(np.abs(seconds - median))),
        "median_peak_rss_kib": float(np.median(rss)),
        "p05_p95_peak_rss_kib": [float(value) for value in np.quantile(rss, (0.05, 0.95))],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    if raw["status"] != "pass":
        raise SystemExit("fresh-process campaign is incomplete or failed")
    torch_metrics = metrics(raw["observations"]["t3-cuda"])
    native_metrics = metrics(raw["observations"]["ng-auto"])
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.l4-analysis.v1",
        "status": "complete",
        "role": "descriptive only; does not revise the completed L3 Case-2 decision",
        "cell": raw["cell"],
        "shape": raw["shape"],
        "t3-cuda": torch_metrics,
        "ng-auto": native_metrics,
        "existing_over_native": (
            torch_metrics["median_seconds"] / native_metrics["median_seconds"]
        ),
        "existing_over_native_peak_rss": (
            torch_metrics["median_peak_rss_kib"] / native_metrics["median_peak_rss_kib"]
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
