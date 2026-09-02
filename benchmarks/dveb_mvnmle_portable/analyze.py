#!/usr/bin/env python3
"""Mechanical analysis for the frozen portable-wheel campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from portable_common import SEED


def _summary(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    return {
        "median_seconds": median,
        "p05_seconds": float(np.percentile(values, 5)),
        "p95_seconds": float(np.percentile(values, 95)),
        "mad_seconds": float(np.median(np.abs(values - median))),
    }


def _bootstrap(torch: np.ndarray, dveb: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.empty(10_000)
    for index in range(len(values)):
        selected = rng.integers(0, len(torch), size=len(torch))
        values[index] = np.median(torch[selected]) / np.median(dveb[selected])
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    if not raw.get("complete") or len(raw.get("lanes", {})) != 18:
        raise SystemExit("campaign is incomplete")
    rng = np.random.default_rng(SEED)
    lanes = {}
    speedups = []
    pytorch_faster = 0
    portable_faster = 0
    preferred = 0
    catastrophic = 0
    correctness = True
    for lane_id, lane in sorted(raw["lanes"].items()):
        torch = np.asarray([row["implementations"]["torch"]["seconds"] for row in lane["observations"]])
        dveb = np.asarray([row["implementations"]["dveb"]["seconds"] for row in lane["observations"]])
        speedup = float(np.median(torch) / np.median(dveb))
        correct = bool(lane.get("pass") and all(row["comparison"]["pass"] for row in lane["observations"]))
        correctness = correctness and correct
        speedups.append(speedup)
        pytorch_faster += int(speedup < 1.0)
        portable_faster += int(speedup > 1.0)
        preferred += int(speedup >= 1.5)
        catastrophic += int(speedup <= 0.10)
        first = lane["observations"][0]["implementations"]
        lanes[lane_id] = {
            "case": lane["case"],
            "threads": lane["threads"],
            "torch": _summary(torch),
            "portable_dveb": _summary(dveb),
            "median_pytorch_over_portable": speedup,
            "paired_bootstrap_95": _bootstrap(torch, dveb, rng),
            "correctness_pass": correct,
            "torch_n_iter": sorted({row["implementations"]["torch"]["n_iter"] for row in lane["observations"]}),
            "dveb_n_iter": sorted({row["implementations"]["dveb"]["n_iter"] for row in lane["observations"]}),
            "dveb_selected_schedule": first["dveb"]["info"].get("dveb_selected_schedule"),
            "dveb_artifact_sha256": lane["installed_identity"]["artifact_sha256"],
            "max_rss_kib_shared": lane["max_rss_kib"],
        }
    geometric_pytorch_over_portable = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    geometric_portable_over_pytorch = 1.0 / geometric_pytorch_over_portable
    systematic = pytorch_faster >= 12 and geometric_portable_over_pytorch >= 2.0
    dominated = catastrophic > 0 or systematic
    go = bool(correctness and not dominated)
    strong_go = bool(go and portable_faster >= 12 and geometric_pytorch_over_portable >= 1.25)
    if not correctness:
        decision = "HALT / correctness failure"
    elif dominated:
        decision = "NO-GO / dominated"
    elif strong_go:
        decision = "STRONG GO"
    else:
        decision = "GO / not dominated"
    rule = {
        "lane_count": 18,
        "correctness_pass": correctness,
        "pytorch_faster_lanes": pytorch_faster,
        "portable_faster_lanes": portable_faster,
        "preferred_portable_wins_at_least_1_5": preferred,
        "catastrophic_lanes": catastrophic,
        "geometric_mean_pytorch_over_portable": geometric_pytorch_over_portable,
        "geometric_mean_portable_over_pytorch": geometric_portable_over_pytorch,
        "systematic_domination": systematic,
        "dominated": dominated,
        "strong_go": strong_go,
    }
    analysis = {
        "schema": "pystatistics.dveb-portable.campaign-analysis.v1",
        "lanes": lanes,
        "decision_rule": rule,
        "decision": decision,
        "go": go,
    }
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "rule": rule}, indent=2))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
