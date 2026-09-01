#!/usr/bin/env python3
"""Mechanical analysis of the frozen PyStatistics/DVEB full-fit campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from common import SEED


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _summary(values: np.ndarray) -> dict[str, float]:
    median = float(np.median(values))
    return {
        "median_seconds": median,
        "p05_seconds": _percentile(values, 5),
        "p95_seconds": _percentile(values, 95),
        "mad_seconds": float(np.median(np.abs(values - median))),
    }


def _bootstrap_paired_speedup(
    torch_times: np.ndarray, dveb_times: np.ndarray, rng: np.random.Generator
) -> list[float]:
    samples = np.empty(10_000, dtype=np.float64)
    for index in range(samples.size):
        selected = rng.integers(0, len(torch_times), size=len(torch_times))
        samples[index] = np.median(torch_times[selected]) / np.median(dveb_times[selected])
    return [_percentile(samples, 2.5), _percentile(samples, 97.5)]


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
    catastrophic = 0
    preferred = 0
    correctness = True
    for lane_id, lane in sorted(raw["lanes"].items()):
        torch_times = np.asarray(
            [row["implementations"]["torch"]["seconds"] for row in lane["observations"]]
        )
        dveb_times = np.asarray(
            [row["implementations"]["dveb"]["seconds"] for row in lane["observations"]]
        )
        torch_median = float(np.median(torch_times))
        dveb_median = float(np.median(dveb_times))
        speedup = torch_median / dveb_median
        lane_correct = bool(
            lane.get("pass") and all(row["comparison"]["pass"] for row in lane["observations"])
        )
        correctness = correctness and lane_correct
        speedups.append(speedup)
        pytorch_faster += int(speedup < 1.0)
        catastrophic += int(speedup <= 0.10)
        preferred += int(speedup >= 1.50)
        first = lane["observations"][0]["implementations"]
        lanes[lane_id] = {
            "case": lane["case"],
            "threads": lane["threads"],
            "torch": _summary(torch_times),
            "dveb": _summary(dveb_times),
            "median_pytorch_over_dveb": speedup,
            "paired_bootstrap_95": _bootstrap_paired_speedup(torch_times, dveb_times, rng),
            "correctness_pass": lane_correct,
            "torch_n_iter": sorted(
                {row["implementations"]["torch"]["n_iter"] for row in lane["observations"]}
            ),
            "dveb_n_iter": sorted(
                {row["implementations"]["dveb"]["n_iter"] for row in lane["observations"]}
            ),
            "torch_n_function_evals": sorted(
                {
                    row["implementations"]["torch"]["info"]["n_function_evals"]
                    for row in lane["observations"]
                }
            ),
            "dveb_n_function_evals": sorted(
                {
                    row["implementations"]["dveb"]["info"]["n_function_evals"]
                    for row in lane["observations"]
                }
            ),
            "dveb_selected_schedule": first["dveb"]["info"].get("dveb_selected_schedule"),
            "dveb_scratch_bytes": first["dveb"]["info"].get("dveb_scratch_bytes"),
            "max_rss_kib": lane["max_rss_kib"],
        }

    geometric_dveb_over_pytorch = math.exp(
        sum(math.log(1.0 / value) for value in speedups) / len(speedups)
    )
    systematic = pytorch_faster >= 12 and geometric_dveb_over_pytorch >= 2.0
    dominated = catastrophic > 0 or systematic
    go = bool(correctness and not dominated)
    rule = {
        "lane_count": 18,
        "correctness_pass": correctness,
        "pytorch_faster_lanes": pytorch_faster,
        "preferred_dveb_wins_at_least_1_5": preferred,
        "catastrophic_lanes_pytorch_over_dveb_at_most_0_10": catastrophic,
        "geometric_mean_dveb_over_pytorch": geometric_dveb_over_pytorch,
        "systematic_domination": systematic,
        "dominated": dominated,
    }
    if not correctness:
        decision = "HALT / correctness failure"
    elif dominated:
        decision = "NO-GO / Case 1"
    else:
        decision = "GO"
    analysis = {
        "schema": "pystatistics.dveb-mvnmle.campaign-analysis.v1",
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
