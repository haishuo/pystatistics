#!/usr/bin/env python3
"""Run one frozen full-fit lane in a fresh process."""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    REPETITIONS,
    SEED,
    WARMUPS,
    affinity,
    array_sha256,
    compare_solutions,
    configure_threads,
    make_case,
    solution_record,
)


def _fit(design, implementation: str):
    from pystatistics.mvnmle import mlest

    if implementation == "torch":
        return mlest(design, method="direct", backend="cpu")
    if implementation == "dveb":
        return mlest(design, method="direct", solver="dveb")
    raise ValueError(f"unknown implementation: {implementation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(f"E{i}" for i in range(1, 7)))
    parser.add_argument("--threads", required=True, type=int, choices=(1, 6, 12))
    args = parser.parse_args()

    cpus = affinity()
    if len(cpus) != args.threads:
        raise SystemExit(f"affinity mismatch: requested {args.threads} CPUs, process has {cpus}")
    # Generate the frozen input before importing/configuring PyTorch. On this
    # environment, torch.set_num_threads(12) can change six E2 samples by one
    # ULP through NumPy's multivariate-normal linear-algebra path. The campaign
    # contract requires identical input bytes across thread-count lanes.
    data = make_case(args.case)
    input_sha256 = array_sha256(data)
    configure_threads(args.threads)

    import numpy as np
    import scipy
    import torch

    from pystatistics import __version__ as pystatistics_version
    from pystatistics.mvnmle import MVNDesign

    design = MVNDesign.from_array(data)

    for _ in range(WARMUPS):
        for implementation in ("torch", "dveb"):
            warmup = _fit(design, implementation)
            if not warmup.converged:
                raise SystemExit(f"{implementation} did not converge during an untimed warmup")

    lane_seed = SEED + sum(ord(char) for char in args.case) * 100 + args.threads
    rng = random.Random(lane_seed)
    observations = []
    for repetition in range(REPETITIONS):
        order = ["torch", "dveb"]
        rng.shuffle(order)
        pair = {
            "repetition": repetition,
            "order": order,
            "started_wall_ns": time.time_ns(),
            "implementations": {},
        }
        for implementation in order:
            started_wall_ns = time.time_ns()
            started = time.perf_counter_ns()
            solution = _fit(design, implementation)
            elapsed_ns = time.perf_counter_ns() - started
            record = solution_record(solution)
            record.update(
                {
                    "seconds": elapsed_ns / 1.0e9,
                    "started_wall_ns": started_wall_ns,
                    "finished_wall_ns": time.time_ns(),
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            pair["implementations"][implementation] = record
        pair["comparison"] = compare_solutions(
            pair["implementations"]["torch"],
            pair["implementations"]["dveb"],
        )
        pair["finished_wall_ns"] = time.time_ns()
        observations.append(pair)

    input_sha256_after = array_sha256(data)
    input_unchanged = input_sha256_after == input_sha256
    payload = {
        "schema": "pystatistics.dveb-mvnmle.lane.v1",
        "case": args.case,
        "threads": args.threads,
        "affinity": cpus,
        "lane_seed": lane_seed,
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "data_sha256": input_sha256,
        "data_sha256_after": input_sha256_after,
        "input_unchanged": input_unchanged,
        "versions": {
            "pystatistics": pystatistics_version,
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "observations": observations,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pass": input_unchanged and all(pair["comparison"]["pass"] for pair in observations),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
