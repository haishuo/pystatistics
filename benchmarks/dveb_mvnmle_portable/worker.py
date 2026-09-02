#!/usr/bin/env python3
"""Run one portable-wheel full-fit timing lane in a fresh process."""

from __future__ import annotations

import argparse
import json
import random
import resource
import time

from portable_common import (
    REPETITIONS,
    SEED,
    WARMUPS,
    affinity,
    array_sha256,
    compare_solutions,
    configure_threads,
    make_case,
    solution_record,
    verify_installed_identity,
)


def _fit(design, implementation: str):
    from pystatistics.mvnmle import mlest

    if implementation == "torch":
        return mlest(design, method="direct", backend="cpu")
    if implementation == "dveb":
        return mlest(design, method="direct", solver="dveb")
    raise ValueError(implementation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(f"E{i}" for i in range(1, 7)))
    parser.add_argument("--threads", required=True, type=int, choices=(1, 6, 12))
    args = parser.parse_args()
    cpus = affinity()
    if len(cpus) != args.threads:
        raise SystemExit(f"affinity mismatch: requested {args.threads}, got {cpus}")
    data = make_case(args.case)
    input_sha256 = array_sha256(data)
    configure_threads(args.threads)
    identity = verify_installed_identity()

    import numpy as np
    import scipy
    import torch
    from pystatistics import __version__ as pystatistics_version
    from pystatistics.mvnmle import MVNDesign

    design = MVNDesign.from_array(data)
    for _ in range(WARMUPS):
        for implementation in ("torch", "dveb"):
            if not _fit(design, implementation).converged:
                raise SystemExit(f"{implementation} failed an untimed warmup")

    lane_seed = SEED + sum(ord(char) for char in args.case) * 100 + args.threads
    rng = random.Random(lane_seed)
    observations = []
    for repetition in range(REPETITIONS):
        order = ["torch", "dveb"]
        rng.shuffle(order)
        pair = {"repetition": repetition, "order": order, "implementations": {}}
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
            pair["implementations"]["torch"], pair["implementations"]["dveb"]
        )
        observations.append(pair)

    unchanged = array_sha256(data) == input_sha256
    payload = {
        "schema": "pystatistics.dveb-portable.lane.v1",
        "case": args.case,
        "threads": args.threads,
        "affinity": cpus,
        "lane_seed": lane_seed,
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "data_sha256": input_sha256,
        "input_unchanged": unchanged,
        "installed_identity": identity,
        "versions": {
            "pystatistics": pystatistics_version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "observations": observations,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pass": unchanged and all(row["comparison"]["pass"] for row in observations),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
