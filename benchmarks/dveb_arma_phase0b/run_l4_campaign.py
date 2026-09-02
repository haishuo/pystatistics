#!/usr/bin/env python3
"""Run the frozen paired E07 fresh-process L4 campaign."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import generate_cell, sha256_array
from reference_impl import cython_batch
from run_l3_campaign import admitted, sha256_outputs

WARMUPS = 10
OBSERVATIONS = 30
ORDER_SEED = 0x41524D415F4C345F
IMPLEMENTATIONS = ("t3-cuda", "ng-auto")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def invoke(
    implementation: str, library: Path, block: int
) -> tuple[float, dict[str, object], str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT}:{HERE}"
    command = [
        sys.executable,
        str(HERE / "l4_worker.py"),
        "--implementation",
        implementation,
        "--cuda-library",
        str(library),
        "--block",
        str(block),
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=environment,
    )
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    return elapsed, json.loads(completed.stdout), completed.stderr


def arrays(record: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(record["nll"], dtype=np.float64),
        np.asarray(record["sigma2"], dtype=np.float64),
        np.asarray(record["status"], dtype=np.uint8).astype(np.bool_),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    floors = json.loads(args.tolerances.read_text())["derived_floor_relative"]
    block = calibration["block_mapping_by_r"]["3"]
    z, phi, loading = generate_cell("E07")
    expected = cython_batch(z, phi, loading)
    input_hashes = tuple(sha256_array(value) for value in (z, phi, loading))

    for implementation in IMPLEMENTATIONS:
        for _ in range(WARMUPS):
            _, record, _ = invoke(implementation, args.cuda_library, block)
            passed, _ = admitted(expected, arrays(record), z.shape[1], phi.shape[1], floors)
            if not passed or not record["input_immutable"]:
                raise RuntimeError(f"{implementation} L4 warmup correctness failure")

    rng = random.Random(ORDER_SEED)
    observations = {name: [] for name in IMPLEMENTATIONS}
    orders = []
    all_pass = True
    for observation in range(OBSERVATIONS):
        order = list(IMPLEMENTATIONS)
        rng.shuffle(order)
        orders.append(order)
        for implementation in order:
            elapsed, record, stderr = invoke(implementation, args.cuda_library, block)
            output = arrays(record)
            passed, comparison = admitted(expected, output, z.shape[1], phi.shape[1], floors)
            passed = bool(
                passed
                and record["input_immutable"]
                and tuple(record["input_sha256"]) == input_hashes
            )
            all_pass &= passed
            observations[implementation].append(
                {
                    "observation": observation,
                    "seconds": elapsed,
                    "peak_rss_kib": record["peak_rss_kib"],
                    "output_sha256": sha256_outputs(output),
                    "comparison": comparison,
                    "admitted": passed,
                    "stderr": stderr,
                }
            )

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.l4.v1",
        "status": "pass" if all_pass else "fail",
        "cell": "E07",
        "shape": {"k": z.shape[0], "n": z.shape[1], "r": phi.shape[1]},
        "block": block,
        "warmups": WARMUPS,
        "observations_count": OBSERVATIONS,
        "order_seed": ORDER_SEED,
        "orders": orders,
        "observations": observations,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
