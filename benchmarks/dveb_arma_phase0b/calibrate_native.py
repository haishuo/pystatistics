#!/usr/bin/env python3
"""Select NG block sizes using calibration cells only."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from common import CELLS, generate_cell
from derive_native_tolerances import CUDA_BLOCKS
from native_runtime import NativeCUDA

CALIBRATION_SEED = 0x41524D415F43414C
WARMUPS = 10
OBSERVATIONS = 10
TARGET_SECONDS = 0.1
MAX_REPEATS = 10_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(CALIBRATION_SEED)
    records = {}
    mapping = {}

    for cell_id, cell in CELLS.items():
        if cell.phase != "calibration":
            continue
        z, phi, loading = generate_cell(cell_id)
        with NativeCUDA(args.cuda_library, z, phi, loading) as cuda:
            for block in CUDA_BLOCKS:
                for _ in range(WARMUPS):
                    cuda.launch(block_size=block)
                cuda.synchronize()
            pilots = {}
            for block in CUDA_BLOCKS:
                started = time.perf_counter_ns()
                cuda.launch(block_size=block)
                cuda.synchronize()
                pilots[str(block)] = (time.perf_counter_ns() - started) * 1.0e-9
            repeats = min(MAX_REPEATS, math.ceil(TARGET_SECONDS / min(pilots.values())))
            observations = {str(block): [] for block in CUDA_BLOCKS}
            orders = []
            for _ in range(OBSERVATIONS):
                order = list(CUDA_BLOCKS)
                rng.shuffle(order)
                orders.append(order)
                for block in order:
                    started = time.perf_counter_ns()
                    for _ in range(repeats):
                        cuda.launch(block_size=block)
                    cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
                    observations[str(block)].append(elapsed / repeats)
        medians = {block: statistics.median(values) for block, values in observations.items()}
        best = min(medians.values())
        selected = min(int(block) for block, value in medians.items() if value <= 1.01 * best)
        r = phi.shape[1]
        mapping[str(r)] = selected
        records[cell_id] = {
            "shape": {"k": z.shape[0], "n": z.shape[1], "r": r},
            "pilots_seconds_per_call": pilots,
            "repeats": repeats,
            "orders": orders,
            "seconds_per_call": observations,
            "medians_seconds_per_call": medians,
            "selected_block": selected,
        }
        print(cell_id, "r", r, "selected", selected, "medians", medians)

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.native-calibration.v1",
        "status": "complete",
        "decision_role": "calibration only; evaluation cells unseen",
        "seed": CALIBRATION_SEED,
        "warmups": WARMUPS,
        "observations": OBSERVATIONS,
        "target_seconds": TARGET_SECONDS,
        "maximum_repeats": MAX_REPEATS,
        "selection_rule": "median; within 1% choose smaller block",
        "block_mapping_by_r": mapping,
        "cells": records,
        "decision_observations": 0,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "mapping": mapping}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
