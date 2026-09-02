#!/usr/bin/env python3
"""Run CUDA L1 resident and L2 transfer-inclusive ancillary endpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from common import CELLS, generate_cell, sha256_array
from native_runtime import NativeCUDA
from reference_impl import cython_batch
from run_l3_campaign import admitted, sha256_outputs
from torch_impl import compile_while_loop

WARMUPS = 10
OBSERVATIONS = 30
ORDER_SEED = 0x41524D415F4C3132
TARGET_SECONDS = 0.1
EVALUATION = tuple(f"E{number:02d}" for number in range(1, 13))


class TorchSingle:
    def __init__(self, z: np.ndarray, phi: np.ndarray, loading: np.ndarray):
        self.host = (z, phi, loading)
        self.device = tuple(torch.from_numpy(item).to("cuda") for item in self.host)
        self.function = compile_while_loop()
        self.function(*self.device)
        torch.cuda.synchronize()

    def resident(self, repeats: int) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        output = None
        started = time.perf_counter_ns()
        for _ in range(repeats):
            output = self.function(*self.device)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        assert output is not None
        return elapsed, tuple(item.detach().cpu().numpy() for item in output)

    def transfer(self, repeats: int) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        output = None
        started = time.perf_counter_ns()
        for _ in range(repeats):
            device = tuple(torch.from_numpy(item).to("cuda") for item in self.host)
            output = tuple(item.detach().cpu() for item in self.function(*device))
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        assert output is not None
        return elapsed, tuple(item.numpy() for item in output)


class NativeSingle:
    def __init__(
        self,
        library: Path,
        z: np.ndarray,
        phi: np.ndarray,
        loading: np.ndarray,
        block: int,
    ):
        self.context = NativeCUDA(library, z, phi, loading)
        self.block = block

    def resident(self, repeats: int) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        started = time.perf_counter_ns()
        for _ in range(repeats):
            self.context.launch(block_size=self.block)
        self.context.synchronize()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        return elapsed, self.context.download()

    def transfer(self, repeats: int) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        output = None
        started = time.perf_counter_ns()
        for _ in range(repeats):
            self.context.upload_base()
            self.context.launch(block_size=self.block)
            self.context.synchronize()
            output = self.context.download()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        assert output is not None
        return elapsed, output

    def close(self) -> None:
        self.context.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    floors = json.loads(args.tolerances.read_text())["derived_floor_relative"]
    rng = random.Random(ORDER_SEED)
    calibration_by_r = {record["shape"]["r"]: record for record in calibration["cells"].values()}
    cells = {}
    all_pass = True

    for cell_id in EVALUATION:
        cell = CELLS[cell_id]
        z, phi, loading = generate_cell(cell_id)
        r = phi.shape[1]
        calibration_cell = calibration_by_r[r]
        block = calibration["block_mapping_by_r"][str(r)]
        base_time = calibration_cell["medians_seconds_per_call"][str(block)]
        base_work = calibration_cell["shape"]["k"] * calibration_cell["shape"]["n"]
        estimated_time = base_time * (cell.k * cell.n) / base_work
        repeats = max(1, math.ceil(TARGET_SECONDS / estimated_time))
        expected = cython_batch(z, phi, loading)
        torch.cuda.reset_peak_memory_stats()
        implementations = {
            "t3-cuda": TorchSingle(z, phi, loading),
            "ng-auto": NativeSingle(args.cuda_library, z, phi, loading, block),
        }
        try:
            endpoint_records = {}
            for endpoint, method in (("L1", "resident"), ("L2", "transfer")):
                for implementation in implementations.values():
                    for _ in range(WARMUPS):
                        elapsed, output = getattr(implementation, method)(repeats)
                        passed, _ = admitted(expected, output, cell.n, r, floors)
                        if not passed:
                            raise RuntimeError(f"{cell_id} {endpoint} warmup failure")
                observations = {name: [] for name in implementations}
                orders = []
                for observation in range(OBSERVATIONS):
                    order = list(implementations)
                    rng.shuffle(order)
                    orders.append(order)
                    for name in order:
                        elapsed, output = getattr(implementations[name], method)(repeats)
                        passed, check = admitted(expected, output, cell.n, r, floors)
                        all_pass &= passed
                        observations[name].append(
                            {
                                "observation": observation,
                                "seconds": elapsed,
                                "seconds_per_likelihood": elapsed / repeats,
                                "output_sha256": sha256_outputs(output),
                                "comparison": check,
                            }
                        )
                endpoint_records[endpoint] = {"orders": orders, "observations": observations}
        finally:
            implementations["ng-auto"].close()
        cells[cell_id] = {
            "shape": {"k": cell.k, "n": cell.n, "r": r},
            "block": block,
            "repeats": repeats,
            "estimated_native_seconds": estimated_time,
            "input_sha256": {
                "z": sha256_array(z),
                "phi": sha256_array(phi),
                "loading": sha256_array(loading),
            },
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "native_payload_bytes": int(
                (cell.k * cell.n + 2 * cell.k * r + 2 * cell.k) * 8 + cell.k
            ),
            "endpoints": endpoint_records,
        }
        print(cell_id, "complete", "repeats", repeats)

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.cuda-ancillary.v1",
        "status": "pass" if all_pass else "fail",
        "warmups": WARMUPS,
        "observations": OBSERVATIONS,
        "order_seed": ORDER_SEED,
        "cells": cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
