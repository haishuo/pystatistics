#!/usr/bin/env python3
"""Run the load-bearing 100-proposal L3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from campaign_existing import CythonParallel
from common import CELLS, generate_cell, sha256_array
from native_runtime import NativeCPU, NativeCUDA
from proposal_trace import proposal_trace
from reference_impl import cython_batch
from run_preflight import compare
from torch_impl import compile_while_loop

PRIMARY = ("E04", "E05", "E06", "E07", "E08", "E09", "E11", "E12")
CPU_THREADS = (1, 6, 12)
WARMUPS = 10
OBSERVATIONS = 30
ORDER_SEED = 0x41524D415F4C335F


def sha256_outputs(result: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in result:
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def tolerance(n: int, r: int, floor: float) -> float:
    return max(32.0 * np.finfo(np.float64).eps * n * max(1, r), floor)


def admitted(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
    r: int,
    floors: dict[str, float],
) -> tuple[bool, dict[str, object]]:
    check = compare(expected, actual)
    check["nll_tolerance"] = tolerance(n, r, floors["nll_rel"])
    check["sigma2_tolerance"] = tolerance(n, r, floors["sigma2_rel"])
    passed = bool(
        check["finite"]
        and check["status_equal"]
        and check["nll_rel"] <= check["nll_tolerance"]
        and check["sigma2_rel"] <= check["sigma2_tolerance"]
    )
    check["admitted"] = passed
    return passed, check


def reference_trace(
    z: np.ndarray, phi_trace: np.ndarray, loading_trace: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = [
        cython_batch(z, phi, loading) for phi, loading in zip(phi_trace, loading_trace, strict=True)
    ]
    return tuple(np.stack([row[index] for row in outputs]) for index in range(3))


class TorchTrace:
    def __init__(
        self,
        z: np.ndarray,
        phi_trace: np.ndarray,
        loading_trace: np.ndarray,
        device: str,
    ):
        self.device = torch.device(device)
        self.z = torch.from_numpy(z).to(self.device)
        self.phi = torch.from_numpy(phi_trace).to(self.device)
        self.loading = torch.from_numpy(loading_trace).to(self.device)
        self.function = compile_while_loop()
        first = self.function(self.z, self.phi[0], self.loading[0])
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        del first

    def run(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        outputs = []
        started = time.perf_counter_ns()
        for proposal in range(self.phi.shape[0]):
            outputs.append(self.function(self.z, self.phi[proposal], self.loading[proposal]))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        result = tuple(
            torch.stack([output[index] for output in outputs]).detach().cpu().numpy()
            for index in range(3)
        )
        return elapsed, result

    def close(self) -> None:
        """Match the native/context adapter lifecycle; tensors need no explicit close."""


class NativeCUDATrace:
    def __init__(
        self,
        library: Path,
        z: np.ndarray,
        phi: np.ndarray,
        loading: np.ndarray,
        phi_trace: np.ndarray,
        loading_trace: np.ndarray,
        block: int,
    ):
        self.context = NativeCUDA(
            library,
            z,
            phi,
            loading,
            phi_trace=phi_trace,
            loading_trace=loading_trace,
        )
        self.block = block
        self.proposals = phi_trace.shape[0]

    def run(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        started = time.perf_counter_ns()
        for proposal in range(self.proposals):
            self.context.launch(proposal=proposal, block_size=self.block)
        self.context.synchronize()
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        rows = [self.context.download(proposal=proposal) for proposal in range(self.proposals)]
        result = tuple(np.stack([row[index] for row in rows]) for index in range(3))
        return elapsed, result

    def close(self) -> None:
        self.context.close()


class NativeCPUTrace:
    def __init__(
        self,
        library: Path,
        z: np.ndarray,
        phi_trace: np.ndarray,
        loading_trace: np.ndarray,
        threads: int,
    ):
        self.context = NativeCPU(library)
        self.z = z
        self.phi = phi_trace
        self.loading = loading_trace
        self.threads = threads

    def run(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rows = []
        started = time.perf_counter_ns()
        for phi, loading in zip(self.phi, self.loading, strict=True):
            rows.append(self.context.evaluate(self.z, phi, loading, threads=self.threads))
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        return elapsed, tuple(np.stack([row[index] for row in rows]) for index in range(3))

    def close(self) -> None:
        self.context.close()


class CythonTrace:
    def __init__(
        self, z: np.ndarray, phi_trace: np.ndarray, loading_trace: np.ndarray, threads: int
    ):
        self.context = CythonParallel(z)
        self.phi = phi_trace
        self.loading = loading_trace
        self.threads = threads

    def run(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rows = []
        started = time.perf_counter_ns()
        for phi, loading in zip(self.phi, self.loading, strict=True):
            rows.append(self.context.evaluate(phi, loading, threads=self.threads))
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        return elapsed, tuple(np.stack([row[index] for row in rows]) for index in range(3))

    def close(self) -> None:
        self.context.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cpu-library", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    floors = json.loads(args.tolerances.read_text())["derived_floor_relative"]
    rng = random.Random(ORDER_SEED + (0 if args.device == "cpu" else 1))
    result_cells = {}
    all_pass = True

    for cell_id in PRIMARY:
        cell = CELLS[cell_id]
        z, phi, loading = generate_cell(cell_id)
        phi_trace, loading_trace, _ = proposal_trace(cell_id)
        expected = reference_trace(z, phi_trace, loading_trace)
        if args.device == "cuda":
            block = calibration["block_mapping_by_r"][str(phi.shape[1])]
            implementations = {
                "t3-cuda": TorchTrace(z, phi_trace, loading_trace, "cuda"),
                "ng-auto": NativeCUDATrace(
                    args.cuda_library,
                    z,
                    phi,
                    loading,
                    phi_trace,
                    loading_trace,
                    block,
                ),
            }
        else:
            implementations = {
                **{
                    f"c2-t{threads}": CythonTrace(z, phi_trace, loading_trace, threads)
                    for threads in CPU_THREADS
                },
                **{
                    f"nc-t{threads}": NativeCPUTrace(
                        args.cpu_library, z, phi_trace, loading_trace, threads
                    )
                    for threads in CPU_THREADS
                },
                "t3-cpu": TorchTrace(z, phi_trace, loading_trace, "cpu"),
            }
        try:
            for implementation in implementations.values():
                for _ in range(WARMUPS):
                    _, output = implementation.run()
                    passed, _ = admitted(expected, output, cell.n, phi.shape[1], floors)
                    if not passed:
                        raise RuntimeError(f"warmup correctness failure: {cell_id}")
            observations = {name: [] for name in implementations}
            orders = []
            for observation in range(OBSERVATIONS):
                order = list(implementations)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    elapsed, output = implementations[name].run()
                    passed, check = admitted(expected, output, cell.n, phi.shape[1], floors)
                    all_pass &= passed
                    observations[name].append(
                        {
                            "observation": observation,
                            "seconds": elapsed,
                            "output_sha256": sha256_outputs(output),
                            "comparison": check,
                        }
                    )
        finally:
            for implementation in implementations.values():
                implementation.close()
        result_cells[cell_id] = {
            "shape": {"k": cell.k, "n": cell.n, "r": phi.shape[1]},
            "input_sha256": {
                "z": sha256_array(z),
                "phi_trace": sha256_array(phi_trace),
                "loading_trace": sha256_array(loading_trace),
            },
            "orders": orders,
            "observations": observations,
        }
        print(args.device, cell_id, "complete")

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.l3-campaign.v1",
        "status": "pass" if all_pass else "fail",
        "device": args.device,
        "endpoint": "L3 fixed 100-proposal trace",
        "warmups": WARMUPS,
        "observations": OBSERVATIONS,
        "order_seed": ORDER_SEED + (0 if args.device == "cpu" else 1),
        "cells": result_cells,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
