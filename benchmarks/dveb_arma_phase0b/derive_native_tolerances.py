#!/usr/bin/env python3
"""Derive independent-compiler bounds on correctness-only inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from common import FAMILIES, sha256_array
from native_runtime import NativeCPU, NativeCUDA
from reference_impl import cython_batch
from run_preflight import compare

DERIVATION_SEED = 0x41524D415F544F4C
CASES = {
    "D01": (7, 257, "F1"),
    "D02": (19, 733, "F3"),
    "D03": (11, 333, "F13"),
    "D04": (5, 777, "F25"),
    "D05": (13, 411, "PERSISTENT"),
}
CPU_THREADS = (1, 6, 12)
CUDA_BLOCKS = (32, 64, 128, 256)


def case_seed(case_id: str) -> int:
    digest = hashlib.sha256(f"{DERIVATION_SEED}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def generate(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k, n, family_id = CASES[case_id]
    if family_id == "PERSISTENT":
        phi_one = np.asarray([0.9999], dtype=np.float64)
        loading_one = np.asarray([1.0], dtype=np.float64)
    else:
        family = FAMILIES[family_id]
        phi_one = np.asarray(family.phi, dtype=np.float64)
        loading_one = np.asarray(family.noise_loading, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(case_seed(case_id)))
    z = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float64)
    phi = np.array(np.broadcast_to(phi_one, (k, phi_one.size)), copy=True, order="C")
    loading = np.array(np.broadcast_to(loading_one, (k, loading_one.size)), copy=True, order="C")
    return z, phi, loading


def upward_power_of_ten(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return 10.0 ** math.ceil(math.log10(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-library", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    maximum = {"nll_rel": 0.0, "sigma2_rel": 0.0}
    all_pass = True
    with NativeCPU(args.cpu_library) as cpu:
        for case_id, (k, n, family_id) in CASES.items():
            z, phi, loading = generate(case_id)
            expected = cython_batch(z, phi, loading)
            candidates = {}
            for threads in CPU_THREADS:
                result = cpu.evaluate(z, phi, loading, threads=threads)
                check = compare(expected, result)
                candidates[f"nc-t{threads}"] = check
            with NativeCUDA(args.cuda_library, z, phi, loading) as cuda:
                for block in CUDA_BLOCKS:
                    result = cuda.evaluate(block_size=block)
                    check = compare(expected, result)
                    candidates[f"ng-b{block}"] = check
            for check in candidates.values():
                maximum["nll_rel"] = max(maximum["nll_rel"], check["nll_rel"])
                maximum["sigma2_rel"] = max(maximum["sigma2_rel"], check["sigma2_rel"])
                all_pass &= check["status_equal"] and check["finite"]
            records[case_id] = {
                "shape": {"k": k, "n": n, "r": phi.shape[1]},
                "family": family_id,
                "seed": case_seed(case_id),
                "input_sha256": {
                    "z": sha256_array(z),
                    "phi": sha256_array(phi),
                    "loading": sha256_array(loading),
                },
                "candidates": candidates,
            }

    derived_floor = {
        "nll_rel": upward_power_of_ten(4.0 * maximum["nll_rel"]),
        "sigma2_rel": upward_power_of_ten(4.0 * maximum["sigma2_rel"]),
    }
    all_pass &= max(derived_floor.values()) <= 1.0e-9
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.native-tolerance-derivation.v1",
        "status": "pass" if all_pass else "fail",
        "decision_role": "correctness-only; disjoint from calibration/evaluation",
        "formula": "max(32*eps*n*max(1,r), derived_floor) relative",
        "maximum_observed_relative": maximum,
        "derived_floor_relative": derived_floor,
        "cases": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "derived_floor": derived_floor}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
