#!/usr/bin/env python3
"""Correctness admission for every native CPU/CUDA forced schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common import CELLS, generate_cell, sha256_array
from derive_native_tolerances import CPU_THREADS, CUDA_BLOCKS
from native_runtime import NativeCPU, NativeCUDA
from reference_impl import cython_batch
from run_preflight import compare

EDGE_SEED = 0x41524D415F454447


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tolerance(n: int, r: int, floor: float) -> float:
    return max(32.0 * np.finfo(np.float64).eps * n * max(1, r), floor)


def admitted(check: dict[str, object], n: int, r: int, floors: dict[str, float]) -> bool:
    return bool(
        check["finite"]
        and check["status_equal"]
        and check["nll_rel"] <= tolerance(n, r, floors["nll_rel"])
        and check["sigma2_rel"] <= tolerance(n, r, floors["sigma2_rel"])
    )


def evaluate_candidates(
    cpu: NativeCPU,
    cuda_library: Path,
    z: np.ndarray,
    phi: np.ndarray,
    loading: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for threads in CPU_THREADS:
        result[f"nc-t{threads}"] = cpu.evaluate(z, phi, loading, threads=threads)
    with NativeCUDA(cuda_library, z, phi, loading) as cuda:
        for block in CUDA_BLOCKS:
            result[f"ng-b{block}"] = cuda.evaluate(block_size=block)
    return result


def edge_inputs() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]]:
    rng = np.random.Generator(np.random.PCG64DXSM(EDGE_SEED))
    result = {}
    for case_id, coefficient, expected_status in (
        ("persistent_stationary", 0.9999, True),
        ("diffuse_nonstationary", 1.1, True),
        ("invalid_nonfinite", float("nan"), False),
    ):
        result[case_id] = (
            np.ascontiguousarray(rng.standard_normal((8, 120)), dtype=np.float64),
            np.full((8, 1), coefficient, dtype=np.float64),
            np.ones((8, 1), dtype=np.float64),
            expected_status,
        )
    return result


def refusal_checks(cpu_library: Path, cuda_library: Path) -> dict[str, bool]:
    z = np.ones((4, 120), dtype=np.float64)
    phi = np.ones((4, 1), dtype=np.float64)
    loading = np.ones((4, 1), dtype=np.float64)
    checks = {}
    with NativeCPU(cpu_library) as cpu:
        for name, candidate in (
            ("cpu_zero_length", (np.ones((4, 0)), phi, loading)),
            ("cpu_wrong_shape", (z, np.ones((3, 1)), loading)),
            ("cpu_noncontiguous", (z[:, ::2], phi, loading)),
            ("cpu_local_capacity", (z, np.ones((4, 26)), np.ones((4, 26)))),
        ):
            try:
                cpu.evaluate(*candidate, threads=1)
            except (RuntimeError, ValueError):
                checks[name] = True
            else:
                checks[name] = False
    for name, candidate in (
        ("cuda_zero_length", (np.ones((4, 0)), phi, loading)),
        ("cuda_wrong_shape", (z, np.ones((3, 1)), loading)),
        ("cuda_noncontiguous", (z[:, ::2], phi, loading)),
        ("cuda_local_capacity", (z, np.ones((4, 26)), np.ones((4, 26)))),
    ):
        try:
            with NativeCUDA(cuda_library, *candidate):
                pass
        except (RuntimeError, ValueError):
            checks[name] = True
        else:
            checks[name] = False
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-library", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derivation = json.loads(args.tolerances.read_text())
    if derivation["status"] != "pass":
        raise SystemExit("tolerance derivation did not pass")
    floors = derivation["derived_floor_relative"]
    all_pass = True
    cell_records = {}
    with NativeCPU(args.cpu_library) as cpu:
        for cell_id in CELLS:
            z, phi, loading = generate_cell(cell_id)
            before = [sha256_array(item) for item in (z, phi, loading)]
            expected = cython_batch(z, phi, loading)
            candidates = evaluate_candidates(cpu, args.cuda_library, z, phi, loading)
            candidate_records = {}
            for name, actual in candidates.items():
                check = compare(expected, actual)
                check["nll_tolerance"] = tolerance(z.shape[1], phi.shape[1], floors["nll_rel"])
                check["sigma2_tolerance"] = tolerance(
                    z.shape[1], phi.shape[1], floors["sigma2_rel"]
                )
                check["admitted"] = admitted(check, z.shape[1], phi.shape[1], floors)
                candidate_records[name] = check
                all_pass &= check["admitted"]
            unchanged = before == [sha256_array(item) for item in (z, phi, loading)]
            all_pass &= unchanged
            cell_records[cell_id] = {
                "shape": {"k": z.shape[0], "n": z.shape[1], "r": phi.shape[1]},
                "input_sha256": {"z": before[0], "phi": before[1], "loading": before[2]},
                "input_unchanged": unchanged,
                "candidates": candidate_records,
            }
            print(
                cell_id,
                "PASS" if all(item["admitted"] for item in candidate_records.values()) else "FAIL",
            )

        edge_records = {}
        for case_id, (z, phi, loading, expected_status) in edge_inputs().items():
            expected = cython_batch(z, phi, loading)
            candidates = evaluate_candidates(cpu, args.cuda_library, z, phi, loading)
            rows = {}
            for name, actual in candidates.items():
                check = compare(expected, actual)
                status_contract = bool(np.all(actual[2] == expected_status))
                check["status_contract"] = status_contract
                check["admitted"] = (
                    admitted(check, z.shape[1], phi.shape[1], floors) and status_contract
                )
                rows[name] = check
                all_pass &= check["admitted"]
            edge_records[case_id] = rows

    refusals = refusal_checks(args.cpu_library, args.cuda_library)
    all_pass &= all(refusals.values())
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.native-qualification.v1",
        "status": "pass" if all_pass else "fail",
        "libraries": {"cpu": sha256(args.cpu_library), "cuda": sha256(args.cuda_library)},
        "tolerance_derivation_sha256": sha256(args.tolerances),
        "tolerance_formula": "max(32*eps*n*max(1,r), derived_floor) relative",
        "derived_floor_relative": floors,
        "cells": cell_records,
        "edge_cases": edge_records,
        "refusals": refusals,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "cells": len(cell_records)}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
