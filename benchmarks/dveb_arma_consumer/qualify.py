#!/usr/bin/env python3
"""Exhaustive Phase-I qualification through the PyStatistics consumer adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE0 = ROOT / "benchmarks/dveb_arma_phase0b"
sys.path.insert(0, str(PHASE0))

from common import CELLS, generate_cell, sha256_array  # noqa: E402
from proposal_trace import proposal_trace  # noqa: E402
from reference_impl import cython_batch  # noqa: E402
from run_preflight import compare  # noqa: E402

from pystatistics.timeseries._dveb_arma import (  # noqa: E402
    DVEBCPUExactArma,
    DVEBCudaTransferExactArma,
)
from pystatistics.timeseries._dveb_arma.loader import (  # noqa: E402
    CPU_ITEM_PARALLEL,
    CPU_LIBRARY_PATH,
    CPU_LIBRARY_SHA256,
    CPU_SERIAL,
    MANIFEST_PATH,
)

OUTPUT = ROOT / "docs/research/evidence/dveb_arma_consumer/qualification.json"
COMBINED = Path("/mnt/artifacts/dveb/trunk009_exact_arma_20260902/exact_arma_abi_v1.so")
FLOORS_PATH = ROOT / "docs/research/evidence/dveb_arma_phase0b_native/tolerance-derivation.json"
CPU_POLICIES = (
    ("cpu-serial-1", CPU_SERIAL, 1),
    ("cpu-parallel-6", CPU_ITEM_PARALLEL, 6),
    ("cpu-parallel-12", CPU_ITEM_PARALLEL, 12),
)
CUDA_BLOCKS = (32, 64, 128, 256, 0)
EPS = np.finfo(np.float64).eps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def admitted(expected, actual, *, steps: int, state: int, floors: dict) -> dict:
    result = compare(expected, actual)
    result["nll_tolerance"] = max(32.0 * EPS * steps * max(1, state), floors["nll_rel"])
    result["sigma2_tolerance"] = max(
        32.0 * EPS * steps * max(1, state), floors["sigma2_rel"]
    )
    result["pass"] = bool(
        result["finite"]
        and result["status_equal"]
        and result["nll_rel"] <= result["nll_tolerance"]
        and result["sigma2_rel"] <= result["sigma2_tolerance"]
    )
    return result


def evaluators(z, phi):
    cpu = {
        name: DVEBCPUExactArma(max_threads=12, schedule=schedule)
        for name, schedule, _threads in CPU_POLICIES
    }
    cuda = DVEBCudaTransferExactArma(
        max_items=z.shape[0], max_steps=z.shape[1], max_state=phi.shape[1],
        library_path=COMBINED,
    )
    return cpu, cuda


def close_evaluators(cpu, cuda):
    for evaluator in cpu.values():
        evaluator.close()
    cuda.close()


def evaluate_all(cpu, cuda, z, phi, loading, expected, floors):
    schedules = {}
    cpu_reference = None
    for name, _schedule, threads in CPU_POLICIES:
        actual = cpu[name].evaluate(z, phi, loading, threads=threads)
        check = admitted(expected, actual, steps=z.shape[1], state=phi.shape[1], floors=floors)
        if cpu_reference is None:
            cpu_reference = actual
        bitwise = all(np.array_equal(a, b) for a, b in zip(cpu_reference, actual, strict=True))
        check.update({
            "selected_schedule": cpu[name].last_selected_schedule,
            "bitwise_with_cpu_serial": bitwise,
            "pass": bool(check["pass"] and bitwise),
        })
        schedules[name] = check
    for block in CUDA_BLOCKS:
        actual = cuda.evaluate(z, phi, loading, block=block)
        check = admitted(expected, actual, steps=z.shape[1], state=phi.shape[1], floors=floors)
        check["selected_block"] = cuda.last_selected_block
        schedules["cuda-auto" if block == 0 else f"cuda-{block}"] = check
    return schedules


def edge_inputs():
    rng = np.random.Generator(np.random.PCG64DXSM(0x41524D415F454447))
    for name, coefficient, expected_status in (
        ("persistent_stationary", 0.9999, True),
        ("diffuse_nonstationary", 1.1, True),
        ("invalid_nonfinite", float("nan"), False),
    ):
        yield name, (
            np.ascontiguousarray(rng.standard_normal((8, 120)), dtype=np.float64),
            np.full((8, 1), coefficient, dtype=np.float64),
            np.ones((8, 1), dtype=np.float64),
        ), expected_status


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    if not COMBINED.is_file():
        raise SystemExit(f"qualified external CUDA artifact missing at {COMBINED}")
    if git("status", "--short"):
        raise SystemExit("refusing qualification from a dirty research worktree")
    floors = json.loads(FLOORS_PATH.read_text())["derived_floor_relative"]
    all_pass = True
    cells = {}

    for cell_id in CELLS:
        z, phi, loading = generate_cell(cell_id)
        before = [sha256_array(value) for value in (z, phi, loading)]
        expected = cython_batch(z, phi, loading)
        cpu, cuda = evaluators(z, phi)
        try:
            schedules = evaluate_all(cpu, cuda, z, phi, loading, expected, floors)
        finally:
            close_evaluators(cpu, cuda)
        unchanged = before == [sha256_array(value) for value in (z, phi, loading)]
        passed = unchanged and all(row["pass"] for row in schedules.values())
        all_pass &= passed
        cells[cell_id] = {
            "shape": {"items": z.shape[0], "steps": z.shape[1], "state": phi.shape[1]},
            "inputs_unchanged": unchanged, "schedules": schedules, "pass": passed,
        }
        print(cell_id, "PASS" if passed else "FAIL", flush=True)

    edges = {}
    for name, values, expected_status in edge_inputs():
        z, phi, loading = values
        expected = cython_batch(z, phi, loading)
        cpu, cuda = evaluators(z, phi)
        try:
            schedules = evaluate_all(cpu, cuda, z, phi, loading, expected, floors)
        finally:
            close_evaluators(cpu, cuda)
        status_contract = all(
            row["status_equal"] for row in schedules.values()
        ) and all(
            np.all(evaluator_result == expected_status)
            for evaluator_result in [expected[2]]
        )
        passed = status_contract and all(row["pass"] for row in schedules.values())
        all_pass &= passed
        edges[name] = {
            "expected_status": expected_status, "status_contract": status_contract,
            "schedules": schedules, "pass": passed,
        }
        print(name, "PASS" if passed else "FAIL", flush=True)

    proposals = {}
    for cell_id in (name for name, cell in CELLS.items() if cell.primary):
        z, _phi, _loading = generate_cell(cell_id)
        phi_trace, loading_trace, minimum_root = proposal_trace(cell_id)
        cpu, cuda = evaluators(z, phi_trace[0])
        aggregate = {
            name: {"proposals": 0, "failures": 0, "max_nll_rel": 0.0,
                   "max_sigma2_rel": 0.0, "status_mismatches": 0}
            for name, *_ in CPU_POLICIES
        }
        aggregate.update({
            "cuda-auto" if block == 0 else f"cuda-{block}": {
                "proposals": 0, "failures": 0, "max_nll_rel": 0.0,
                "max_sigma2_rel": 0.0, "status_mismatches": 0,
            } for block in CUDA_BLOCKS
        })
        before = sha256_array(z)
        try:
            for phi, loading in zip(phi_trace, loading_trace, strict=True):
                expected = cython_batch(z, phi, loading)
                schedules = evaluate_all(cpu, cuda, z, phi, loading, expected, floors)
                for schedule, check in schedules.items():
                    row = aggregate[schedule]
                    row["proposals"] += 1
                    row["failures"] += 0 if check["pass"] else 1
                    row["max_nll_rel"] = max(row["max_nll_rel"], check["nll_rel"])
                    row["max_sigma2_rel"] = max(
                        row["max_sigma2_rel"], check["sigma2_rel"]
                    )
                    row["status_mismatches"] += 0 if check["status_equal"] else 1
        finally:
            close_evaluators(cpu, cuda)
        unchanged = before == sha256_array(z)
        passed = unchanged and all(row["failures"] == 0 for row in aggregate.values())
        all_pass &= passed
        proposals[cell_id] = {
            "proposal_count": phi_trace.shape[0], "minimum_ar_root_modulus": minimum_root,
            "z_unchanged": unchanged, "schedules": aggregate, "pass": passed,
        }
        print(cell_id, "proposal-trace", "PASS" if passed else "FAIL", flush=True)

    result = {
        "schema": "pystatistics-dveb-arma-consumer-qualification-v1",
        "status": "pass" if all_pass else "fail",
        "repository": {
            "commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "branch": git("branch", "--show-current"), "clean_before": True,
        },
        "frozen_authority_commit": "f156ee5a305238583a4b7b332462ab6740cadc23",
        "adapter_commit": git("rev-parse", "HEAD"),
        "artifacts": {
            "consumer_manifest_sha256": sha256(MANIFEST_PATH),
            "cpu_library_sha256": sha256(CPU_LIBRARY_PATH),
            "cpu_library_expected_sha256": CPU_LIBRARY_SHA256,
            "combined_library_sha256": sha256(COMBINED),
        },
        "tolerance": "max(32*eps*steps*max(1,state), frozen derived floor), scale-relative",
        "cells": cells, "edge_cases": edges, "proposal_traces": proposals,
    }
    OUTPUT.write_text(json.dumps(
        result, indent=2, sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n")
    print(json.dumps({"status": result["status"], "cells": len(cells),
                      "proposal_cells": len(proposals)}))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
