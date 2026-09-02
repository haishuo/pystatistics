#!/usr/bin/env python3
"""Exercise persistent, diffuse, invalid, and boundary behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from reference_impl import cython_batch
from run_preflight import compare, to_numpy
from torch_impl import likelihood_while_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no fallback attempted")

    rng = np.random.Generator(np.random.PCG64DXSM(0x41524D415F454447))
    cases = {
        "persistent_stationary": 0.9999,
        "diffuse_nonstationary": 1.1,
        "invalid_nonfinite": float("nan"),
    }
    records = {}
    all_pass = True
    for case_id, coefficient in cases.items():
        z = np.ascontiguousarray(rng.standard_normal((8, 120)), dtype=np.float64)
        phi = np.full((8, 1), coefficient, dtype=np.float64)
        loading = np.ones((8, 1), dtype=np.float64)
        expected = cython_batch(z, phi, loading)
        tensors = tuple(torch.from_numpy(value).to(args.device) for value in (z, phi, loading))
        actual = to_numpy(likelihood_while_loop(*tensors))
        check = compare(expected, actual)
        expected_status = case_id != "invalid_nonfinite"
        status_contract = bool(np.all(actual[2] == expected_status))
        passed = check["preflight_pass"] and status_contract
        records[case_id] = {
            "coefficient": coefficient if np.isfinite(coefficient) else "nan",
            "comparison": check,
            "expected_status": expected_status,
            "status_contract": status_contract,
            "pass": passed,
        }
        all_pass &= passed

    # The public preflight boundary requires C-contiguous float64 arrays with
    # matching K/r shapes. These are checked before any timed campaign.
    z = np.ones((4, 120), dtype=np.float64)
    phi = np.ones((3, 1), dtype=np.float64)
    loading = np.ones((4, 1), dtype=np.float64)
    wrong_shape_refused = False
    try:
        likelihood_while_loop(
            torch.from_numpy(z).to(args.device),
            torch.from_numpy(phi).to(args.device),
            torch.from_numpy(loading).to(args.device),
        )
    except (RuntimeError, ValueError):
        wrong_shape_refused = True
    records["wrong_shape"] = {
        "refused": wrong_shape_refused,
        "pass": wrong_shape_refused,
    }
    all_pass &= wrong_shape_refused

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.edge-preflight.v1",
        "device": args.device,
        "torch": torch.__version__,
        "status": "pass" if all_pass else "fail",
        "cases": records,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
