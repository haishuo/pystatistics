#!/usr/bin/env python3
"""Correctness-first consumer/null preflight; intentionally no timing campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from common import CELLS, generate_cell, sha256_array
from reference_impl import cython_batch, numpy_sample
from torch_impl import (
    compile_fullgraph,
    compile_while_loop,
    likelihood_for_loop,
    likelihood_while_loop,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def compare(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    e_nll, e_sigma2, e_status = expected
    a_nll, a_sigma2, a_status = actual
    nll_abs = float(np.max(np.abs(e_nll - a_nll)))
    sigma_abs = float(np.max(np.abs(e_sigma2 - a_sigma2)))
    nll_rel = float(np.max(np.abs(e_nll - a_nll) / np.maximum(1.0, np.abs(e_nll))))
    sigma_rel = float(np.max(np.abs(e_sigma2 - a_sigma2) / np.maximum(1.0, np.abs(e_sigma2))))
    return {
        "nll_abs": nll_abs,
        "nll_rel": nll_rel,
        "sigma2_abs": sigma_abs,
        "sigma2_rel": sigma_rel,
        "status_equal": bool(np.array_equal(e_status, a_status)),
        "finite": bool(np.isfinite(a_nll).all() and np.isfinite(a_sigma2).all()),
        "preflight_pass": bool(
            nll_rel <= 1.0e-9 and sigma_rel <= 1.0e-9 and np.array_equal(e_status, a_status)
        ),
    }


def to_numpy(result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    return tuple(item.detach().cpu().numpy() for item in result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--variant",
        choices=("eager", "compile", "while", "compile-while"),
        default="eager",
    )
    parser.add_argument("--cells", nargs="+", default=["C01", "C02"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no fallback attempted")
    device = torch.device(args.device)
    implementations = {
        "eager": lambda: likelihood_for_loop,
        "compile": compile_fullgraph,
        "while": lambda: likelihood_while_loop,
        "compile-while": compile_while_loop,
    }
    implementation = implementations[args.variant]()
    records = {}
    all_pass = True

    for cell_id in args.cells:
        if cell_id not in CELLS:
            raise SystemExit(f"unknown frozen cell: {cell_id}")
        z, phi, loading = generate_cell(cell_id)
        before = tuple(sha256_array(value) for value in (z, phi, loading))
        cython = cython_batch(z, phi, loading)
        numpy = numpy_sample(z, phi, loading)
        cython_sample = tuple(value[: numpy[0].shape[0]] for value in cython)
        numpy_check = compare(numpy, cython_sample)
        tensors = tuple(torch.from_numpy(value).to(device) for value in (z, phi, loading))
        actual = to_numpy(implementation(*tensors))
        if device.type == "cuda":
            torch.cuda.synchronize()
        torch_check = compare(cython, actual)
        after = tuple(sha256_array(value) for value in (z, phi, loading))
        unchanged = before == after
        passed = numpy_check["preflight_pass"] and torch_check["preflight_pass"] and unchanged
        records[cell_id] = {
            "shape": {"k": z.shape[0], "n": z.shape[1], "r": phi.shape[1]},
            "input_sha256": {"z": before[0], "phi": before[1], "noise_loading": before[2]},
            "numpy_vs_cython_sample": numpy_check,
            "torch_vs_cython": torch_check,
            "input_unchanged": unchanged,
            "pass": passed,
        }
        all_pass &= passed
        print(cell_id, "PASS" if passed else "FAIL", torch_check)

    result = {
        "schema": "pystatistics.dveb-arma-phase0b.preflight.v1",
        "status": "pass" if all_pass else "fail",
        "decision_role": "correctness/expressibility preflight; no performance timing",
        "source": {
            "commit": git_value("rev-parse", "HEAD"),
            "tree": git_value("rev-parse", "HEAD^{tree}"),
            "input_freeze_sha256": sha256(HERE / "input-freeze.json"),
            "common_sha256": sha256(HERE / "common.py"),
            "reference_impl_sha256": sha256(HERE / "reference_impl.py"),
            "torch_impl_sha256": sha256(HERE / "torch_impl.py"),
            "runner_sha256": sha256(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "variant": args.variant,
        },
        "cells": records,
    }
    output = args.output or HERE / f"preflight-{args.device}-{args.variant}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
