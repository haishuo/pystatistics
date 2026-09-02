#!/usr/bin/env python3
"""Freeze Phase-II inputs, starts, environment, and implementation identities."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE.parent)]

from benchmarks.dveb_arma_fit_phase2.common import (  # noqa: E402
    CALIBRATION_CELLS,
    EVALUATION_CELLS,
    OPTIMIZER_WORKERS,
    MAX_ACTIVE_OPTIMIZERS,
    THREAD_STACK_BYTES,
    input_start_record,
    public_airpassengers,
    sha256_array,
    sha256_file,
)

INPUT_OUTPUT = HERE / "input-start-freeze.json"
ENV_OUTPUT = HERE / "environment-freeze.json"
PROTOCOL = ROOT / "docs/research/DVEB_ARMA_FIT_PHASE2_PROTOCOL.md"
DVEB_ENV_LOCK = Path("/mnt/projects/dveb/comparator/environment.lock")
SCIPY_WHEEL = (
    "scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64."
    "manylinux_2_28_x86_64.whl"
)
SCIPY_WHEEL_SHA256 = "f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def refuse_outputs() -> None:
    existing = [str(path) for path in (INPUT_OUTPUT, ENV_OUTPUT) if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite frozen outputs: {existing}")


def main() -> int:
    refuse_outputs()
    expected_head = "46e321c"
    if not git("rev-parse", "HEAD").startswith(expected_head):
        raise SystemExit("input freeze must be generated directly after protocol commit 46e321c")
    public = public_airpassengers()
    input_result = {
        "schema": "pystatistics.dveb-arma-fit-phase2.input-start-freeze.v1",
        "status": "frozen-before-fit-implementation",
        "source": {
            "branch": git("branch", "--show-current"),
            "generation_parent_commit": git("rev-parse", "HEAD"),
            "protocol_sha256": sha256_file(PROTOCOL),
            "common_sha256": sha256_file(HERE / "common.py"),
            "coordinator_model_sha256": sha256_file(HERE / "coordinator_model.py"),
            "freeze_script_sha256": sha256_file(Path(__file__)),
        },
        "coordinator": {
            "optimizer_workers": OPTIMIZER_WORKERS,
            "max_active_optimizers": MAX_ACTIVE_OPTIMIZERS,
            "thread_stack_bytes": THREAD_STACK_BYTES,
            "ordering": "consecutive-input-order chunks; ascending row within barrier",
            "coalescing": "all-live-worker barrier only; no timeout or time window",
        },
        "implementation_identities": {
            "S0": "Cython exact likelihood; independent SciPy L-BFGS-B; forward FD",
            "TC1": "public-op eager PyTorch CPU exact recurrence; autograd",
            "TC2": "fullgraph structured-while PyTorch CPU exact recurrence; autograd",
            "TG1": "public-op eager PyTorch CUDA exact recurrence; autograd",
            "TG2": "fullgraph structured-while PyTorch CUDA exact recurrence; autograd",
            "D1": "frozen Phase-I DVEB CPU item-parallel likelihood; selected batched FD",
            "D2": "frozen Phase-I DVEB cuda-transfer likelihood; selected batched FD",
        },
        "cells": {
            cell_id: input_start_record(cell_id)
            for cell_id in CALIBRATION_CELLS + EVALUATION_CELLS
        },
        "public_case": {
            "source": "tests/fixtures/arima_kalman_r_reference.json:series.airpassengers",
            "transformation": "diff(log(x)); subtract sample mean",
            "shape": list(public.shape),
            "sha256": sha256_array(public),
            "batch_sizes": [1, 32],
            "timed": False,
        },
        "phase2_results_exist": False,
    }
    INPUT_OUTPUT.write_text(json.dumps(input_result, indent=2, sort_keys=True) + "\n")

    gpu = {
        "available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    env_result = {
        "schema": "pystatistics.dveb-arma-fit-phase2.environment-freeze.v1",
        "status": "frozen-before-fit-implementation",
        "executable": sys.executable,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": command(sys.executable, "-c", "import triton; print(triton.__version__)"),
        "gpu": gpu,
        "pip_freeze_all": command(sys.executable, "-m", "pip", "freeze", "--all").splitlines(),
        "wheel_hash_authorities": {
            "dveb_environment_lock": str(DVEB_ENV_LOCK),
            "dveb_environment_lock_sha256": sha256_file(DVEB_ENV_LOCK),
            "scipy_wheel": SCIPY_WHEEL,
            "scipy_wheel_sha256": SCIPY_WHEEL_SHA256,
        },
        "phase2_results_exist": False,
    }
    ENV_OUTPUT.write_text(json.dumps(env_result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"inputs": str(INPUT_OUTPUT), "environment": str(ENV_OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
