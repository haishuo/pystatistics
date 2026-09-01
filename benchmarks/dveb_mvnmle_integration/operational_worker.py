#!/usr/bin/env python3
"""One operational endpoint worker for the frozen integration protocol."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import affinity, configure_threads, make_case  # noqa: E402


class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch blocked by DVEB operational qualification")
        return None


def _resident(data, case_id: str, threads: int) -> dict:  # noqa: ANN001
    import numpy as np

    configure_threads(threads)
    from pystatistics.mvnmle._dveb.objective import DVEBDenseObjective
    from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64

    torch_objective = GPUObjectiveFP64(data, device="cpu")
    dveb_objective = DVEBDenseObjective(data, threads=threads)
    theta = np.ascontiguousarray(torch_objective.get_initial_parameters())
    implementations = {
        "torch": torch_objective.compute_value_and_gradient,
        "dveb": dveb_objective.compute_value_and_gradient,
    }
    for _ in range(3):
        for function in implementations.values():
            function(theta)
    rng = random.Random(0xD0EB1008 + sum(ord(c) for c in case_id) + threads)
    observations = []
    for repetition in range(30):
        order = list(implementations)
        rng.shuffle(order)
        row = {"repetition": repetition, "order": order, "implementations": {}}
        for name in order:
            started = time.perf_counter_ns()
            value, gradient = implementations[name](theta)
            elapsed = time.perf_counter_ns() - started
            row["implementations"][name] = {
                "seconds": elapsed / 1.0e9,
                "value": float(value),
                "gradient_max_abs": float(np.max(np.abs(gradient))),
            }
        observations.append(row)
    dveb_objective.close()
    return {
        "mode": "resident",
        "case": case_id,
        "threads": threads,
        "observations": observations,
    }


def _dveb_load_first_answer(data, case_id: str, threads: int) -> dict:  # noqa: ANN001
    started = time.perf_counter_ns()
    from pystatistics.mvnmle._dveb.objective import DVEBDenseObjective

    objective = DVEBDenseObjective(data, threads=threads)
    theta = objective.get_initial_parameters()
    value, gradient = objective.compute_value_and_gradient(theta)
    elapsed = time.perf_counter_ns() - started
    objective.close()
    return {
        "mode": "dveb_load_first_answer",
        "case": case_id,
        "threads": threads,
        "seconds": elapsed / 1.0e9,
        "value": float(value),
        "gradient_length": len(gradient),
    }


def _torch_blocked_fit(data, case_id: str, threads: int) -> dict:  # noqa: ANN001
    sys.meta_path.insert(0, _BlockTorch())
    started = time.perf_counter_ns()
    from pystatistics.mvnmle import mlest

    result = mlest(data, method="direct", solver="dveb")
    elapsed = time.perf_counter_ns() - started
    return {
        "mode": "torch_blocked_load_to_fit",
        "case": case_id,
        "threads": threads,
        "seconds": elapsed / 1.0e9,
        "converged": bool(result.converged),
        "backend": result.backend_name,
        "loglik": float(result.loglik),
        "torch_imported": "torch" in sys.modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("resident", "dveb-load-first-answer", "torch-blocked-fit"),
        required=True,
    )
    parser.add_argument("--case", default="E1")
    parser.add_argument("--threads", type=int, choices=(1, 6, 12), required=True)
    args = parser.parse_args()
    cpus = affinity()
    if len(cpus) != args.threads:
        raise SystemExit(f"affinity mismatch: requested {args.threads} CPUs, process has {cpus}")
    data = make_case(args.case)
    if args.mode == "resident":
        payload = _resident(data, args.case, args.threads)
    elif args.mode == "dveb-load-first-answer":
        payload = _dveb_load_first_answer(data, args.case, args.threads)
    else:
        payload = _torch_blocked_fit(data, args.case, args.threads)
    payload.update(
        {
            "schema": "pystatistics.dveb-mvnmle.operational-worker.v1",
            "affinity": cpus,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
