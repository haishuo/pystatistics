#!/usr/bin/env python3
"""Run and checkpoint the prospectively frozen 18-lane campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from common import CASE_IDS, SEED, THREADS

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "docs/research/evidence/dveb_mvnmle/campaign-freeze.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_list(count: int, available: list[int]) -> str:
    if len(available) < count:
        raise RuntimeError(f"need {count} logical CPUs, only {available} are available")
    return ",".join(str(cpu) for cpu in available[:count])


def _verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("status") != "frozen-before-timing":
        raise SystemExit("campaign freeze is not active")
    for relative, wanted in freeze["hashes"].items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != wanted:
            raise SystemExit(f"frozen input drift: {relative}: expected {wanted}, got {actual}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite campaign evidence: {output}")

    _verify_freeze()
    lanes = [(case_id, threads) for case_id in CASE_IDS for threads in THREADS]
    random.Random(SEED).shuffle(lanes)
    report = {
        "schema": "pystatistics.dveb-mvnmle.campaign-raw.v1",
        "freeze_sha256": _sha256(FREEZE),
        "randomization_seed": hex(SEED),
        "lane_order": [f"{case_id}-T{threads}" for case_id, threads in lanes],
        "started_wall_ns": time.time_ns(),
        "lanes": {},
        "complete": False,
    }
    available_cpus = sorted(os.sched_getaffinity(0))
    report["orchestrator_affinity"] = available_cpus
    output.parent.mkdir(parents=True, exist_ok=True)
    for case_id, threads in lanes:
        lane_id = f"{case_id}-T{threads}"
        command = [
            "taskset",
            "-c",
            _cpu_list(threads, available_cpus),
            sys.executable,
            str(Path(__file__).with_name("worker.py")),
            "--case",
            case_id,
            "--threads",
            str(threads),
        ]
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        }
        run = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode != 0:
            report["failure"] = {
                "lane": lane_id,
                "returncode": run.returncode,
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise SystemExit(f"lane {lane_id} failed; evidence checkpointed")
        lane = json.loads(run.stdout)
        lane["command"] = command
        lane["stderr"] = run.stderr
        report["lanes"][lane_id] = lane
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        medians = {
            implementation: statistics.median(
                observation["implementations"][implementation]["seconds"]
                for observation in lane["observations"]
            )
            for implementation in ("torch", "dveb")
        }
        print(
            f"{lane_id}: torch={medians['torch']:.6f}s "
            f"dveb={medians['dveb']:.6f}s "
            f"speedup={medians['torch'] / medians['dveb']:.3f}x",
            flush=True,
        )

    report["complete"] = True
    report["finished_wall_ns"] = time.time_ns()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
