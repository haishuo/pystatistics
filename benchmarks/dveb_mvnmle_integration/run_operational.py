#!/usr/bin/env python3
"""Run the non-decision operational endpoints in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import CASE_IDS, THREADS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "pystatistics/mvnmle/_dveb/artifacts/mvnmle_cpu_abi_v1.so"


def _run(mode: str, case_id: str, threads: int, cpus: list[int]) -> dict:
    cpu_list = ",".join(str(cpu) for cpu in cpus[:threads])
    command = [
        "taskset",
        "-c",
        cpu_list,
        sys.executable,
        str(Path(__file__).with_name("operational_worker.py")),
        "--mode",
        mode,
        "--case",
        case_id,
        "--threads",
        str(threads),
    ]
    run = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(run.stdout)
    payload["command"] = command
    payload["stderr"] = run.stderr
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite operational evidence: {output}")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 12:
        raise SystemExit(f"need 12 logical CPUs, only {cpus} are available")
    report = {
        "schema": "pystatistics.dveb-mvnmle.operational.v1",
        "started_wall_ns": time.time_ns(),
        "resident": {},
        "dveb_load_first_answer": [],
        "torch_blocked_load_to_fit": [],
        "artifact": {
            "path": str(ARTIFACT.relative_to(ROOT)),
            "bytes": ARTIFACT.stat().st_size,
            "ldd": subprocess.run(
                ["ldd", str(ARTIFACT)], text=True, capture_output=True, check=True
            ).stdout,
            "dynamic_section": subprocess.run(
                ["readelf", "-d", str(ARTIFACT)],
                text=True,
                capture_output=True,
                check=True,
            ).stdout,
        },
        "complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS:
        for threads in THREADS:
            lane_id = f"{case_id}-T{threads}"
            report["resident"][lane_id] = _run("resident", case_id, threads, cpus)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"resident {lane_id}: complete", flush=True)
    for _ in range(30):
        report["dveb_load_first_answer"].append(_run("dveb-load-first-answer", "E1", 1, cpus))
        report["torch_blocked_load_to_fit"].append(_run("torch-blocked-fit", "E1", 1, cpus))
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["complete"] = True
    report["finished_wall_ns"] = time.time_ns()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
