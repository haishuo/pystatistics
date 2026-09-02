#!/usr/bin/env python3
"""Run all untimed portable-wheel qualification lanes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from portable_common import CASE_IDS, THREADS


def _cpu_list(count: int, available: list[int]) -> str:
    if len(available) < count:
        raise RuntimeError(f"need {count} logical CPUs, only {available} available")
    return ",".join(str(cpu) for cpu in available[:count])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    available = sorted(os.sched_getaffinity(0))
    report = {
        "schema": "pystatistics.dveb-portable.fit-qualification-suite.v1",
        "started_wall_ns": time.time_ns(),
        "orchestrator_affinity": available,
        "lanes": {},
        "complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS:
        for threads in THREADS:
            lane_id = f"{case_id}-T{threads}"
            command = [
                "taskset", "-c", _cpu_list(threads, available), sys.executable,
                str(Path(__file__).with_name("qualify_worker.py")),
                "--case", case_id, "--threads", str(threads),
            ]
            environment = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_DYNAMIC": "FALSE",
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
            }
            run = subprocess.run(command, cwd="/tmp", env=environment, text=True, capture_output=True)
            if run.returncode != 0:
                report["failure"] = {"lane": lane_id, "stdout": run.stdout, "stderr": run.stderr}
                output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                raise SystemExit(f"qualification failed at {lane_id}")
            report["lanes"][lane_id] = json.loads(run.stdout)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"{lane_id}: PASS", flush=True)
    report["complete"] = True
    report["pass"] = all(lane["pass"] for lane in report["lanes"].values())
    report["finished_wall_ns"] = time.time_ns()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
