#!/usr/bin/env python3
"""Run 30 randomized fresh-process operational pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from portable_common import SEED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    cpu = min(os.sched_getaffinity(0))
    rng = random.Random(SEED + 77)
    report = {
        "schema": "pystatistics.dveb-portable.operational.v1",
        "started_wall_ns": time.time_ns(),
        "cpu": cpu,
        "observations": [],
        "complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(30):
        order = ["torch", "dveb"]
        rng.shuffle(order)
        row = {"repetition": repetition, "order": order, "implementations": {}}
        for implementation in order:
            command = [
                "taskset", "-c", str(cpu), sys.executable,
                str(Path(__file__).with_name("operational_worker.py")),
                "--implementation", implementation,
            ]
            environment = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_DYNAMIC": "FALSE",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            run = subprocess.run(command, cwd="/tmp", env=environment, text=True, capture_output=True)
            if run.returncode != 0:
                report["failure"] = {
                    "repetition": repetition, "implementation": implementation,
                    "stdout": run.stdout, "stderr": run.stderr,
                }
                output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                raise SystemExit("operational worker failed")
            row["implementations"][implementation] = json.loads(run.stdout)
        report["observations"].append(row)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"operational pair {repetition + 1}/30", flush=True)
    report["complete"] = True
    report["finished_wall_ns"] = time.time_ns()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
