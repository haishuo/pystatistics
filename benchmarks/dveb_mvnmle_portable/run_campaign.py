#!/usr/bin/env python3
"""Run and checkpoint the frozen portable-wheel campaign."""

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

from portable_common import CASE_IDS, SEED, THREADS

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "docs/research/evidence/dveb_mvnmle_portable/campaign-freeze-v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_list(count: int, available: list[int]) -> str:
    if len(available) < count:
        raise RuntimeError(f"need {count} logical CPUs, only {available} available")
    return ",".join(str(cpu) for cpu in available[:count])


def _verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text())
    if freeze.get("status") != "frozen-before-timing":
        raise SystemExit("campaign freeze is not active")
    for relative, wanted in freeze["hashes"].items():
        actual = _sha256(ROOT / relative)
        if actual != wanted:
            raise SystemExit(f"frozen input drift: {relative}: {actual} != {wanted}")
    for case_id, fixture in freeze["fixtures"].items():
        actual = _sha256(ROOT / fixture["path"])
        if actual != fixture["file_sha256"]:
            raise SystemExit(f"frozen fixture-file drift: {case_id}: {actual}")
    wheel = Path(freeze["wheel"]["path"])
    if _sha256(wheel) != freeze["wheel"]["sha256"]:
        raise SystemExit("frozen wheel drift")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    _verify_freeze()
    lanes = [(case_id, threads) for case_id in CASE_IDS for threads in THREADS]
    random.Random(SEED).shuffle(lanes)
    available = sorted(os.sched_getaffinity(0))
    report = {
        "schema": "pystatistics.dveb-portable.campaign-raw.v1",
        "freeze_sha256": _sha256(FREEZE),
        "randomization_seed": hex(SEED),
        "lane_order": [f"{case}-T{threads}" for case, threads in lanes],
        "orchestrator_affinity": available,
        "started_wall_ns": time.time_ns(),
        "lanes": {},
        "complete": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for case_id, threads in lanes:
        lane_id = f"{case_id}-T{threads}"
        command = [
            "taskset", "-c", _cpu_list(threads, available), sys.executable,
            str(Path(__file__).with_name("worker.py")),
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
            report["failure"] = {
                "lane": lane_id, "returncode": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr,
            }
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            raise SystemExit(f"lane {lane_id} failed; evidence checkpointed")
        lane = json.loads(run.stdout)
        lane["command"] = command
        lane["stderr"] = run.stderr
        report["lanes"][lane_id] = lane
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        medians = {
            name: statistics.median(row["implementations"][name]["seconds"] for row in lane["observations"])
            for name in ("torch", "dveb")
        }
        print(
            f"{lane_id}: torch={medians['torch']:.6f}s portable={medians['dveb']:.6f}s "
            f"speedup={medians['torch'] / medians['dveb']:.3f}x",
            flush=True,
        )
    report["complete"] = True
    report["finished_wall_ns"] = time.time_ns()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
