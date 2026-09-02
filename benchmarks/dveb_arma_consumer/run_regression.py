#!/usr/bin/env python3
"""Run and record the Phase-I PyStatistics regression firewall."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/research/evidence/dveb_arma_consumer/regression.json"
PYTHON = "/home/haishuo/miniconda3/envs/gradflow/bin/python"
KNOWN_BASELINE_FAILURE = (
    "tests/multinomial/test_multinom.py::TestFailureCases::"
    "test_complete_separation_vcov_fails_loud"
)


def run(command, *, require_success=True):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.time()
    process = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True
    )
    record = {
        "command": command,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
        "stdout_tail": process.stdout.splitlines()[-30:],
        "stderr_tail": process.stderr.splitlines()[-30:],
    }
    if require_success and process.returncode:
        raise RuntimeError(json.dumps(record, indent=2))
    return record, process.stdout + "\n" + process.stderr


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    if subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).strip():
        raise SystemExit("refusing regression from a dirty research worktree")

    records = []
    for command in (
        [PYTHON, "-m", "pytest", "tests/timeseries", "-q"],
        [PYTHON, "benchmarks/dveb_arma_phase0b/verify_campaign.py"],
        [PYTHON, "benchmarks/dveb_arma_consumer/verify_evidence.py"],
    ):
        record, _ = run(command)
        records.append(record)

    full, output = run([PYTHON, "-m", "pytest", "tests", "-q"], require_success=False)
    failures = sorted(set(re.findall(r"^FAILED\s+([^\s]+)", output, flags=re.MULTILINE)))
    no_new_failure = full["returncode"] == 1 and failures == [KNOWN_BASELINE_FAILURE]
    if not no_new_failure:
        raise RuntimeError(json.dumps({"full_suite": full, "failures": failures}, indent=2))
    records.append(full)

    version = json.loads(subprocess.check_output(
        [PYTHON, "-c", "import json,pystatistics; print(json.dumps(pystatistics.__version__))"],
        cwd=ROOT, text=True,
    ))
    if version != "6.2.0.dev0+dveb":
        raise RuntimeError(f"unexpected research version {version!r}")

    result = {
        "schema": "pystatistics-dveb-arma-consumer-regression-v1",
        "status": "pass",
        "generated_wall_ns": time.time_ns(),
        "records": records,
        "timeseries_suite": "PASS",
        "phase0b_archived_evidence": "PASS",
        "consumer_evidence": "PASS",
        "complete_suite": {
            "no_new_failure": True,
            "known_baseline_failure": KNOWN_BASELINE_FAILURE,
            "baseline_reproduced_at": "a9c7e4d (existing committed evidence)",
        },
        "version": version,
        "main_modified": False,
        "public_arima_batch_modified": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("DVEB exact-ARMA consumer regression: PASS; no new full-suite failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
