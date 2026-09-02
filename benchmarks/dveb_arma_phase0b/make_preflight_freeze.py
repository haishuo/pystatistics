#!/usr/bin/env python3
"""Create the source freeze consumed by formal preflight evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "preflight-freeze.json"
SOURCES = (
    "common.py",
    "reference_impl.py",
    "torch_impl.py",
    "run_preflight.py",
    "run_edge_cases.py",
    "inspect_compile.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.preflight-freeze.v1",
        "status": "frozen-before-formal-preflight",
        "source_commit": commit,
        "protocol_commit": "bb8b84cc0b79dc339190779e33044d82a60429cd",
        "input_freeze_sha256": sha256(HERE / "input-freeze.json"),
        "sources": {name: sha256(HERE / name) for name in SOURCES},
        "authorized_scope": "NumPy/Cython/PyTorch workload/null preflight only",
        "native_diagnostic_authorized": False,
        "dveb_compiler_work_authorized": False,
        "decision_observations": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "source_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
