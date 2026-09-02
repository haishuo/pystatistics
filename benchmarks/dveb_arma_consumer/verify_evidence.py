#!/usr/bin/env python3
"""Offline verifier for the Phase-I DVEB exact-ARMA consumer evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/evidence/dveb_arma_consumer"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads((EVIDENCE / "qualification.json").read_text())
    regression = json.loads((EVIDENCE / "regression.json").read_text())
    manifest = json.loads((EVIDENCE / "handoff-manifest.json").read_text())
    assert result["status"] == "pass"
    assert len(result["cells"]) == 16
    assert len(result["edge_cases"]) == 3
    assert len(result["proposal_traces"]) == 8
    assert all(cell["pass"] and cell["inputs_unchanged"]
               for cell in result["cells"].values())
    assert all(len(cell["schedules"]) == 8
               and all(row["pass"] for row in cell["schedules"].values())
               for cell in result["cells"].values())
    assert all(edge["pass"] and edge["status_contract"]
               for edge in result["edge_cases"].values())
    for trace in result["proposal_traces"].values():
        assert trace["pass"] and trace["z_unchanged"]
        assert len(trace["schedules"]) == 8
        assert all(row["proposals"] == 100 and row["failures"] == 0
                   and row["status_mismatches"] == 0
                   for row in trace["schedules"].values())
    assert result["artifacts"]["combined_library_sha256"] == \
        manifest["files"]["exact_arma_abi_v1.so"]["sha256"]
    assert result["artifacts"]["cpu_library_sha256"] == \
        manifest["files"]["exact_arma_cpu_abi_v1.so"]["sha256"]
    assert sha256(EVIDENCE / "handoff-manifest.json") == \
        "dcff39c98c6d3442ff3a255e55ba2fa90a705dbfc028e2739abcbe53d18e07bb"
    assert regression["status"] == "pass"
    assert regression["timeseries_suite"] == "PASS"
    assert regression["phase0b_archived_evidence"] == "PASS"
    assert regression["consumer_evidence"] == "PASS"
    assert regression["complete_suite"]["no_new_failure"] is True
    assert set(regression["complete_suite"]["known_baseline_failures"]) == {
        "tests/descriptive/test_gpu.py::TestGPUvsCPU::test_describe_kurtosis",
        (
            "tests/multinomial/test_multinom.py::TestFailureCases::"
            "test_complete_separation_vcov_fails_loud"
        ),
    }
    assert regression["version"] == "6.2.0.dev0+dveb"
    assert regression["main_modified"] is False
    assert regression["public_arima_batch_modified"] is False
    assert (ROOT / "docs/research/DVEB_ARMA_CONSUMER_INTEGRATION_RESULTS.md").is_file()
    print("DVEB exact-ARMA consumer Phase-I evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
