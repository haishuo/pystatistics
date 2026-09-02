#!/usr/bin/env python3
"""Mechanically verify committed DVEB CPU-only wheel v2 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/evidence/dveb_cpu_only_wheel_v2"
V1 = ROOT / "docs/research/evidence/dveb_cpu_only_wheel/qualification.json"
CURRENT_ARTIFACT = ROOT / "pystatistics/mvnmle/_dveb/artifacts/mvnmle_cpu_abi_v1.so"
OLD_ARTIFACT = (
    ROOT
    / "docs/research/evidence/dveb_mvnmle/baseline_artifacts"
    / "mvnmle_cpu_abi_v1_forge_native.elf"
)
ALLOWLIST = [
    "tests/descriptive/test_gpu.py::TestGPUvsCPU::test_describe_kurtosis",
    "tests/multinomial/test_multinom.py::TestFailureCases::test_complete_separation_vcov_fails_loud",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    summary = json.loads((EVIDENCE / "qualification.json").read_text())
    for relative, wanted in summary["evidence_files"].items():
        actual = sha256(EVIDENCE / relative)
        if actual != wanted:
            raise SystemExit(f"evidence drift: {relative}: {actual} != {wanted}")

    source = summary["source"]
    if sha256(CURRENT_ARTIFACT) != source["manylinux_builder_artifact_sha256"]:
        raise SystemExit("source-tree manylinux artifact drift")
    if sha256(OLD_ARTIFACT) != source["frozen_forge_native_artifact_sha256"]:
        raise SystemExit("archived Forge-native performance artifact drift")

    expected_runtime_artifact = source["auditwheel_final_artifact_sha256"]
    for runtime in summary["runtime_environments"].values():
        report = json.loads((EVIDENCE / runtime["evidence"]).read_text())
        if report["status"] != "pass" or report["torch_present"]:
            raise SystemExit(f"runtime qualification failed: {runtime['name']}")
        if report["artifact_sha256"] != expected_runtime_artifact:
            raise SystemExit(f"runtime artifact drift: {runtime['name']}")
        if report["abi_version"] != 1 or report["x86_64_v2_missing"]:
            raise SystemExit(f"runtime ABI/ISA failure: {runtime['name']}")
        if not report["apple"]["converged"] or not report["apple"]["input_unchanged"]:
            raise SystemExit(f"runtime Apple fit failure: {runtime['name']}")
        dependencies = "\n".join(report["ldd"]).lower()
        if "not found" in dependencies:
            raise SystemExit(f"unresolved dependency: {runtime['name']}")
        if any(word in dependencies for word in ("torch", "cuda", "nvidia")):
            raise SystemExit(f"forbidden native dependency: {runtime['name']}")

    suite = summary["full_branch_suite"]
    if suite["failures"] != ALLOWLIST or suite["unexpected_failures"]:
        raise SystemExit("baseline-relative failure allowlist drift")
    if (suite["passed"], suite["failed"], suite["skipped"], suite["deselected"]) != (
        4478,
        2,
        94,
        27,
    ):
        raise SystemExit("full-suite count drift")
    if not suite["qualitative_signatures_match_v1"] or not suite["collection_completed"]:
        raise SystemExit("full-suite baseline comparison failed")
    if summary["correctness"]["focused_dveb_integration"] != {"passed": 18, "failed": 0}:
        raise SystemExit("focused DVEB integration drift")

    full_log = (EVIDENCE / "full-suite.txt").read_text()
    for node in ALLOWLIST:
        if f"FAILED {node}" not in full_log:
            raise SystemExit(f"full-suite log missing allowlisted failure: {node}")
    if "2 failed, 4478 passed, 94 skipped, 27 deselected" not in full_log:
        raise SystemExit("full-suite summary missing")

    v1 = json.loads(V1.read_text())
    if v1["formal_protocol_decision"] != "NO-GO":
        raise SystemExit("v1 permanent decision drift")
    if summary["formal_protocol_decision"] != "GO":
        raise SystemExit("v2 formal decision drift")
    if summary["regression_rule"] != "no new failures relative to da75578":
        raise SystemExit("v2 regression rule drift")

    print(
        json.dumps(
            {
                "status": "pass",
                "runtime_environments": len(summary["runtime_environments"]),
                "focused_tests": 18,
                "baseline_failures": suite["failed"],
                "unexpected_failures": len(suite["unexpected_failures"]),
                "v1_decision_preserved": v1["formal_protocol_decision"],
                "v2_decision": summary["formal_protocol_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
