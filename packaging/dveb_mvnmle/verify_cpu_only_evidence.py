#!/usr/bin/env python3
"""Mechanically verify committed DVEB CPU-only wheel evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/evidence/dveb_cpu_only_wheel"
OLD_ARTIFACT = (
    ROOT
    / "docs/research/evidence/dveb_mvnmle/baseline_artifacts"
    / "mvnmle_cpu_abi_v1_forge_native.elf"
)
CURRENT_ARTIFACT = (
    ROOT / "pystatistics/mvnmle/_dveb/artifacts/mvnmle_cpu_abi_v1.so"
)


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

    if not summary["correctness"]["torch_absent_before_and_after_fit"]:
        raise SystemExit("torch-free correctness claim is false")
    if summary["engineering_result"] != "PASS":
        raise SystemExit("engineering result is not PASS")
    if summary["formal_protocol_decision"] != "NO-GO":
        raise SystemExit("formal decision drift")
    suite = summary["full_branch_suite"]
    if suite["failed"] != 2 or not suite["pre_task_commit_reproduces_both_failures"]:
        raise SystemExit("full-suite limitation record drift")

    print(
        json.dumps(
            {
                "status": "pass",
                "runtime_environments": len(summary["runtime_environments"]),
                "wheels_recorded": 2,
                "engineering_result": summary["engineering_result"],
                "formal_protocol_decision": summary["formal_protocol_decision"],
                "baseline_failures": suite["failed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
