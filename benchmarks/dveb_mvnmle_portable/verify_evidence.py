#!/usr/bin/env python3
"""Mechanically verify the portable MVN-MLE performance evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/evidence/dveb_mvnmle_portable"
FREEZE = EVIDENCE / "campaign-freeze-v2.json"
EXPECTED_ARTIFACT = "65438a0d742257acc0dc7bd2ff13d2017669fd21dc07f64add41b18fb229fe38"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    freeze = json.loads(FREEZE.read_text())
    if freeze["status"] != "frozen-before-timing" or freeze["v1"]["timed_observations"] != 0:
        raise SystemExit("freeze or v1 invalidation drift")
    for relative, wanted in freeze["hashes"].items():
        if sha256(ROOT / relative) != wanted:
            raise SystemExit(f"frozen source drift: {relative}")
    for case_id, fixture in freeze["fixtures"].items():
        path = ROOT / fixture["path"]
        if sha256(path) != fixture["file_sha256"]:
            raise SystemExit(f"fixture-file drift: {case_id}")
        array = np.ascontiguousarray(np.load(path, allow_pickle=False))
        if hashlib.sha256(array.view(np.uint8)).hexdigest() != fixture["array_sha256"]:
            raise SystemExit(f"fixture-array drift: {case_id}")

    summary = json.loads((EVIDENCE / "qualification-summary-v2.json").read_text())
    for relative, wanted in summary["evidence_files"].items():
        if sha256(EVIDENCE / relative) != wanted:
            raise SystemExit(f"result evidence drift: {relative}")
    if sha256(EVIDENCE / "artifacts/mvnmle_cpu_abi_v1_portable.so") != EXPECTED_ARTIFACT:
        raise SystemExit("portable artifact drift")

    qualification = json.loads((EVIDENCE / "fit-qualification-v2.json").read_text())
    if not qualification["complete"] or not qualification["pass"] or len(qualification["lanes"]) != 18:
        raise SystemExit("qualification incomplete or failed")
    for lane_id, lane in qualification["lanes"].items():
        if lane["input_sha256"] != freeze["fixtures"][lane["case"]]["array_sha256"]:
            raise SystemExit(f"qualification input drift: {lane_id}")
        if not lane["pass"] or not lane["input_unchanged"]:
            raise SystemExit(f"qualification failure: {lane_id}")
        if lane["installed_identity"]["artifact_sha256"] != EXPECTED_ARTIFACT:
            raise SystemExit(f"qualification artifact drift: {lane_id}")

    raw = json.loads((EVIDENCE / "campaign-raw-v2.json").read_text())
    pairs = 0
    if not raw["complete"] or len(raw["lanes"]) != 18:
        raise SystemExit("timing campaign incomplete")
    for lane_id, lane in raw["lanes"].items():
        if len(lane["observations"]) != 30 or not lane["pass"] or not lane["input_unchanged"]:
            raise SystemExit(f"timed lane failure: {lane_id}")
        if lane["data_sha256"] != freeze["fixtures"][lane["case"]]["array_sha256"]:
            raise SystemExit(f"timed input drift: {lane_id}")
        if lane["installed_identity"]["artifact_sha256"] != EXPECTED_ARTIFACT:
            raise SystemExit(f"timed artifact drift: {lane_id}")
        for observation in lane["observations"]:
            pairs += 1
            if not observation["comparison"]["pass"]:
                raise SystemExit(f"timed correctness failure: {lane_id}")
    if pairs != 540:
        raise SystemExit(f"expected 540 pairs, got {pairs}")

    analysis = json.loads((EVIDENCE / "campaign-analysis-v2.json").read_text())
    rule = analysis["decision_rule"]
    if analysis["decision"] != "STRONG GO" or not analysis["go"] or not rule["strong_go"]:
        raise SystemExit("formal decision drift")
    if rule["portable_faster_lanes"] != 18 or rule["pytorch_faster_lanes"] != 0:
        raise SystemExit("lane decision drift")
    if rule["preferred_portable_wins_at_least_1_5"] != 17 or rule["catastrophic_lanes"] != 0:
        raise SystemExit("threshold count drift")

    operational = json.loads((EVIDENCE / "operational-v2.json").read_text())
    if not operational["complete"] or len(operational["observations"]) != 30:
        raise SystemExit("operational evidence incomplete")
    for row in operational["observations"]:
        if not row["implementations"]["torch"]["converged"]:
            raise SystemExit("operational PyTorch failure")
        dveb = row["implementations"]["dveb"]
        if not dveb["converged"] or dveb["torch_imported"]:
            raise SystemExit("operational torch-free DVEB failure")

    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_lanes": 18,
                "timed_pairs": pairs,
                "portable_wins": rule["portable_faster_lanes"],
                "geometric_pytorch_over_portable": rule["geometric_mean_pytorch_over_portable"],
                "decision": analysis["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
