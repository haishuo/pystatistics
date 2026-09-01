#!/usr/bin/env python3
"""Verify the committed DVEB MVN-MLE consumer evidence mechanically."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/evidence/dveb_mvnmle"
EXPECTED_LANES = {f"E{case}-T{threads}" for case in range(1, 7) for threads in (1, 6, 12)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    freeze = json.loads((EVIDENCE / "campaign-freeze.json").read_text())
    for relative, wanted in freeze["hashes"].items():
        actual = _sha256(ROOT / relative)
        if actual != wanted:
            raise SystemExit(f"frozen input drift: {relative}: {actual} != {wanted}")

    raw = json.loads((EVIDENCE / "campaign-raw.json").read_text())
    analysis = json.loads((EVIDENCE / "campaign-analysis.json").read_text())
    operational = json.loads((EVIDENCE / "operational.json").read_text())
    qualification = json.loads((EVIDENCE / "fit-qualification.json").read_text())
    if not qualification.get("complete") or not qualification.get("pass"):
        raise SystemExit("fit qualification is not a complete pass")
    if set(qualification["lanes"]) != EXPECTED_LANES:
        raise SystemExit("fit qualification lane set drifted")
    if not raw.get("complete") or set(raw.get("lanes", {})) != EXPECTED_LANES:
        raise SystemExit("raw campaign is incomplete or has the wrong lane set")
    if raw["freeze_sha256"] != _sha256(EVIDENCE / "campaign-freeze.json"):
        raise SystemExit("raw campaign does not identify the committed freeze")

    speedups = []
    pytorch_faster = 0
    catastrophic = 0
    preferred = 0
    for lane_id, lane in raw["lanes"].items():
        if lane["repetitions"] != 30 or lane["warmups"] != 3:
            raise SystemExit(f"repeat/warmup drift in {lane_id}")
        if len(lane["affinity"]) != lane["threads"]:
            raise SystemExit(f"affinity drift in {lane_id}")
        if lane["data_sha256"] != freeze["input_sha256"][lane["case"]]:
            raise SystemExit(f"input drift in {lane_id}")
        if not lane["input_unchanged"] or lane["data_sha256_after"] != lane["data_sha256"]:
            raise SystemExit(f"caller-owned input changed during {lane_id}")
        if len(lane["observations"]) != 30:
            raise SystemExit(f"observation-count drift in {lane_id}")
        if not all(row["comparison"]["pass"] for row in lane["observations"]):
            raise SystemExit(f"correctness failure in {lane_id}")
        if not all(
            row["implementations"][name]["converged"]
            for row in lane["observations"]
            for name in ("torch", "dveb")
        ):
            raise SystemExit(f"non-converged fit in {lane_id}")
        torch_times = np.asarray(
            [row["implementations"]["torch"]["seconds"] for row in lane["observations"]]
        )
        dveb_times = np.asarray(
            [row["implementations"]["dveb"]["seconds"] for row in lane["observations"]]
        )
        speedup = float(np.median(torch_times) / np.median(dveb_times))
        recorded = analysis["lanes"][lane_id]["median_pytorch_over_dveb"]
        if not math.isclose(speedup, recorded, rel_tol=0.0, abs_tol=1.0e-15):
            raise SystemExit(f"analysis mismatch in {lane_id}")
        speedups.append(speedup)
        pytorch_faster += int(speedup < 1.0)
        catastrophic += int(speedup <= 0.10)
        preferred += int(speedup >= 1.50)

    geometric_dveb_over_pytorch = math.exp(
        sum(math.log(1.0 / speedup) for speedup in speedups) / len(speedups)
    )
    systematic = pytorch_faster >= 12 and geometric_dveb_over_pytorch >= 2.0
    rule = analysis["decision_rule"]
    checks = {
        "pytorch_faster_lanes": pytorch_faster,
        "preferred_dveb_wins_at_least_1_5": preferred,
        "catastrophic_lanes_pytorch_over_dveb_at_most_0_10": catastrophic,
    }
    for key, value in checks.items():
        if rule[key] != value:
            raise SystemExit(f"decision count mismatch for {key}")
    if not math.isclose(
        rule["geometric_mean_dveb_over_pytorch"],
        geometric_dveb_over_pytorch,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise SystemExit("geometric-mean decision value mismatch")
    if rule["systematic_domination"] != systematic:
        raise SystemExit("systematic-domination decision mismatch")
    if analysis["decision"] != "GO" or not analysis["go"]:
        raise SystemExit("campaign decision is not GO")

    if not operational.get("complete") or set(operational["resident"]) != EXPECTED_LANES:
        raise SystemExit("operational evidence is incomplete")
    if len(operational["dveb_load_first_answer"]) != 30:
        raise SystemExit("load-to-first-answer repetition drift")
    blocked = operational["torch_blocked_load_to_fit"]
    if len(blocked) != 30 or not all(
        row["converged"] and not row["torch_imported"] for row in blocked
    ):
        raise SystemExit("torch-blocked fit evidence failed")
    dependencies = operational["artifact"]["ldd"].lower()
    if any(name in dependencies for name in ("torch", "cuda", "nvidia")):
        raise SystemExit("forbidden artifact dependency detected")

    print(
        json.dumps(
            {
                "status": "pass",
                "qualification_lanes": 18,
                "campaign_pairs": 540,
                "operational_resident_lanes": 18,
                "torch_blocked_fits": 30,
                "decision": analysis["decision"],
                "geometric_mean_pytorch_over_dveb": 1.0 / geometric_dveb_over_pytorch,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
