#!/usr/bin/env python3
"""Offline standard-library verifier for the exact-ARMA Phase-0B campaign."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/research/evidence/dveb_arma_phase0b_campaign"
NATIVE = ROOT / "docs/research/evidence/dveb_arma_phase0b_native"

EXPECTED_HASHES = {
    "cuda-l1-l2-analysis.json": "73569e8c9b3fbd997033f34327075bb4b44d95b7c1e8ac642ebe8f902025e5a0",
    "cuda-l1-l2-raw.json": "462f0d23861cd0e881b60f93ea220802980329a89e6f01a4f8d4cb278c22ad54",
    "l3-cuda-analysis.json": "e7b37a17d913edafbd4473aa87f0c53837f5e5238d8a3342803f698ec0c7920a",
    "l3-cuda-raw.json": "c05cf329bbf84c651915ae989a10172e7137d4644aab0fcd570969e59c079c63",
    "l4-analysis.json": "37e4c21e60edf3fa332adf1df03f8a68c9f68baf16526837e480782c32fb8170",
    "l4-raw.json": "a77a85e07fa12a49dcaf945b0c2b580975581d8b8c158edb4a5d1b22aa60458e",
    "regression-result.json": "e3a0c9fa1e76a4a56d7d33d2e627f3cfb64296f5b5ffe30dd922dfb8ab32e0c7",
}
PRIMARY = {"E04", "E05", "E06", "E07", "E08", "E09", "E11", "E12"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def verify_row(row: dict[str, object], label: str) -> None:
    comparison = row["comparison"]
    require(bool(comparison["admitted"]), f"{label}: unadmitted observation")
    require(bool(comparison["finite"]), f"{label}: non-finite result")
    require(bool(comparison["status_equal"]), f"{label}: status mismatch")


def main() -> int:
    for name, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((CAMPAIGN / name).read_bytes()).hexdigest()
        require(actual == expected, f"evidence hash drift: {name}")

    qualification = load(NATIVE / "qualification.json")
    require(qualification["status"] == "pass", "native qualification did not pass")
    require(len(qualification["cells"]) == 16, "native qualification grid is incomplete")
    freeze = load(NATIVE / "native-freeze-v2.json")
    for kind in ("cpu", "cuda"):
        artifact = NATIVE / freeze["artifacts"][kind]["path"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        require(digest == freeze["artifacts"][kind]["sha256"], f"{kind} artifact drift")

    l3_raw = load(CAMPAIGN / "l3-cuda-raw.json")
    l3 = load(CAMPAIGN / "l3-cuda-analysis.json")
    require(l3_raw["status"] == "pass", "L3 raw campaign did not pass")
    require(set(l3_raw["cells"]) == PRIMARY, "L3 primary grid drift")
    require(l3["decision"] == "CASE 2" and l3["case_2"], "L3 decision drift")
    ratios = []
    wins = 0
    for cell_id, cell in l3_raw["cells"].items():
        for implementation, rows in cell["observations"].items():
            require(len(rows) == 30, f"{cell_id}/{implementation}: incomplete L3 block")
            for row in rows:
                verify_row(row, f"{cell_id}/{implementation}")
        existing = statistics.median(
            row["seconds"] for row in cell["observations"]["t3-cuda"]
        )
        native = statistics.median(
            row["seconds"] for row in cell["observations"]["ng-auto"]
        )
        ratio = existing / native
        recorded = l3["cells"][cell_id]
        require(
            math.isclose(ratio, recorded["existing_over_native"], rel_tol=1e-15),
            f"{cell_id}: L3 ratio drift",
        )
        require(recorded["paired_bootstrap_95"][0] > 1.0, f"{cell_id}: CI floor failed")
        wins += int(ratio >= 1.5 and recorded["paired_bootstrap_95"][0] > 1.0)
        ratios.append(ratio)
    require(wins == 8, "L3 qualifying-win count drift")
    require(min(ratios) >= 0.80, "L3 catastrophic floor failed")
    geometric = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    require(
        math.isclose(geometric, l3["geometric_mean_existing_over_native"], rel_tol=1e-15),
        "L3 geometric mean drift",
    )

    endpoints_raw = load(CAMPAIGN / "cuda-l1-l2-raw.json")
    endpoints = load(CAMPAIGN / "cuda-l1-l2-analysis.json")
    require(endpoints_raw["status"] == "pass", "L1/L2 campaign did not pass")
    require(len(endpoints_raw["cells"]) == 12, "L1/L2 grid is incomplete")
    require(endpoints_raw["repetition_policy"]["maximum_repeats"] == 8, "repeat cap drift")
    for cell_id, cell in endpoints_raw["cells"].items():
        for endpoint, record in cell["endpoints"].items():
            for implementation, rows in record["observations"].items():
                require(len(rows) == 30, f"{cell_id}/{endpoint}/{implementation}: incomplete")
                for row in rows:
                    verify_row(row, f"{cell_id}/{endpoint}/{implementation}")
            existing = statistics.median(
                row["seconds_per_likelihood"]
                for row in record["observations"]["t3-cuda"]
            )
            native = statistics.median(
                row["seconds_per_likelihood"]
                for row in record["observations"]["ng-auto"]
            )
            recorded = endpoints["cells"][cell_id]["endpoints"][endpoint]
            require(
                math.isclose(existing / native, recorded["existing_over_native"], rel_tol=1e-15),
                f"{cell_id}/{endpoint}: ratio drift",
            )

    l4_raw = load(CAMPAIGN / "l4-raw.json")
    l4 = load(CAMPAIGN / "l4-analysis.json")
    require(l4_raw["status"] == "pass", "L4 campaign did not pass")
    require(l4_raw["cell"] == "E07", "L4 cell drift")
    for implementation, rows in l4_raw["observations"].items():
        require(len(rows) == 30, f"L4/{implementation}: incomplete block")
        for row in rows:
            verify_row(row, f"L4/{implementation}")
            require(bool(row["admitted"]), f"L4/{implementation}: worker admission failed")
    existing = statistics.median(row["seconds"] for row in l4_raw["observations"]["t3-cuda"])
    native = statistics.median(row["seconds"] for row in l4_raw["observations"]["ng-auto"])
    require(
        math.isclose(existing / native, l4["existing_over_native"], rel_tol=1e-15),
        "L4 ratio drift",
    )

    regression = load(CAMPAIGN / "regression-result.json")
    require(regression["status"] == "pass", "campaign regression did not pass")
    require(regression["passed"] == 153, "campaign regression count drift")
    require(regression["failed"] == 0 and regression["errors"] == 0, "regression failure")

    print(
        "PASS: Phase-0B CASE 2; "
        f"L3 wins={wins}/8, geomean={geometric:.6f}x, "
        f"L4={existing / native:.6f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
