#!/usr/bin/env python3
"""Mechanically verify the exact-ARMA Phase-0B preflight evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/research/evidence/dveb_arma_phase0b_preflight"
FREEZE = HERE / "preflight-freeze.json"
SUMMARY = EVIDENCE / "preflight-summary.json"

ALL_CELLS = {f"C{number:02d}" for number in range(1, 5)} | {
    f"E{number:02d}" for number in range(1, 13)
}
EAGER_CELLS = {"C01", "C02"}
EDGE_CASES = {
    "persistent_stationary",
    "diffuse_nonstationary",
    "invalid_nonfinite",
    "wrong_shape",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_source_freeze() -> dict[str, object]:
    freeze = load(FREEZE)
    require(freeze["status"] == "frozen-before-formal-preflight", "freeze status drift")
    require(freeze["decision_observations"] == 0, "pre-freeze decision observation drift")
    require(not freeze["native_diagnostic_authorized"], "native authorization drift")
    require(not freeze["dveb_compiler_work_authorized"], "DVEB authorization drift")
    require(
        sha256(HERE / "input-freeze.json") == freeze["input_freeze_sha256"],
        "input freeze drift",
    )
    for relative, wanted in freeze["sources"].items():
        require(sha256(HERE / relative) == wanted, f"frozen source drift: {relative}")
    return freeze


def verify_grid(path: Path, expected_cells: set[str]) -> tuple[float, float, float, float]:
    result = load(path)
    require(result["status"] == "pass", f"failed result: {path.name}")
    require(set(result["cells"]) == expected_cells, f"cell-set drift: {path.name}")
    maxima = [0.0, 0.0, 0.0, 0.0]
    metrics = ("nll_abs", "nll_rel", "sigma2_abs", "sigma2_rel")
    for cell_id, cell in result["cells"].items():
        require(cell["pass"], f"cell failed: {path.name}:{cell_id}")
        require(cell["input_unchanged"], f"input changed: {path.name}:{cell_id}")
        for comparison_name in ("numpy_vs_cython_sample", "torch_vs_cython"):
            comparison = cell[comparison_name]
            require(comparison["preflight_pass"], f"comparison failed: {path.name}:{cell_id}")
            require(comparison["status_equal"], f"status mismatch: {path.name}:{cell_id}")
            require(comparison["finite"], f"nonfinite output: {path.name}:{cell_id}")
        for index, metric in enumerate(metrics):
            maxima[index] = max(maxima[index], cell["torch_vs_cython"][metric])
    return tuple(maxima)


def verify_edges(path: Path) -> None:
    result = load(path)
    require(result["status"] == "pass", f"failed edge result: {path.name}")
    require(set(result["cases"]) == EDGE_CASES, f"edge-case set drift: {path.name}")
    for case_id, case in result["cases"].items():
        require(case["pass"], f"edge case failed: {path.name}:{case_id}")


def verify_compile(path: Path) -> None:
    result = load(path)
    require(result["pass"], f"compile inspection failed: {path.name}")
    require(result["graph_count"] == 1, f"graph-count drift: {path.name}")
    require(result["graph_break_count"] == 0, f"graph break: {path.name}")
    require(result["structured_while_count"] == 2, f"structured-loop drift: {path.name}")


def main() -> int:
    freeze = verify_source_freeze()
    summary = load(SUMMARY)
    require(summary["status"] == "pass", "summary status drift")
    require(
        summary["outcome"] == "preflight-pass-phase0b-decision-deferred",
        "preflight outcome drift",
    )
    require(summary["decision_observations"] == 0, "performance observation drift")
    require(not summary["case_1"], "preflight cannot award Case 1")
    require(not summary["case_2"], "preflight cannot award Case 2")
    require(summary["preflight_freeze_sha256"] == sha256(FREEZE), "freeze hash drift")
    require(
        summary["input_freeze_sha256"] == freeze["input_freeze_sha256"],
        "summary input-freeze drift",
    )
    for relative, wanted in summary["evidence_files"].items():
        require(sha256(EVIDENCE / relative) == wanted, f"evidence drift: {relative}")

    regression = load(EVIDENCE / "regression-result.json")
    require(regression["status"] == "pass", "focused regression failure")
    require(regression["passed"] == 153, "focused regression count drift")
    require(regression["failed"] == 0 and regression["errors"] == 0, "regression failure")

    maxima = [0.0, 0.0, 0.0, 0.0]
    for device in ("cpu", "cuda"):
        eager = verify_grid(EVIDENCE / f"{device}-eager-c01-c02.json", EAGER_CELLS)
        structured = verify_grid(EVIDENCE / f"{device}-structured-grid.json", ALL_CELLS)
        verify_edges(EVIDENCE / f"{device}-edge-cases.json")
        verify_compile(EVIDENCE / f"{device}-compile-inspection.json")
        maxima = [max(current, observed) for current, observed in zip(maxima, eager, strict=True)]
        maxima = [
            max(current, observed) for current, observed in zip(maxima, structured, strict=True)
        ]

    recorded = summary["maximum_torch_vs_cython_difference"]
    for value, key in zip(maxima, ("nll_abs", "nll_rel", "sigma2_abs", "sigma2_rel"), strict=True):
        require(value == recorded[key], f"maximum drift: {key}")

    print(
        json.dumps(
            {
                "status": "pass",
                "structured_device_cells": 32,
                "eager_device_cells": 4,
                "edge_device_cases": 8,
                "existing_tests_passed": 153,
                "graphs": {"cpu": 1, "cuda": 1, "breaks": 0},
                "outcome": summary["outcome"],
                "decision_observations": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
