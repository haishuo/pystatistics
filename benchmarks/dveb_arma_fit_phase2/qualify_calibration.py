#!/usr/bin/env python3
"""Run correctness admission on calibration cells only; never times evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE.parent)]

from benchmarks.dveb_arma_fit_phase2.backends import (  # noqa: E402
    DVEBFiniteDifferenceBackend,
    FD_CENTRAL,
    FD_FORWARD,
    TorchAutogradBackend,
    expand_fit_parameters,
)
from benchmarks.dveb_arma_fit_phase2.common import (  # noqa: E402
    CALIBRATION_CELLS,
    FIT_FAMILIES,
    generate_cell,
    generate_starts,
    minimum_ar_root,
    sha256_array,
    sha256_file,
    truth_vector,
)
from benchmarks.dveb_arma_fit_phase2.fit import (  # noqa: E402
    CythonRowObjective,
    FitBatch,
    fit_coordinated,
    fit_s0,
)

OUTPUT = HERE / "calibration-correctness-v2.json"
COMBINED = Path("/mnt/artifacts/dveb/trunk009_exact_arma_20260902/exact_arma_abi_v1.so")
EPS = np.finfo(np.float64).eps


def tolerance(reference: float, n: int, state: int) -> float:
    return max(32.0 * EPS * n * max(1, state), 1.0e-14) * max(1.0, abs(reference))


def cython_value_gradient(z, parameters, family_id, *, step, central):
    values = np.empty(parameters.shape[0], dtype=np.float64)
    gradients = np.empty_like(parameters)
    sigma2 = np.empty_like(values)
    status = np.empty(parameters.shape[0], dtype=np.bool_)
    for row, theta in enumerate(parameters):
        objective = CythonRowObjective(z[row], family_id)
        values[row], sigma2[row], status[row] = objective.evaluate(theta)
        for column in range(theta.size):
            plus = theta.copy()
            plus[column] += step
            if central:
                minus = theta.copy()
                minus[column] -= step
                displacement = plus[column] - minus[column]
                gradients[row, column] = (objective(plus) - objective(minus)) / displacement
            else:
                displacement = plus[column] - theta[column]
                gradients[row, column] = (objective(plus) - values[row]) / displacement
    return values, gradients, sigma2, status


def interior_points(family_id: str):
    truth = truth_vector(family_id)
    rng = np.random.Generator(
        np.random.PCG64DXSM(int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:8], "little"))
    )
    result = [truth.copy()]
    while len(result) < 21:
        candidate = truth + rng.uniform(-0.04, 0.04, truth.size)
        if minimum_ar_root(candidate, family_id) >= 1.01:
            result.append(candidate)
    return np.ascontiguousarray(result)


def backend_factory(route, z, family_id):
    if route == "D1-F":
        return DVEBFiniteDifferenceBackend(z, family_id, FD_FORWARD, route="D1")
    if route == "D1-C":
        return DVEBFiniteDifferenceBackend(z, family_id, FD_CENTRAL, route="D1")
    if route == "D2-F":
        return DVEBFiniteDifferenceBackend(
            z, family_id, FD_FORWARD, route="D2", combined_library=COMBINED
        )
    if route == "D2-C":
        return DVEBFiniteDifferenceBackend(
            z, family_id, FD_CENTRAL, route="D2", combined_library=COMBINED
        )
    identities = {
        "TC1": ("cpu", False), "TC2": ("cpu", True),
        "TG1": ("cuda", False), "TG2": ("cuda", True),
    }
    device, compiled = identities[route]
    return TorchAutogradBackend(z, family_id, device=device, compiled=compiled)


def close(backend):
    if hasattr(backend, "close"):
        backend.close()


def gradient_audit(route, family_id, calibration_cell):
    z_source, _, _ = generate_cell(calibration_cell)
    parameters = interior_points(family_id)
    z = np.ascontiguousarray(np.repeat(z_source[:1], len(parameters), axis=0))
    forward = route in {"D1-F", "D2-F"}
    reference, gradient, _sigma2, status = cython_value_gradient(
        z, parameters, family_id,
        step=1.0e-8 if forward else 1.0e-5,
        central=not forward,
    )
    backend = backend_factory(route, z, family_id)
    try:
        actual, actual_gradient = backend.value_and_gradient(
            tuple(range(len(parameters))), parameters
        )
    except BaseException as exc:
        return {"pass": False, "refusal": f"{type(exc).__name__}: {exc}"}
    finally:
        close(backend)
    likelihood_error = np.abs(actual - reference)
    likelihood_tolerance = np.asarray(
        [tolerance(value, z.shape[1], FIT_FAMILIES[family_id].state) for value in reference]
    )
    gradient_rtol = 1.0e-5 if route.startswith("D") else 1.0e-4
    gradient_atol = 1.0e-5 if route.startswith("D") else 1.0e-6
    gradient_close = np.isclose(
        actual_gradient, gradient, rtol=gradient_rtol, atol=gradient_atol
    )
    passed = bool(
        status.all() and np.all(likelihood_error <= likelihood_tolerance)
        and gradient_close.all()
    )
    return {
        "pass": passed,
        "points": len(parameters),
        "max_likelihood_abs_error": float(likelihood_error.max()),
        "max_likelihood_tolerance": float(likelihood_tolerance.max()),
        "max_gradient_abs_error": float(np.max(np.abs(actual_gradient - gradient))),
        "gradient_rtol": gradient_rtol,
        "gradient_atol": gradient_atol,
        "gradient_elements_failed": int(np.size(gradient_close) - np.count_nonzero(gradient_close)),
    }


def fit_equal(first: FitBatch, second: FitBatch) -> bool:
    return bool(
        np.array_equal(first.success, second.success)
        and np.array_equal(first.parameters, second.parameters)
        and np.array_equal(first.nll, second.nll)
        and np.array_equal(first.sigma2, second.sigma2)
    )


def check_fit(reference: FitBatch, candidate: FitBatch, z, starts, family_id):
    success_equal = np.array_equal(reference.success, candidate.success)
    selected = reference.success & candidate.success
    coefficient_error = (
        float(np.max(np.abs(reference.parameters[selected] - candidate.parameters[selected])))
        if np.any(selected) else 0.0
    )
    nll_error = (
        float(np.max(np.abs(reference.nll[selected] - candidate.nll[selected])))
        if np.any(selected) else 0.0
    )
    sigma_close = bool(
        np.allclose(reference.sigma2[selected], candidate.sigma2[selected], rtol=1e-4, atol=1e-8)
    )
    reported_errors = np.abs(
        np.asarray([row.reported_nll for row in candidate.rows]) - candidate.nll
    )
    reported_tolerances = np.asarray(
        [tolerance(value, z.shape[1], FIT_FAMILIES[family_id].state) for value in candidate.nll]
    )
    reported_sigma_close = np.allclose(
        np.asarray([row.reported_sigma2 for row in candidate.rows]),
        candidate.sigma2, rtol=1e-4, atol=1e-8,
    )
    reported_status = np.asarray([row.reported_status for row in candidate.rows])
    phi, loading = expand_fit_parameters(candidate.parameters, family_id)
    spec = FIT_FAMILIES[family_id]
    fixed_ar = [index for index in range(spec.state) if index not in spec.ar_free]
    fixed_ma = [index for index in range(spec.state - 1) if index not in spec.ma_free]
    fixed_zero = bool(
        (not fixed_ar or np.array_equal(phi[:, fixed_ar], np.zeros_like(phi[:, fixed_ar])))
        and (not fixed_ma or np.array_equal(
            loading[:, np.asarray(fixed_ma) + 1],
            np.zeros_like(loading[:, np.asarray(fixed_ma) + 1]),
        ))
    )
    stationary = all(
        minimum_ar_root(row.parameters, family_id) > 1.0
        for row in candidate.rows if row.success
    )
    finite = all(
        np.isfinite(row.parameters).all() and np.isfinite(row.nll) and np.isfinite(row.sigma2)
        for row in candidate.rows if row.success
    )
    passed = bool(
        success_equal and coefficient_error <= 5e-3 and nll_error <= 0.05
        and sigma_close and np.all(reported_errors <= reported_tolerances)
        and reported_sigma_close and np.array_equal(reported_status, np.ones_like(reported_status))
        and fixed_zero and stationary and finite
    )
    return {
        "pass": passed,
        "success_masks_equal": bool(success_equal),
        "successes": int(candidate.success.sum()),
        "maximum_coefficient_abs_error": coefficient_error,
        "maximum_nll_abs_error": nll_error,
        "sigma2_close": sigma_close,
        "maximum_reported_nll_abs_error": float(reported_errors.max()),
        "maximum_reported_nll_tolerance": float(reported_tolerances.max()),
        "reported_sigma2_close": bool(reported_sigma_close),
        "reported_status_all_valid": bool(reported_status.all()),
        "fixed_zeros": fixed_zero,
        "stationary": stationary,
        "finite": finite,
    }


def fit_route(route, z, starts, family_id):
    if route == "S0":
        return fit_s0(z, starts, family_id)
    backend = backend_factory(route, z, family_id)
    try:
        return fit_coordinated(z, starts, family_id, route, backend)
    finally:
        close(backend)


def main():
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    route_order = ("D1-F", "D1-C", "D2-F", "D2-C", "TC1", "TC2", "TG1", "TG2")
    family_cells = {"F1": "C01", "F3": "C02", "F13": "C03", "F25": "C04"}
    gradients = {}
    admitted = {}
    for route in route_order:
        gradients[route] = {}
        for family_id, cell_id in family_cells.items():
            check = gradient_audit(route, family_id, cell_id)
            gradients[route][family_id] = check
            print("gradient", route, family_id, "PASS" if check["pass"] else "FAIL", flush=True)
        admitted[route] = all(check["pass"] for check in gradients[route].values())

    cells = {}
    for cell_id in CALIBRATION_CELLS:
        z, _phi, _loading = generate_cell(cell_id)
        starts = generate_starts(cell_id)
        before = (sha256_array(z), sha256_array(starts))
        family_id = next(
            family_id for family_id, spec in FIT_FAMILIES.items()
            if spec.state == CELLS_STATE[cell_id]
        )
        reference = fit_route("S0", z, starts, family_id)
        reference_repeat = fit_route("S0", z, starts, family_id)
        row = {
            "S0": {"pass": fit_equal(reference, reference_repeat), "deterministic": fit_equal(reference, reference_repeat), "fit": reference.record()}
        }
        print("fit", cell_id, "S0", "PASS" if row["S0"]["pass"] else "FAIL", flush=True)
        for route in route_order:
            if not admitted[route]:
                row[route] = {"pass": False, "not_run": "gradient admission failed"}
                continue
            candidate = fit_route(route, z, starts, family_id)
            repeat = fit_route(route, z, starts, family_id)
            check = check_fit(reference, candidate, z, starts, family_id)
            check["deterministic"] = fit_equal(candidate, repeat)
            check["pass"] = bool(check["pass"] and check["deterministic"])
            check["fit"] = candidate.record()
            row[route] = check
            print("fit", cell_id, route, "PASS" if check["pass"] else "FAIL", flush=True)
        unchanged = before == (sha256_array(z), sha256_array(starts))
        cells[cell_id] = {"inputs_unchanged": unchanged, "routes": row, "pass": bool(unchanged and all(item["pass"] for key, item in row.items() if key == "S0" or admitted.get(key, False)))}

    result = {
        "schema": "pystatistics.dveb-arma-fit-phase2.calibration-correctness.v1",
        "status": "complete",
        "source": {
            "implementation_commit": subprocess_git("rev-parse", "HEAD"),
            "script_sha256": sha256_file(Path(__file__)),
            "input_freeze_sha256": sha256_file(HERE / "input-start-freeze.json"),
        },
        "gradients": gradients,
        "gradient_admitted": admitted,
        "cells": cells,
        "pass": bool(all(cell["pass"] for cell in cells.values())),
        "evaluation_cells_observed": False,
        "timing_results_exist": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("OVERALL", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


def subprocess_git(*arguments):
    import subprocess
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


CELLS_STATE = {cell_id: generate_cell(cell_id)[1].shape[1] for cell_id in CALIBRATION_CELLS}


if __name__ == "__main__":
    raise SystemExit(main())
