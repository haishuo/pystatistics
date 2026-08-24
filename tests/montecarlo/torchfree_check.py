#!/usr/bin/env python3
"""The torch-free proof, as a runnable script rather than a claim.

Not collected by pytest: it must run in an environment that pytest's own
environment is not -- an installed wheel with PyStatistics' base dependencies
and NO PyTorch. Build a wheel, install it into a fresh venv, and run this:

    python -m pip wheel . --no-deps -w /tmp/psbuild
    python -m venv /tmp/psvenv && /tmp/psvenv/bin/pip install /tmp/psbuild/*.whl
    cd /tmp && /tmp/psvenv/bin/python <path-to>/torchfree_check.py

Exits nonzero on any failure. What it establishes, with torch neither installed nor imported throughout:

  * the eligible mean bootstrap succeeds through packaged ALLOY;
  * a missing `gpu_statistic='mean'` raises the existing DECLARATION error,
    not "No GPU available ... install PyTorch";
  * an unsupported method or configuration raises the existing CONFIGURATION
    error, for the same reason;
  * `auto`, the CPU path, and the shipped licence notice are unmoved.
"""

import importlib.util
import pathlib
import sys

import numpy as np


def mean_stat(data, indices):
    return np.array([np.mean(data[indices])])


def main() -> int:
    if importlib.util.find_spec("torch") is not None:
        print("FAIL: torch is installed; this check needs an environment "
              "without it", file=sys.stderr)
        return 2

    import pystatistics
    from pystatistics.core.exceptions import ValidationError
    from pystatistics.montecarlo import boot

    pkg = pathlib.Path(pystatistics.__file__).resolve().parent
    if "Projects/Dev" in str(pkg):
        print(f"FAIL: importing the source tree, not an installed package "
              f"({pkg})", file=sys.stderr)
        return 2

    data = np.arange(1.0, 51.0)
    r = boot(data, mean_stat, n_resamples=4000, seed=42, backend="gpu",
             gpu_statistic="mean")

    fails = []
    if r.backend_name != "gpu_mps_alloy_bootstrap":
        fails.append(f"backend_name is {r.backend_name!r}")
    if "torch" in sys.modules:
        fails.append("torch was imported during selection")
    if r.t0[0] != 25.5:
        fails.append(f"t0 is {r.t0[0]}")
    se = float(np.std(data, ddof=1)) / np.sqrt(len(data))
    if not 0.85 < r.standard_errors[0] / se < 1.15:
        fails.append(f"SE/analytic is {r.standard_errors[0] / se}")
    again = boot(data, mean_stat, n_resamples=4000, seed=42, backend="gpu",
                 gpu_statistic="mean")
    if not np.array_equal(r.t, again.t):
        fails.append("a fixed seed did not reproduce")

    # ERROR PATHS, which are the half this check was extended to cover. An
    # unsupported configuration is not a missing device, and on a torch-free
    # machine the resolver used to say it was.
    def expect(exc_type, fragment, label, **kwargs):
        try:
            boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu",
                 **kwargs)
        except exc_type as exc:
            if fragment not in str(exc):
                fails.append(f"{label}: wrong message: {exc}")
        except Exception as exc:  # noqa: BLE001
            fails.append(f"{label}: raised {type(exc).__name__}: {exc}")
        else:
            fails.append(f"{label}: did not raise")

    expect(ValidationError, "requires gpu_statistic='mean'",
           "missing declaration")
    expect(ValidationError, "supports only", "balanced method",
           gpu_statistic="mean", method="balanced")
    expect(ValidationError, "supports only", "frequency statistic",
           gpu_statistic="mean", statistic_type="frequency")
    expect(ValidationError, "supports only", "strata",
           gpu_statistic="mean", strata=np.repeat([0, 1], 25))

    # Nothing else moved.
    if "cpu" not in boot(data, mean_stat, n_resamples=200, seed=1,
                         backend="cpu").backend_name:
        fails.append("the CPU path changed")
    if "cpu" not in boot(data, mean_stat, n_resamples=200, seed=1,
                         backend="auto", gpu_statistic="mean").backend_name:
        fails.append("auto no longer resolves to the CPU on Apple Silicon")

    lic = pkg / "montecarlo/backends/_alloy/LICENSE-ALLOY"
    if not lic.is_file():
        fails.append("LICENSE-ALLOY is not in the installed package")

    # Asserted LAST as well as first: every branch above must have stayed off
    # PyTorch, including the ones that only raise.
    if "torch" in sys.modules:
        fails.append("torch was imported somewhere in this check")

    for line in (f"package        : {pkg}",
                 f"torch installed: {importlib.util.find_spec('torch') is not None}",
                 f"torch imported : {'torch' in sys.modules}",
                 f"backend_name   : {r.backend_name}",
                 f"t0             : {r.t0[0]}",
                 f"SE / analytic  : {r.standard_errors[0] / se:.4f}",
                 f"error paths    : declaration + configuration refusals "
                 f"checked, torch still unimported"):
        print("  " + line)

    if fails:
        print("\nFAIL:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nPASS -- the eligible explicit-GPU bootstrap runs on packaged "
          "ALLOY with no PyTorch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
