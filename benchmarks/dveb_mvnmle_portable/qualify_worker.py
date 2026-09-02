#!/usr/bin/env python3
"""Untimed installed-wheel fit qualification for one frozen lane."""

from __future__ import annotations

import argparse
import json

from portable_common import (
    affinity,
    array_sha256,
    compare_solutions,
    configure_threads,
    make_case,
    solution_record,
    verify_installed_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(f"E{i}" for i in range(1, 7)))
    parser.add_argument("--threads", required=True, type=int, choices=(1, 6, 12))
    args = parser.parse_args()
    cpus = affinity()
    if len(cpus) != args.threads:
        raise SystemExit(f"affinity mismatch: requested {args.threads}, got {cpus}")
    data = make_case(args.case)
    input_sha256 = array_sha256(data)
    configure_threads(args.threads)
    identity = verify_installed_identity()

    from pystatistics.mvnmle import MVNDesign, mlest

    design = MVNDesign.from_array(data)
    torch_result = solution_record(mlest(design, method="direct", backend="cpu"))
    dveb_result = solution_record(mlest(design, method="direct", solver="dveb"))
    torch_result.pop("timing", None)
    dveb_result.pop("timing", None)
    comparison = compare_solutions(torch_result, dveb_result)
    unchanged = array_sha256(data) == input_sha256
    payload = {
        "schema": "pystatistics.dveb-portable.fit-qualification.v1",
        "case": args.case,
        "threads": args.threads,
        "affinity": cpus,
        "input_sha256": input_sha256,
        "input_unchanged": unchanged,
        "installed_identity": identity,
        "torch": torch_result,
        "dveb": dveb_result,
        "comparison": comparison,
        "pass": comparison["pass"] and unchanged,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
