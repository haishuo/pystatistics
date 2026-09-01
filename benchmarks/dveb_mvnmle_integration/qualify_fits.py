#!/usr/bin/env python3
"""Untimed end-to-end fit qualification for one frozen evaluation lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    affinity,
    array_sha256,
    compare_solutions,
    configure_threads,
    make_case,
    solution_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(f"E{i}" for i in range(1, 7)))
    parser.add_argument("--threads", required=True, type=int, choices=(1, 6, 12))
    args = parser.parse_args()

    cpus = affinity()
    if len(cpus) != args.threads:
        raise SystemExit(f"affinity mismatch: requested {args.threads} CPUs, process has {cpus}")
    data = make_case(args.case)
    input_sha256 = array_sha256(data)
    configure_threads(args.threads)

    from pystatistics.mvnmle import MVNDesign, mlest

    design = MVNDesign.from_array(data)
    torch_result = solution_record(mlest(design, method="direct", backend="cpu"))
    dveb_result = solution_record(mlest(design, method="direct", solver="dveb"))
    # This admission step is deliberately untimed. The public result carries
    # internal timing metadata, so remove it before recording qualification
    # evidence; runtime comparisons belong only to the frozen campaign.
    torch_result.pop("timing", None)
    dveb_result.pop("timing", None)
    comparison = compare_solutions(torch_result, dveb_result)
    input_sha256_after = array_sha256(data)
    input_unchanged = input_sha256_after == input_sha256
    payload = {
        "schema": "pystatistics.dveb-mvnmle.fit-qualification.v1",
        "case": args.case,
        "threads": args.threads,
        "affinity": cpus,
        "input_sha256": input_sha256,
        "input_sha256_after": input_sha256_after,
        "input_unchanged": input_unchanged,
        "torch": torch_result,
        "dveb": dveb_result,
        "comparison": comparison,
        "pass": comparison["pass"] and input_unchanged,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
