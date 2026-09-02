#!/usr/bin/env python3
"""Fresh-process import-to-fit probe for one installed implementation."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import resource
import sys
import time

from portable_common import affinity, make_case, verify_installed_identity


class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch blocked during portable-DVEB operational probe")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("torch", "dveb"), required=True)
    args = parser.parse_args()
    if len(affinity()) != 1:
        raise SystemExit("operational worker requires one-CPU affinity")
    data = make_case("E1")
    if args.implementation == "dveb":
        sys.meta_path.insert(0, _BlockTorch())

    started_wall_ns = time.time_ns()
    started = time.perf_counter_ns()
    identity = verify_installed_identity()
    if args.implementation == "torch":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    from pystatistics.mvnmle import mlest

    if args.implementation == "torch":
        result = mlest(data, method="direct", backend="cpu")
    else:
        result = mlest(data, method="direct", solver="dveb")
    elapsed_ns = time.perf_counter_ns() - started
    payload = {
        "schema": "pystatistics.dveb-portable.operational-observation.v1",
        "implementation": args.implementation,
        "seconds": elapsed_ns / 1.0e9,
        "started_wall_ns": started_wall_ns,
        "finished_wall_ns": time.time_ns(),
        "converged": bool(result.converged),
        "loglik": float(result.loglik),
        "backend": result.backend_name,
        "torch_imported": "torch" in sys.modules,
        "installed_identity": identity,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.converged and (args.implementation == "torch" or not payload["torch_imported"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
