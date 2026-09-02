#!/usr/bin/env python3
"""Record public structured-loop graph capture without performance timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from common import generate_cell
from torch_impl import likelihood_while_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no fallback attempted")

    z, phi, loading = generate_cell("C01")
    tensors = tuple(torch.from_numpy(value).to(args.device) for value in (z, phi, loading))
    explanation = torch._dynamo.explain(likelihood_while_loop)(*tensors)
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.compile-inspection.v1",
        "device": args.device,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "op_count": explanation.op_count,
        "break_reasons": [str(item) for item in explanation.break_reasons],
        "ops_per_graph": [
            [str(operation) for operation in graph] for graph in explanation.ops_per_graph
        ],
        "structured_while_count": sum(
            str(operation) == "while_loop"
            for graph in explanation.ops_per_graph
            for operation in graph
        ),
        "pass": explanation.graph_count == 1 and explanation.graph_break_count == 0,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "pass": result["pass"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
