#!/usr/bin/env python3
"""One fresh-process E07 likelihood evaluation for the L4 endpoint."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np

from common import generate_cell, sha256_array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("t3-cuda", "ng-auto"), required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    args = parser.parse_args()
    z, phi, loading = generate_cell("E07")
    before = tuple(sha256_array(value) for value in (z, phi, loading))

    if args.implementation == "t3-cuda":
        import torch

        from torch_impl import compile_while_loop

        device = tuple(torch.from_numpy(value).to("cuda") for value in (z, phi, loading))
        function = compile_while_loop()
        device_output = function(*device)
        torch.cuda.synchronize()
        output = tuple(value.detach().cpu().numpy() for value in device_output)
    else:
        from native_runtime import NativeCUDA

        with NativeCUDA(args.cuda_library, z, phi, loading) as context:
            output = context.evaluate(block_size=args.block)

    after = tuple(sha256_array(value) for value in (z, phi, loading))
    result = {
        "implementation": args.implementation,
        "input_immutable": before == after,
        "input_sha256": before,
        "nll": output[0].tolist(),
        "sigma2": output[1].tolist(),
        "status": output[2].astype(np.uint8).tolist(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
