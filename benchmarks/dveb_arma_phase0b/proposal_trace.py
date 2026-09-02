"""Deterministic L3 proposal traces for exact-ARMA Phase-0B."""

from __future__ import annotations

import hashlib

import numpy as np
from common import CELLS, MASTER_SEED, ar_root_moduli, generate_cell
from numpy.typing import NDArray

PROPOSALS = 100
SERIES_BUCKETS = 17
AR_RELATIVE_RADIUS = 2.0e-4
AR_ZERO_RADIUS = 1.0e-5
LOADING_RELATIVE_RADIUS = 2.0e-3
ROOT_FLOOR = 1.0001


def trace_seed(cell_id: str) -> int:
    digest = hashlib.sha256(f"{MASTER_SEED}:proposal-trace:{cell_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def proposal_trace(cell_id: str) -> tuple[NDArray, NDArray, float]:
    """Return `(phi_trace, loading_trace, minimum_root_modulus)`."""
    if cell_id not in CELLS:
        raise ValueError(f"unknown frozen cell: {cell_id}")
    _, base_phi, base_loading = generate_cell(cell_id)
    k, r = base_phi.shape
    phi_one = base_phi[0]
    loading_one = base_loading[0]
    phi_trace = np.empty((PROPOSALS, k, r), dtype=np.float64)
    loading_trace = np.empty_like(phi_trace)
    phi_trace[0] = base_phi
    loading_trace[0] = base_loading
    minimum_root = float(ar_root_moduli(tuple(phi_one)).min())
    rng = np.random.Generator(np.random.PCG64DXSM(trace_seed(cell_id)))
    row_bucket = np.arange(k) % SERIES_BUCKETS

    for proposal in range(1, PROPOSALS):
        ar_noise = rng.uniform(-1.0, 1.0, (SERIES_BUCKETS, r))
        ar_radius = AR_RELATIVE_RADIUS * np.abs(phi_one) + AR_ZERO_RADIUS
        candidates = phi_one + ar_noise * ar_radius
        for bucket in range(SERIES_BUCKETS):
            root = float(ar_root_moduli(tuple(candidates[bucket])).min())
            if root < ROOT_FLOOR:
                raise RuntimeError(f"{cell_id} proposal {proposal} bucket {bucket} root {root}")
            minimum_root = min(minimum_root, root)
        phi_trace[proposal] = candidates[row_bucket]

        loading_noise = rng.uniform(-1.0, 1.0, (SERIES_BUCKETS, r))
        loading_radius = LOADING_RELATIVE_RADIUS * np.maximum(np.abs(loading_one), 0.05)
        loading_candidates = loading_one + loading_noise * loading_radius
        loading_trace[proposal] = loading_candidates[row_bucket]

    return (
        np.ascontiguousarray(phi_trace),
        np.ascontiguousarray(loading_trace),
        minimum_root,
    )
