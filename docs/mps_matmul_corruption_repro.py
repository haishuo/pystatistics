"""Minimal standalone repro for the MPS strided-matmul buffer corruption
documented in GPU_NOTES.md ("MPS strided-matmul buffer corruption").

No pystatistics imports — this is a pure torch/MPS bug demonstration for
version retests and an upstream issue. Measured 2026-08-13 on an M2 Max
(macOS 25.5.0): identical wrong values on torch 2.7.1, 2.9.0, 2.11.0,
2.12.0, 2.12.1.

VERSION CAVEAT (retest, same day): this MINIMAL trigger fires only on torch
<= 2.12.x. On torch 2.13.0 this script prints clean numbers, yet the
underlying bug is STILL PRESENT — the production-shaped polyreg fixture
(GPU_NOTES.md, "MPS strided-matmul buffer corruption") corrupts identically
on 2.13.0. A clean run of this script therefore proves nothing on >= 2.13.
For a self-contained empirical verdict on any torch/macOS combination, use
``pystatistics.mice.diagnostics.mps_matmul_canary()`` — it runs the
production fixture (bisected boundary: 2.14.0.dev20260622 corrupts,
2.14.0.dev20260624+ clean via pytorch #187441, which MASKS the misread with
an allocator-bucketing change rather than fixing the kernel).

Expected output on an affected torch:

    no trigger   : mps err ~9e-05                       (correct)
    after trigger: mps err ~13-28 on scale ~90 (15-30%) (CORRUPTED)
    after trigger + empty_cache: mps err ~9e-05         (correct again)
    contiguous-LHS after trigger: ~1e-03                (correct)

Run:  KMP_DUPLICATE_LIB_OK=TRUE python docs/mps_matmul_corruption_repro.py
"""
import numpy as np
import torch


def victim_err():
    """|A_w' A - fp64 reference|_max for a 50k x 25 weighted Gram, computed
    on MPS from CPU-copied fp32 tensors through a transpose view."""
    rng = np.random.default_rng(0)
    n, p = 50_000, 25
    A = torch.tensor(rng.standard_normal((n, p)), dtype=torch.float32)
    u = rng.random(n)
    # logistic-saturation-like weight profile (any weights work; the plain
    # unweighted A'A corrupts too, see GPU_NOTES)
    w = torch.tensor(
        np.where(u < 0.03, 0.05, 10.0 ** (-8 + 5 * rng.random(n))),
        dtype=torch.float32,
    )
    Aw = A * w.unsqueeze(-1)
    ref = Aw.double().T @ A.double()
    got = (Aw.to("mps").T @ A.to("mps")).cpu().double()
    got_contig = (
        torch.mm(Aw.to("mps").T.contiguous(), A.to("mps")).cpu().double()
    )
    scale = float(ref.diagonal().abs().max())
    return (
        float((got - ref).abs().max()),
        float((got_contig - ref).abs().max()),
        scale,
    )


def trigger():
    """One large prior matmul through the same caching allocator."""
    rng = np.random.default_rng(1)
    big = torch.tensor(rng.standard_normal((200_000, 25)), dtype=torch.float32)
    wb = torch.tensor(rng.random(200_000), dtype=torch.float32)
    ((big * wb.unsqueeze(-1)).to("mps").T @ big.to("mps")).cpu()
    torch.mps.synchronize()


def main():
    assert torch.backends.mps.is_available(), "needs an MPS device"
    e0, _, scale = victim_err()
    print(f"torch {torch.__version__}, Gram scale {scale:.1f}")
    print(f"  no trigger                  : mps err {e0:.3e}")
    trigger()
    e1, e1c, _ = victim_err()
    print(f"  after trigger               : mps err {e1:.3e}"
          + ("   <-- CORRUPTED" if e1 > 1e-2 * scale else ""))
    print(f"  contiguous-LHS after trigger: mps err {e1c:.3e}")
    torch.mps.empty_cache()
    e2, _, _ = victim_err()
    print(f"  after trigger + empty_cache : mps err {e2:.3e}")


if __name__ == "__main__":
    main()
