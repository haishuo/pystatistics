# Phase-II process attempts

## Calibration correctness attempt 1 — invalid checker, preserved

`calibration-correctness.json` was produced after implementation commit
`78fa2cf` and before any evaluation or comparative timing.  Its DVEB FD-F
gradient admission was invalid: the checker compared the declared forward
`h=1e-8` gradient against a Cython **central** `h=1e-5` calculation.  The
frozen protocol instead requires each DVEB finite-difference policy to agree
with the **same formula** evaluated through Cython.

The attempt is retained verbatim.  It must never be used for selection or a
Phase-II decision.  The corrected runner writes a distinct v2 output, uses
forward/`1e-8` for FD-F, central/`1e-5` for FD-C, and leaves every frozen
tolerance, candidate, cell, start, and decision rule unchanged.

The attempt did establish one valid diagnostic outside the erroneous check:
an independent rerun of the worst C02 D1-C row exactly reproduced both the
coordinated D1-C result and the independently fitted S0 result.  Thus the
observed difference between those derivative policies was not introduced by
barrier batching.
