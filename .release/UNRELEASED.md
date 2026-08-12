# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **REVERTED the 6.1.0 retirement of the MICE MPS merge-rank bridge**
  (`mice/backends/_gpu_methods.py::_insertion_rank`) — 6.1.0's switch to
  native `searchsorted` on torch >= 2.13 was a correctness regression on
  heavily tied data. The "±1 tie convention" that justified the swap holds
  only for near-unique values: on tied survey columns the bridge behaves like
  `side='right'` while native `searchsorted` defaults to `side='left'`, so
  the PMM donor-window seed shifts by whole tie-block widths and different
  (individually valid) donors are drawn. On the float32-only MPS path the
  changed chained-equations trajectory collapsed real-survey mixed-type fits
  into non-finite refusals: CSES n=10,000 (m=20, maxit=10) completes in
  ~45 s under 6.0.1 and under this revert, but was refused by 6.1.0 after a
  ~20-minute degenerate grind (isolated by A/B with `mps_native_kernels`
  monkeypatching; the always-series-inverse change was tested the same way
  and exonerated — it stays). CUDA is unaffected (it has always used native
  `searchsorted` and runs its discrete-GLM fits in float64). Do not retire
  the bridge again without a robustness fix that lets fp32 categorical fits
  survive either donor trajectory.
- **Known issue (pre-existing, NOT from 6.1.x): GSS-on-MPS mixed-type MICE
  is refused since 6.0.1.** Bisection (same data, seed 20260617, n=10,000,
  m=20, maxit=10, torch 2.13.0, M2 Max): 3.16.3 and 6.0.0 complete in
  ~115 s; 6.0.1, 6.0.2, and 6.1.0 refuse with the end-of-sweep non-finite
  guard after the same ~115 s. The flip is exactly at 6.0.1 ("Fix 12
  mice/mvnmle correctness defects"). Undetermined whether this is an honest
  refusal (6.0.1 exposing a genuinely degenerate fp32 fit whose earlier
  "successful" output was never fidelity-validated — GSS-MPS has no R
  baseline because R's own mice fails on this host) or a false refusal (an
  over-tightened guard, an A6 violation) — deciding requires identifying
  which of the twelve 6.0.1 fixes fires and auditing 6.0.0's completed GSS
  imputations for silent collapse.
