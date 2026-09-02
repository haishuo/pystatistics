# DVEB exact-ARMA Phase-II fit-suitability result

**Date:** 2026-09-02

**Outcome:** **HALT / Case 2 at calibration correctness.**  No implementation
selection, evaluation-cell run, admitted campaign, or comparative timing was
legal after this result.

## What was tested

The protocol at commit `46e321c` asked whether the immutable, already-qualified
DVEB fixed-likelihood artifact could support statistically equivalent
many-series exact-ML fitting without a new DVEB feature.  Inputs, Yule--Walker
starts, environment, optimizer semantics, and implementation identities were
frozen at `64fad11`; the private coordinator and S0/PyTorch/DVEB routes were
implemented at `17423d8` and `78fa2cf`.

Only the disjoint calibration cells C01--C04 were observed.  Each complete-fit
check was repeated from the same start, and every reported route was
deterministic.  The evaluation cells E01--E12 were not opened.

## Gradient admission

| Route | Result | Reason |
|---|---|---|
| D1-F / D2-F | FAIL | The `h=1e-8` forward differences amplify independent likelihood roundoff; F3 and F13 exceed the frozen `rtol=1e-5, atol=1e-5` same-formula Cython check.  Maximum absolute discrepancies were `6.82e-5` and `3.64e-4`. |
| D1-C / D2-C | PASS | Central `h=1e-5` gradients pass all four families. |
| TC1 / TG1 | PASS | Eager public-op PyTorch autograd passes all four Cython central-difference audits. |
| TC2 / TG2 | FAIL | The public structured-while forward value is correct, but its autograd result is identically zero for every tested element.  No graph-break or eager fallback was substituted. |

An initial record used the wrong authority for FD-F (central rather than
forward).  It is marked invalid, compressed, and preserved with its exact hash.
The corrected v2 run uses the predeclared matching formula; FD-F still fails,
for the legitimate roundoff-amplification reason above.  No threshold changed.

## Complete-fit admission

Only gradient-admitted routes were fitted.  S0 repeated exactly at every cell.

| Cell | S0 successes | D1-C | D2-C | TC1 | TG1 |
|---|---:|---|---|---|---|
| C01 | 8/8 | PASS | PASS | FAIL | FAIL |
| C02 | 64/64 | FAIL | FAIL | FAIL | FAIL |
| C03 | 256/256 | PASS | PASS | FAIL | FAIL |
| C04 | 1022/1024 | FAIL | FAIL | FAIL | FAIL |

The DVEB central-difference routes reproduce C01 and C03 closely, then fail
the frozen S0-equivalence contract at C02/C04.  At C02, D1-C keeps the same
success mask and remains within the coefficient/NLL bounds, but misses the
sigma2 bound; D2-C additionally differs on one success row.  At C04 both routes
change failure membership and reach maximum coefficient/NLL differences of
about `0.393` and `2.915`.

Eager PyTorch AD is internally CPU/CUDA consistent and passes its derivative
audits, but its optimizer trajectories land at materially different optima
from the incumbent finite-difference S0 fits.  Maximum coefficient/NLL
differences grow from about `0.00386`/`0.00198` at C01 to `1.030`/`6.787` at
C04, with especially large C03 NLL differences (`38.51`).  Correct derivatives
alone therefore do not satisfy this protocol's deliberately strict
incumbent-fit equivalence rule on the nonconvex likelihood.

## Coordinator audit

The worst C02 D1-C row was rerun independently outside the 64-row barrier.
The standalone D1-C parameters reproduced the coordinated parameters bit for
bit, and standalone S0 reproduced coordinated S0 bit for bit.  The two policies
still ended at distinct points (`692.2228218824683` versus
`692.2221539429454` NLL).  Barrier batching did not create the difference.

## Mechanical consequence

No predeclared selection exists:

- FD-F is not gradient-admitted;
- FD-C is not complete-fit-admitted across C01--C04;
- eager PyTorch is not complete-fit-admitted;
- compiled structured-while PyTorch is not gradient-admitted.

Therefore there is no legal D1, D2, PyTorch-CPU, or PyTorch-CUDA selection.
Calibration timing was not run, evaluation cells were not observed, an
admitted-campaign manifest was not created, and the 30-observation performance
campaign did not run.  Performance thresholds were never applied.

Under the frozen protocol this is **HALT / Case 2**, not performance Case 1.
The fixed likelihood remains numerically qualified, but the ABI emits no
derivative.  The incumbent-compatible forward step is too sensitive to
independent-compiler roundoff, while the numerically admitted central step
changes optimizer outcomes beyond the frozen equivalence contract.  A future
attempt requires a separately approved, general DVEB derivative capability or
owned/resident parameter-data bridge protocol.  This study may not be resumed
by changing bounds, swapping optimizers, or selecting by cell after seeing
these results.

## Evidence

- Valid corrected record: `benchmarks/dveb_arma_fit_phase2/calibration-correctness.v2.json.gz`
  (gzip SHA-256 `114c6adc7cf36b942a6ecb5095adf80809009ba8508574405a72e36df62596e1`;
  uncompressed SHA-256 `67da57674062521374a91dd6aae515615886b17b273e8c522085fbffe807fe1f`).
- Invalid first attempt, preserved but inadmissible:
  `calibration-correctness.invalid-attempt1.json.gz` (gzip SHA-256
  `57e99dd245604bf8eb9c7287896fb9b15adc814778d81f7dc308fc2e70c1dad0`).
- Machine-readable selection/halt record: `calibration-selection.json`.
- Offline verifier: `verify_calibration_halt.py`.

The PyStatistics public API and protected main branch were never changed.
DVEB itself was not changed.  Nothing in this result is a claim about a public
backend, paper release, merge, tag, PyPI package, or general DVEB compiler
failure.
