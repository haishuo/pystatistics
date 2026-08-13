# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Fixed the GSS mixed-type MICE refusal on Apple Silicon (the 6.1.1 "known
  issue"): the GPU `logreg` now carries the same genuine separation penalty as
  the CPU method.** The 6.0.1 CPU fix (penalty in the *gradient*, toward the
  marginal-rate intercept) was never ported to `_gpu_logreg.py`, which kept the
  pre-6.0.1 Hessian-only ridge: at convergence the batched IRLS solved the
  UNPENALISED score equation, so a (quasi-)completely separated binary column
  still diverged. On CUDA the fp64 fit hides this; on fp32-only MPS the
  diverging fit goes non-finite. On real GSS data (n=10,000, m=20, maxit=10,
  seed 20260617) the "case was retrieved" flag is a deterministic recode of the
  occupation-coding-status columns — every sweep fit of that column is
  completely separated — and ~30% of chain-sweep logreg fits (up to 9 of 20
  chains in a sweep) went all-NaN, which the end-of-sweep non-finite guard
  correctly refused since 6.0.1 (`ValidationError: GPU MICE produced non-finite
  imputations`). `batched_logistic_irls` now mirrors the CPU
  `methods/logreg._fit_logistic` penalty exactly (slope ridge
  `1e-5 * mean-column-second-moment * n`, intercept anchored to the marginal
  logit at `1e-5 * n`, `beta0` start), plus a non-finite step guard mirroring
  the polr Newton, so a separated fit shrinks toward the marginal rate and
  stays finite. GSS-on-MPS now completes (~2 min at the config above) with
  imputed marginals matching the CPU reference (max column gap ≤3.6% except
  one near-degenerate column at 9.6%, within its seed-to-seed spread); the
  batched fit matches the CPU `_fit_logistic` to 5.6e-17 in fp64. Full mice
  suite: 240 passed / 24 skipped, including new complete-separation
  regression tests (`TestMpsLogregCompleteSeparation` in `test_gpu_mps.py`;
  a device-parametrized twin in `test_gpu_glm_separation.py` covers CUDA)
  that fail on the pre-fix code, plus the existing R-fixture and CPU-parity
  gates. CSES full config re-verified completing on MPS post-fix. CUDA
  re-verified on an RTX 5070 Ti (torch 2.13.0+cu130): 224 passed / 40
  skipped, and the same GSS config completes in ~21 s with marginals tracking
  the CPU reference. The same fix also resolves the CSES n=50,000 MPS refusal
  surfaced by the 6.1.1 benchmark grid (same flip at 6.0.1, same mechanism:
  two extremely imbalanced binary "expert ideology" columns degenerate in all
  20 chains from the first sweep at n=50k while surviving at n=10k; verified
  by instrumentation and an isolated penalty-only run — CSES has no unordered
  columns, so this cell is logreg alone). 6.0.0's CSES@50k GPU "completions"
  were corrupted the same way as GSS's (those cells silently imputed as the
  ~3% minority class). Full analysis: `docs/GSS_MPS_MICE_AUDIT_2026-08.md`.

- **The same defect silently distorted GSS on CUDA (no refusal) in every
  version before this fix.** On CUDA the discrete fits run in fp64, which
  kept the unpenalized separated logreg finite — but stalled far from any
  optimum, with a garbage posterior-draw covariance in the saturated
  directions. Measured on the GSS configuration above, pre-fix CUDA imputed
  the "case was retrieved" flag at 44.6% minority class vs 8.0% observed
  (penalized CPU reference: 17.1%) and distorted every downstream column,
  while completing normally. GSS `backend='gpu'` MICE results produced on
  CUDA under any earlier version should be treated as suspect, the same as
  the Apple-GPU results under <= 6.0.0 (there the collapse was total).

- **Globalized the logreg fit (CPU and GPU) with the family's line search, and
  made the GPU Gram factorization robust at survey scale — together resolving
  the GSS n=50,000 GPU refusal that survived the penalty alone.** Three
  layered defects, found by staged instrumentation of the GSS n=50k MPS run:
  (1) The penalized IRLS was UNGLOBALIZED on both CPU and GPU: on real-survey
  designs whose structurally-zero dummy columns (levels never co-occurring
  with the target's observed rows — GSS has 12 of 23 predictors zeroed for
  some columns) dilute the mean-second-moment ridge, the undamped iteration
  oscillates (measured step growth 13 -> 2e5) and stops at the iteration cap
  wildly non-stationary — measured |beta| ~ 1e5 with penalized |grad| ~ 3e4
  where the true optimum has |beta| ~ 8, ON CPU fp64 TOO (so GSS n=50k CPU
  logreg imputations under all earlier versions drew from non-stationary
  fits; the eta clip kept them finite and directionally sane, but the fits
  were not optima). Both fits now use the polyreg/polr per-chain halving line
  search on the penalized NLL; full steps are accepted whenever they descend,
  so well-identified fits are bit-identical (CPU-GPU fp64 parity unchanged at
  5.6e-17). A chain with no observed-descent step freezes at its last finite
  iterate (the polr semantics) with a dtype-aware acceptance slack — in fp32
  the NLL evaluation noise at survey n (~3e-5) is ~200x the fp64-appropriate
  slack, and judging descent against the tighter slack spuriously starved
  near-converged chains. (2) The GPU posterior-draw Cholesky could fail per
  chain at survey n even with a healthy fit: the fp32 Gram accumulation error
  (~entries*eps ~ 5e-3 at n=50k) exceeds the diluted penalty floor (~0.03),
  making numerical positive-definiteness a per-chain coin flip (measured: one
  chain of 20, every sweep, |beta| ~ 1 fit). The logreg Gram factorizations
  now use the polr `_pd_cholesky` Levenberg-Marquardt escalation: escalate a
  tiny jitter x10 until the factor is finite; well-conditioned chains factor
  first-try bit-identically. Verified end to end: GSS and CSES, n=10k and
  n=50k, m=20/maxit=10, all COMPLETE on MPS (M2 Max: GSS@50k 181 s, CSES@50k
  26 s) and CUDA (RTX 5070 Ti: 73 s / 11 s) with coherent marginals (the
  GSS recode identity holds to 0.0% at the marginal level; CUDA GSS@10k
  marginals now sit closer to observed than any earlier version). New
  deterministic fp64 regression tests pin the oscillation fixture
  (`TestLogregIrlsGlobalisation`: structurally-zero columns + near-duplicate
  quasi-separating dummies; pre-fix |beta| ~ 9e4 non-stationary, post-fix
  |beta| ~ 9 stationary, both backends). Full mice suite: 254 passed / 24
  skipped on MPS, 236 / 42 on CUDA (composite working tree).

- **Fixed the polyreg sibling of the logreg separation defect: `polyreg` (CPU
  and GPU) now carries the family's genuine slope ridge, closing a silent
  column collapse on MPS and a posterior-draw blow-up on CPU/CUDA.**
  `_gpu_polyreg.batched_multinomial_newton` had the same latent pattern as the
  pre-fix logreg (ridge in the Hessian only, unpenalised score, no
  non-finite-step guard): under a COMPLETELY separated unordered target (a
  categorical column that is a deterministic recode of another — the GSS
  structure) the fp32 batched Newton saturated at |beta| ~1e26 FINITE, and the
  posterior draw collapsed every missing cell of every chain onto ONE category
  — silent corruption invisible to the end-of-sweep non-finite guard (measured
  imputed marginal [0, 0, 1] on a 92/6/2 column; the audit's predicted NaN
  refusal was the milder outcome). The CPU side was also defective: it
  delegated to the UNPENALISED `multinomial.multinom`, which under complete
  separation "converges" wherever L-BFGS-B stalls (|coef| ~51) with a
  near-singular information whose inverse blew up the coefficient draw —
  measured CPU imputed marginal [0.18, 0.82, 0.00] on the same 92/6/2 column.
  Calibration decision (the audit §5 open item): penalise BOTH sides with the
  `polr` slope ridge (`1e-5 * mean-slope-column-second-moment * n_obs`, slopes
  only, per-class intercepts unpenalised), in the score and the Hessian, with
  a per-chain/step halving line search (load-bearing: the unpenalised
  intercept block loses curvature under saturation, so the undamped step
  diverges — measured 2e10 in fp64 — exactly the polr issue-#8 overshoot).
  The "port the CPU fallback" alternative was refuted empirically: the CPU
  falls back only on hard fit failures, NOT under separation, so a GPU-side
  marginal fallback would systematically diverge from the CPU reference.
  `methods/polyreg.py` is now self-contained (penalised Newton, like
  `methods/logreg.py`); the visible marginal-draw fallback is retained for
  hard failures, and the multinom over-parameterisation gate
  (`n_obs <= (K-1)*P` -> visible marginal draw) is preserved on CPU and newly
  ported to the GPU (which previously fit over-parameterised sub-problems
  silently). Because the intercepts are unpenalised, the in-sample predicted
  marginal equals the observed marginal exactly (4e-16, even under complete
  separation); on well-identified data the penalised fit matches R
  `nnet::multinom` (reltol 1e-12) to max|diff| 5.4e-5, and the R `mice`
  distributional gates still pass. GPU-vs-CPU fp64 fit parity: 1.1e-16
  well-identified, 1.7e-13 completely separated. Post-fix MPS imputes the
  separated fixture at [0.881, 0.071, 0.048] vs CPU [0.882, 0.079, 0.038].
  Also fixed en route: the fp32 Newton step tolerance is floored at 1e-5 (the
  fp64 1e-8 sits below the fp32 solve noise floor, measured ~2e-8-4e-8 stall
  on MPS, so every fp32 polyreg fit burned the full 100-iteration budget
  without ever reporting convergence — now ~5 iterations). Full mice suite:
  252 passed / 24 skipped (MPS machine), including a new CPU calibration
  characterization module (`test_polyreg_ridge_characterization.py`, the polr
  precedent's structure), fp64 CPU-GPU parity tests, and complete-separation
  regression tests (`TestMpsPolyregCompleteSeparation` in `test_gpu_mps.py`;
  a device-parametrized twin in `test_gpu_glm_separation.py` covers CUDA)
  verified to fail on the pre-fix code. Note for existing results: unordered
  (polyreg) columns under complete separation were mis-imputed by ALL prior
  versions on every backend (CPU included, via the draw blow-up); GSS has no
  unordered columns, so the GSS paper legs are unaffected by this one — CSES
  and any polyreg-bearing runs with deterministic recodes should be
  re-examined.

- **Fixed the fp32 iteration-budget stall in the GPU `logreg` IRLS and `polr`
  Newton (the polyreg tol-floor siblings): both step tolerances are now
  floored at 1e-5 in FP32 (`_IRLS_TOL_FP32` / `_NEWTON_TOL_FP32`), the
  polyreg Newton / multinom solver pattern.** The fp64 step tolerance of
  1e-8 sits below the fp32 solve noise floor, which is data-dependent
  (measured ~3e-8-4e-8 on well-identified n=1,000 fixtures, up to ~4e-5 on
  GSS n=10,000 survey columns), so on fp32-only MPS `batched_logistic_irls`
  (cap 50) and `batched_polr_newton` (cap 100) could never satisfy 1e-8: on
  the real survey configs (paper-harness curation, n=10,000 subsample,
  m=20, maxit=10, seed 20260617, `backend='gpu'`) EVERY logreg and polr call
  of every sweep burned its full iteration budget — GSS 40/40 logreg calls
  (0% of chain-fits ever reported convergence) and 40/40 polr calls, CSES
  40/40 and 20/20 — each wasted polr iteration being a P+1-backward autograd
  Hessian, the dominant GPU-mice cost. Post-fix, combined with the logreg
  globalisation recorded above ("Globalized the logreg fit"), both
  surveys run with ZERO capped calls and 100% chain convergence: GSS logreg
  mean 16.7 iters/call (667 total vs 2,000 pre-fix), polr 29.0 (1,160 vs
  4,000); CSES logreg 8.8 (351 vs 2,000), polr 4.9 (98 vs 2,000). Toggling
  only the floors back to 1e-8 on the current tree re-caps all 80 GSS / 60
  CSES calls, so the floor is the load-bearing piece for the budget waste.
  Wall clock on MPS (M2 Max, torch 2.12.1): full GSS config A/B (warm
  min-of-3, floors toggled) 127.3 -> 34.2 s (3.7x); CSES instrumented single
  runs 61.7 -> 7.1 s; isolated polr fit bench (m=20, n=1,000, q=5, K=5)
  2089 -> 86 ms median (24.2x), with floored-vs-full-budget fit agreement
  <= 1.6e-5 (the stall wanders at that level anyway). Imputed GSS marginals
  verified against a fresh same-tree CPU reference: max gap 4.6% (column 1,
  the near-degenerate column with ~±10% seed-to-seed spread), <= 2.3% on
  every other audit column. The floor is dtype-conditional: the fp64 path is
  bit-identical under any floor value (verified with `torch.equal`), so CUDA
  — whose discrete-GLM fits always run fp64 — provably cannot change; the
  CUDA suite run green on the composite working tree (236 / 42, see the
  globalisation entry above). polr's
  convergence check sits on the damped update and interacts with the
  backtracking line search; separation/draw outcomes verified unchanged
  (full mice suite 254 passed / 24 skipped on the MPS machine, plus
  `test_gpu_polr_separation.py`, `test_gpu_polr_draw.py`,
  `TestMpsLogregCompleteSeparation` and the device-parametrized separation
  twins run explicitly: 30 passed / 1 skipped). Full analysis:
  `docs/GSS_MPS_MICE_AUDIT_2026-08.md` section 6.

- **Audit record: 6.0.0's "successful" GSS-on-MPS imputations were silently
  corrupted; the 6.0.1+ refusal was honest.** Under 6.0.0, the same degenerate
  all-NaN logreg fits were silently cast to level 0 (the minority class) by
  `indices_to_codes` — the returned data had the binary flag imputed at 71.4%
  minority class vs 8.0% observed, and the corruption propagated through the
  chained equations to every other imputed column (e.g. an ordered
  coding-status column imputed at 50.9% for a level observed at 6.3%). Any
  GSS `backend='gpu'` MICE results produced on Apple Silicon under
  pystatistics <= 6.0.0 should be considered invalid. The 6.0.1
  `indices_to_codes` NaN-sentinel propagation (the exact change that flipped
  GSS from "completes" to "refused") remains in place; with the logreg penalty
  above, the sentinel simply no longer fires on GSS.
