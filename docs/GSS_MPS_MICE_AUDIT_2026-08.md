# GSS-on-MPS MICE refusal & CPU polr cost — audit and fix (2026-08-12)

Investigation of two facts established in `docs/ACCELERATION_AUDIT_2026-08.md`
and the 6.1.1 release notes:

1. Mixed-type MICE on the GSS survey (n=10,000 subsample, seed 20260617, m=20,
   maxit=10, `backend='gpu'` on Apple MPS/fp32) **completes** under 3.16.3 and
   6.0.0 but is **refused** by the end-of-sweep non-finite guard under 6.0.1
   through 6.1.1.
2. The GSS CPU legs were reported as ground down by repeated
   `polr fit failed (ConvergenceError); using a marginal draw` warnings,
   with GSS n=50k CPU at ~66 min/repeat.

All experiments below ran on the Mac Studio (M2 Max), torch 2.13.0, with the
paper harness's exact GSS subsample and seed. Timing numbers were collected
while a separate 6.1.1 validation grid was running on the same machine and are
therefore mildly inflated; behavioural results (failure counts, marginals,
completion/refusal) are unaffected.

## 1. Which 6.0.1 fix flips the behaviour

**The `indices_to_codes` NaN-sentinel propagation** (`mice/backends/
_gpu_encode.py`, commit b1a57dd). Before 6.0.1, a non-finite class index —
the degenerate-fit sentinel the GPU categorical samplers emit — was cast
straight to `long`, which turns NaN into index 0 and silently writes
`levels[0]` into every affected cell. Since 6.0.1 the NaN survives to the
backend's end-of-sweep guard, which refuses the run.

None of the other eleven 6.0.1 fixes participates: the sweep-visible GPU
changes are this one line plus a docstring, and the run under 6.0.0
(site-packages) reproduces the **identical** degeneracy map as 6.1.1 — same
sweeps, same chains, same cell counts — differing only in what happens to the
sentinel (silent collapse vs refusal).

### Where the sentinel comes from

Instrumenting every GPU sweep step (observer on `GPU_METHODS`, no RNG
perturbation) at the failing configuration:

- **First non-finite emission: iteration 1, column 0, method `logreg`** —
  9 of 20 chains, all 2,470 missing cells of the column at once.
- Eight further emissions, all column 0, iterations 2–9, 1–9 chains each;
  ~30% of all (chain, sweep) logreg fits of column 0 degenerate. Chains are
  "repaired" the next sweep when their fit happens to succeed (the sweep
  rewrites the whole column), so the run only dies because late-sweep
  emissions (9 of 20 chains at iteration 9) reach the final data.

**Why column 0, and why GSS:** column 0 ("case was retrieved", binary,
observed 604:6926) shares its 2,470-row missingness block with the
occupation-coding-status columns 3 and 5, and is a **deterministic recode** of
them (col 3 > 0 ⟺ col 0 = 0; the marginals match exactly: 473+131 = 604).
Every chained-equations fit of column 0 on the dummy-encoded columns 3/5 is
therefore **completely separated by construction** — not an artifact of a
sweep state, but a property of the data.

**Why the fit dies on MPS:** `_gpu_logreg.batched_logistic_irls` still carried
the **pre-6.0.1** logistic fit: ridge in the Gram only (a Hessian damper),
gradient unpenalised — so at convergence it solves the *unpenalised* score
equation, which has no finite solution under complete separation. The 6.0.1
CPU fix (`methods/logreg.py`: genuine penalty in the gradient, toward the
marginal-rate intercept) **was never ported to the GPU file**. On CUDA the
discrete-GLM fits run in fp64 (`discrete_glm_compute_dtype`) and the diverging
iterates limp along finitely; MPS has no fp64, and in fp32 the fit reaches
non-finite values. The IRLS loop also had no non-finite-step guard (the polr
Newton got one; logreg didn't), so one bad step permanently poisoned the whole
chain's fit for that sweep step.

## 2. Honest or false refusal? (CONVENTIONS.md A6)

**Honest.** Auditing 6.0.0's "successful" run (site-packages 6.0.0, version
asserted at runtime, same seed/config, observer on its `indices_to_codes`):

- 9 silent-collapse events, 103,740 cells total, all cast to `levels[0]` = the
  **minority** class (8% of observed data).
- 22,230 of those cells (9 of 20 chains × 2,470) were collapsed **in the final
  iteration** and are present in the returned datasets.
- Returned-data marginals vs observed:

  | column | observed | 6.0.0 imputed |
  |---|---|---|
  | 0 binary, level 0 | 8.0% | **71.4%** |
  | 1 binary, level 0 | 5.5% | **36.6%** |
  | 3 ordered, level 1 | 6.3% | **50.9%** |
  | 2 ordered, level 2 | 86.4% | **29.7%** |
  | 6 binary, level 0 | 49.0% | **11.0%** |

  The corruption propagates through the chained equations to every column,
  including the numeric ones (col 11 imputed mean −0.209 vs 0.020 observed).

Any GSS `backend='gpu'` MICE result produced on Apple Silicon under
pystatistics ≤ 6.0.0 is invalid. This was visible in the paper-era fidelity
metrics: `results/categorical_survey_gss_mps.json` (June 2026) records
tv_distance 0.44–0.48 against R for exactly the binary columns — numbers now
explained.

So per A6 the 6.0.1+ refusal was the *correct guard* — but it was also a
symptom of a fixable defect: the request **can** be honored on MPS once the
fit carries the same penalty the CPU reference has. Leaving the refusal in
place with the fit unfixed would keep refusing a computable request (the A6
obligation-2 concern); fixing the fit resolves both.

## 3. The fix

`mice/backends/_gpu_logreg.py` now mirrors `methods/logreg._fit_logistic`
exactly (6.0.1 semantics), batched per chain:

- slope ridge `1e-5 · mean-column-second-moment · n` **in the gradient and the
  Gram**, centred on `beta0 = [logit(marginal rate), 0, …]`;
- intercept anchored toward the marginal logit at strength `1e-5 · n`;
- IRLS starts at `beta0`;
- posterior draw covariance `(X'WX + R)^{-1}`, as on CPU;
- non-finite Newton-step guard mirroring the polr Newton: a transiently
  non-finite fp32 solve zeroes that step instead of poisoning the chain.

Verification:

- fp64 parity with the CPU fit: max coefficient difference **5.6e-17**
  (well-identified) and **3.6e-14** (completely separated); covariance
  1.9e-10.
- Full GSS (n=10k, m=20, maxit=10, seed 20260617) on MPS: **completes** (~2.3
  min, contended machine; the pre-fix run refused after ~2 min). Imputed
  marginals now track the CPU reference (§3.1).
- `tests/mice/`: 240 passed, 24 skipped (CPU + MPS), including the existing R
  fixture and CPU-parity gates and three new complete-separation regression
  tests that fail on the pre-fix code (`TestMpsLogregCompleteSeparation` in
  `test_gpu_mps.py`, plus a device-parametrized twin in
  `test_gpu_glm_separation.py` that gives CUDA the same coverage — the
  existing fixture there is only *quasi*-separated, too mild to trip fp32).
  The twin's recode column is **ordered** (polr — penalised and
  line-search-globalised on GPU), deliberately not an unordered polyreg
  column: coupling to polyreg confounds the regression with polyreg's own
  open separation-calibration item (§5), which on CUDA distorts the joint
  badly enough to fail the assertion for reasons unrelated to logreg.
- CSES (the other real survey) full config re-verified completing on MPS
  post-fix (60.6 s at n=10k; the previously-refused n=50k cell in §3.3).

### 3.1 CUDA re-verification (Forge, RTX 5070 Ti, torch 2.13.0+cu130)

Run 2026-08-12 in a scratch worktree at v6.1.1 + this patch (worktree removed
after; prebuilt 6.1.1 Cython kernels reused; GPU idle before and after).

**New finding: the pre-fix CUDA path was not benign.** fp64 keeps the
unpenalised separated fit finite, but *not stationary* — an isolated separated
fit stalls at the iteration cap far from the penalised optimum (measured
|penalised grad| ≈ 2.9 at a coefficient of 575 vs the true optimum at 35), and
the posterior draw covariance is garbage in the saturated directions. On real
GSS this **silently distorted** the imputations rather than refusing: pre-fix
CUDA imputed the col-0 flag at **44.6%** minority vs 8.0% observed (CPU
reference: 17.1%), col 2 level-2 at 44.9% vs the CPU's 72.7%. The June 2026
paper results carry the same signature on the CUDA legs
(`categorical_survey_gss_cuda.json`: tv_distance up to 0.41; the v6.1.0 rerun
still 0.13–0.17 on the recode columns) — so the paper's GSS `backend='gpu'`
rows are invalid on **both** devices, not only MPS.

Post-fix on CUDA:

- Full `tests/mice/` suite: **224 passed, 40 skipped, 0 failed** (skips are
  MPS-only and fixture-gated tests). The new complete-separation test passes;
  run against the unpatched 6.1.1 site-packages install it **fails** on CUDA
  exactly as it does on MPS — the regression discriminates on both devices.
- Full GSS config (n=10k, m=20, maxit=10, seed 20260617), `backend='gpu'`:
  completes in 21.3 s with marginals tracking the CPU reference — col 0:
  15.8% (CPU 17.1%, pre-fix CUDA 44.6%); col 2 lvl 2: 73.8% (CPU 72.7%,
  pre-fix 44.9%); col 3: 84.8% (CPU 82.3%, pre-fix 67.2%); cols 1/7 within
  the seed-to-seed spread established in §3.2.

### 3.2 Post-fix GSS marginals, GPU vs CPU

Same configuration, both backends, imputed-cell marginals pooled over m=20
(observed% for reference):

| col (kind, level) | observed | CPU imputed | GPU imputed (fixed) |
|---|---|---|---|
| 0 (binary, 0)  | 8.0%  | 17.1% | 18.0% |
| 1 (binary, 0)  | 5.5%  | 43.5% | 33.9% |
| 2 (ordered, 2) | 86.4% | 72.7% | 71.2% |
| 3 (ordered, 0) | 92.0% | 82.3% | 80.1% |
| 4 (ordered, 2) | 86.3% | 70.5% | 70.8% |
| 5 (ordered, 0) | 92.0% | 81.8% | 81.8% |
| 6 (binary, 0)  | 49.0% | 50.2% | 53.8% |
| 7 (binary, 0)  | 17.2% | 36.1% | 35.6% |
| 9 (numeric mean) | 0.020 | −0.123 | −0.120 |
| 11 (numeric mean) | 0.020 | −0.195 | −0.189 |

GPU tracks CPU within ≤3.6% on every column except column 1 (9.6% gap, which
is pure chain noise: re-running the GPU at seeds 20260618/20260619 gives
pooled 43.5%/33.2% — the CPU's own 43.5% is inside the GPU's seed-to-seed
range, and per-chain values span 0.00–0.94 on this near-degenerate column,
so a 20-chain pooled proportion has ~±0.10 spread). The imputed marginals
legitimately deviate from the *observed* marginal on these columns: the
missing block is the "item not asked" subsample, and the deterministically
linked columns are imputed jointly — the CPU reference deviates identically.
The A6-relevant comparison is GPU-vs-CPU, not GPU-vs-observed.

### 3.3 CSES at n=50,000 — same defect, not survey-specific

The 6.1.1 benchmark grid surfaced (cross-session, 2026-08-12) that CSES at
n=50,000 on MPS flips at exactly the same version boundary as GSS: completes
under 6.0.0, refused under 6.0.1+ — while CSES at n=10,000 never fails.
Instrumenting the unpatched 6.1.1 run at that cell (same observer, m=20,
maxit=10, seed 20260617):

- Origin: **logreg again** — columns 8 and 9 ("EXPERT: IDEOLOGY" binary
  flags, observed minority 2.9% and 8.1%, 14,311 missing cells each). Both
  degenerate in **all 20 chains from iteration 0**, every sweep (40 of 40
  fits) — more severe than GSS's ~30%. The curated CSES problem has no
  unordered columns, so the polyreg sibling is not involved; the
  size-sensitivity is the fp32 divergence threshold of the unpenalised IRLS
  (the same columns' fits survive at n=10k).
- Under 6.0.0, those cells were silently cast to the ~3% minority class in
  every chain of every sweep — 6.0.0's CSES@50k "completions" (the grid's
  195 s baseline) are corrupted the same way as GSS's.
- **The logreg penalty alone resolves it**: an isolated worktree at v6.1.1 +
  only the `_gpu_logreg.py` penalty patch completes the cell in 255.9 s
  (unpatched: refused after 238.2 s); the full working tree (with the polyreg
  fix and the fp32 tol floor also present) completes it as well (297.6 s).

### 3.4 GSS at n=50,000 — two more layers under the same rock

The penalty alone did **not** clear GSS@50k on MPS (refused after ~580 s).
Staged instrumentation (per-stage finiteness inside the logreg impute,
identical RNG order) plus tensor capture of the failing chain isolated two
further defects, both fixed:

1. **The penalized IRLS was unglobalized — on CPU too.** GSS columns carry
   12-of-23 structurally-zero predictors (dummy levels that never co-occur
   with the target's observed rows), which dilutes the mean-second-moment
   ridge; the undamped iteration then oscillates (measured step growth
   13 → 2e5) and stops at the cap wildly non-stationary. On the captured
   chain-19 tensors, CPU fp64 `_fit_logistic` stopped at |beta| = 101,632
   with penalized |grad| = 14,190; the true optimum has |beta| = 7.85. In
   fp32 a transient overshoot reaches inf → frozen non-finite iterate →
   all-NaN draw → refusal. Both fits now carry the polyreg/polr per-chain
   halving line search on the penalized NLL (full steps accepted whenever
   they descend → well-identified fits bit-identical, parity 5.6e-17). Two
   calibration subtleties mattered: no-descent chains must **freeze at the
   last finite iterate** (polr semantics), not be poisoned — in fp32 the NLL
   evaluation noise at survey n (~3e-5) is ~200× the fp64-appropriate slack,
   and poisoning on noise re-refused the run; the acceptance slack is now
   dtype-aware (100·eps).
2. **The fp32 Gram Cholesky was a per-chain coin flip at survey n.** With the
   fit healthy (|beta| ~ 1), `cholesky_ex` still failed for one chain of 20,
   every sweep: the fp32 Gram accumulation error (~entries·eps ≈ 5e-3 at
   n≈44k) exceeds the diluted penalty floor (~0.03). The logreg Gram
   factorizations now reuse polr's `_pd_cholesky` Levenberg-Marquardt
   escalation (jitter ×10 until finite; first-try chains bit-identical).

End-to-end after both: GSS and CSES complete at n=10k and n=50k on **both**
devices — MPS: GSS@50k 181 s, CSES@50k 26 s, GSS@10k 44 s, CSES@10k 6 s
(the fp32 tolerance floors from the concurrent session cut sweep times
~2.5–10×); CUDA: 73 s / 11 s / 19 s. GSS@50k marginals are coherent (the
recode identity holds exactly at the marginal level: imputed col 0 level-1 =
imputed col 3/5 level-0 = 97.1%), and CUDA GSS@10k marginals sit closer to
observed than any earlier version (col 0: 9.5% vs 8.0% observed). A
deterministic fp64 regression fixture pins the oscillation
(`TestLogregIrlsGlobalisation` in `test_gpu_glm_separation.py`; pre-fix
|beta| ~ 9e4 non-stationary, post-fix ~9 stationary, CPU and GPU).

**Disclosure for existing results:** GSS n=50k **CPU** logreg imputations
under all earlier versions drew from non-stationary fits (finite and
directionally sane via the eta clip, but not optima, with a mis-scaled draw
covariance). The GSS@50k CPU paper legs should be re-examined alongside the
GPU ones.

**Residual siblings flagged, not changed here:** `_gpu_polyreg`'s new line
search poisons on no-descent with a fixed 1e-8 slack — the same fp32
large-n noise hazard this session hit in logreg (its fixtures are n=4000;
no failing case in hand). The shared `_cholesky_ridged` used by the
numeric-column draw (`_gpu_linreg`) keeps its fixed jitter; PMM's donor-copy
output stays finite regardless, but a failed factor there scrambles that
chain-step's donors silently — same fp32-at-scale mechanism, bounded impact.

## 4. The CPU polr "grind"

**Not reproduced under 6.1.1.** Instrumenting every `fit_polr` call (timing,
iterations, exception reason) on the canonical GSS subsample and seed:

- n=10,000, m=20, maxit=10 (the full bench config): **800/800 fits converged,
  zero ConvergenceErrors, zero warnings.** 409 s total, 97.6% of it inside
  polr fits (~500 ms each).
- n=50,000, m=5, maxit=3: **60/60 converged, zero warnings**, ~2.85 s/fit.

The cost story needs no failures: at n=50k the full config implies 800 polr
fits × ~2.9 s ≈ 38 min in polr alone, plus 800 logreg/pmm steps — consistent
with ~53–66 min/repeat. The June 2026 baseline
(`categorical_survey_gss_mps.json`) already recorded CPU median **3166 s
(~53 min)** at n=50k — today's cost is in line with history, not 5× it. For
scale: R mice 3.19.0 took 1554 s on the same problem (pystatistics CPU has
always been ~0.4–0.5× R on GSS; that is the pre-existing polr cost profile,
not a regression).

No surviving log from the 6.1.x benchmarking sessions contains the
"polr fit failed" warning, and the mice CPU path is unchanged since 6.0.1
(`git diff b1a57dd..HEAD -- pystatistics/mice` touches only GPU files), so a
CPU behavioural regression between 6.0.x and 6.1.x is not possible from the
code history. Residual uncertainty: the full n=50k config (800 fits) was not
probed end-to-end (the machine was occupied by the 6.1.1 validation grid);
if the warnings reappear there, each failed fit costs L-BFGS-B's full budget
before falling back, and the place to look is the polish-stage
`ConvergenceError` reason (`non-PD observed information` vs `score test
failed`, `ordinal/_solver.py::_polish_to_mle`), captured by wrapping
`pystatistics.ordinal.polr` with a timing/exception observer as done here.

On the statistical question as posed: at the observed failure rate (zero),
the marginal-draw fallback contributes nothing on GSS under 6.1.1. Had it
fired at the reported per-sweep-step frequency, it would be a real concern —
a marginal draw is predictor-blind, and a high fallback rate on a column
means its imputations ignore the conditional structure — but that scenario is
not the current behaviour.

## 5. Siblings and follow-ups

- **GPU `polyreg` carries the same latent pattern** (Hessian-only ridge,
  unpenalised score, no non-finite-step guard, no fallback). It is not
  exercised by GSS (no unordered columns) and CSES completes, but a
  completely separated unordered target on MPS would hit the same
  refusal-instead-of-fit. The calibration decision differs from logreg:
  the CPU `polyreg` is an *unpenalised* multinomial fit with a visible
  marginal-draw fallback, so a GPU penalty would diverge from the CPU
  reference. Making the pair consistent (penalise both? port the fallback?)
  is a design decision needing its own R-oracle validation — deliberately
  not folded into this fix.

  **RESOLVED (same day, follow-up session).** Empirics overturned two of the
  premises above. (1) The MPS failure is NOT a refusal: the fp32 Newton
  saturates at |beta| ~1e26 *finite* and the draw silently collapses every
  chain onto one category (imputed marginal [0, 0, 1] on a 92/6/2 recode
  fixture) — invisible to the non-finite guard, i.e. the 6.0.0 failure class
  by another route. (2) The CPU fallback does NOT fire under separation:
  the unpenalised `multinom` "converges" where L-BFGS-B stalls (|coef| ~51)
  and the near-singular-information draw relocates the imputed marginal
  ([0.18, 0.82, 0.00] on the same fixture) — so porting the fallback would
  have diverged from the CPU reference, and the CPU itself needed the fix.
  Calibration chosen: the polr slope ridge on BOTH sides (score + Hessian,
  slopes only, intercepts unpenalised, per-chain damped Newton — the
  undamped step diverges under saturation, 2e10 in fp64). R-oracle:
  penalised fit matches `nnet::multinom` to 5.4e-5 on well-identified data;
  in-sample marginal identity exact (4e-16) under separation; R `mice`
  distributional gates unchanged. fp64 CPU-GPU parity 1.7e-13 separated.
  See the 6.1.x UNRELEASED entry and
  `tests/mice/test_polyreg_ridge_characterization.py`.
- **GPU `polyreg` on CUDA is confirmed distorting, not just latent:** in the
  coupled synthetic (binary recode of an unordered categorical, shared
  missing block) the fp64 polyreg fit drives per-chain level proportions to
  0.02–0.85 against a 0.92 truth and breaks the deterministic recode relation
  (down to P = 0.08 in some chains). That measurement is direct evidence for
  the follow-up task's priority.
- **GPU `polr`** already mirrors the CPU objective ridge and has the step
  guard; no gap.
- **CUDA re-verification: done** (§3.2) — and it upgraded the finding: the
  pre-fix CUDA path silently distorted GSS rather than refusing.
- **Paper legs:** the GSS `backend='gpu'` rows in the June 2026 paper results
  are invalid on **both** devices — MPS under ≤ 6.0.0 collapsed outright
  (tv_distance 0.44–0.48 was the corruption), and the CUDA legs carry the
  milder fp64 distortion (tv up to 0.41; still 0.13–0.17 in the v6.1.0
  rerun). Re-run both after this fix lands.

## 6. The fp32 tolerance stall — logreg IRLS and polr Newton (follow-up, 2026-08-12)

The polyreg fix found en route that the fp64 step tolerance 1e-8 sits below
the fp32 solve noise floor on MPS, so every fp32 polyreg fit burned its full
Newton budget without ever reporting convergence; the fix floored polyreg's
tolerance at 1e-5 in fp32. This section closes the two predicted siblings:
`_gpu_logreg.batched_logistic_irls` (tol 1e-8, cap 50) and
`_gpu_polr.batched_polr_newton` (tol 1e-8, cap 100).

**Measured — both stall.** Instrumented replicas of the exact loop bodies
(per-iteration max|step| among active chains), m=20 / n=1,000 / q=5
well-identified fixtures, MPS fp32 vs CPU fp64 (torch 2.12.1, M2 Max):

| loop | fp64 tol 1e-8 | fp32 tol 1e-8 (pre-fix) | fp32 floor 1e-5 (post-fix) |
|---|---|---|---|
| logreg IRLS | 5 iters, 20/20 conv | **50/50 iters, 2/20 conv**, stall ~3–4e-8 | 4 iters, 20/20 |
| polr Newton | 5 iters, 20/20 conv | **100/100 iters, 6/20 conv**, stall ~1e-7–1.5e-6 | 5 iters, 20/20 |

On the real survey configs (paper harness curation, n=10,000 subsample,
m=20, maxit=10, seed 20260617, `backend='gpu'`), pre-fix **every** logreg and
polr call in the whole run hit its iteration cap — GSS: 40/40 logreg calls
(0% of chain-fits ever converged), 40/40 polr calls (37% converged); CSES:
40/40 and 20/20. Each wasted polr iteration is a P+1-backward autograd
Hessian — the dominant cost of the heaviest GPU method.

**Fix.** The polyreg pattern, ported: step tolerance floored at 1e-5 when the
compute dtype is not float64 (`_IRLS_TOL_FP32` / `_NEWTON_TOL_FP32`). The
fp64 path resolves to the same 1e-8 tolerance as before — verified
bit-identical (`torch.equal` on all outputs) with the floor constant set to
an absurd value, so CUDA (whose discrete-GLM fits always run fp64 via
`discrete_glm_compute_dtype`) provably cannot change. polr's convergence
check sits on the damped update and interacts with the line search; the polr
separation and natural-draw test outcomes were verified unchanged under the
floor.

**Post-fix on the surveys — floor alone (pre-line-search loop).** polr is
fully cured: zero capped calls, 100% of chains converge (~29 iters/call on
GSS — real data is harder than the synthetic fixture — ~5 on CSES). logreg
dropped from 2,000 to 1,278 total iterations on GSS (711 on CSES) with two
residual cap populations, both understood: (a) the completely separated
recode columns oscillated at large steps (~1e5) that no tolerance can catch
— undamped IRLS wandering in the penalised separated region, finite via the
penalty + eta clip; (b) the fp32 noise floor is data-dependent and on one
GSS binary column (n_obs=7,530) sat at ~7e-6–4e-5, straddling the 1e-5
floor, so 6/10 of its calls still capped on a straggler chain. The floor is
deliberately kept at the family-standard 1e-5 (polyreg/multinom precedent)
rather than tuned upward per-column.

**Post-fix on the surveys — current tree (floor + IRLS line search).** The
logreg oscillation population (a) was independently root-caused
(cross-session, same day) as the GSS n=50k refusal mechanism and closed by
adding the polyreg/polr halving line search to the logreg IRLS (CPU and GPU
— documented with that change); the convergence check now sits on the DAMPED
update, which also dissolves population (b). Re-traced on the current tree:
**zero capped calls anywhere and 100% chain convergence on both surveys** —
GSS logreg mean 16.7 iters/call (667 total, was 2,000), polr 29.0 (1,160,
was 4,000); CSES logreg 8.8 (351, was 2,000), polr 4.9 (98, was 2,000).
With the floors dropped back to 1e-8 on this same tree, every one of the 80
GSS / 60 CSES calls still hits its cap — the line search alone does not
remove the tolerance stall (11% of GSS logreg chain-fits converge at 1e-8),
so the floor stays load-bearing for the budget waste and the line search
for the separated-fit dynamics.

**Wall clock.** Isolated fit benches (MPS fp32, warm, median of 5;
pre-line-search loop for logreg): logreg 263→22 ms (12.2x), polr 2089→86 ms
(24.2x — polr's loop is untouched by the globalisation work);
floored-vs-full-budget fit agreement max|diff| ≤1.6e-5 (the stall wanders at
that level anyway). Full GSS config A/B on the current tree (warm min-of-3,
floors toggled, line search in both arms): **127.3 → 34.2 s (3.7x)**; CSES
instrumented single runs 61.7 → 7.1 s.

**Marginals on the current tree.** The §3.2 CPU reference predates the CPU
logreg line search, and the globalisation moves BOTH backends identically on
the separated columns (they now converge to the penalised optimum instead of
stopping at the cap on a non-stationary iterate — e.g. col 0: CPU 17.1% →
9.5%, GPU 18.0% → 9.2%; col 2: CPU 72.7% → 82.7%, GPU 71.2% → 81.4%). GPU vs
a fresh same-tree CPU reference (390 s run, same config): max gap 4.6% on
col 1 (the near-degenerate column with ~±10% seed-to-seed spread), ≤2.3% on
every other audit column; numerics col 9 −0.129 vs −0.134, col 11 −0.204 vs
−0.211. Reproduction note: the CPU reference must run in a **torch-free
process** — with torch imported alongside miniconda MKL, the long CPU run
segfaults mid-sweep (the duplicate-OpenMP-runtime hazard that
`KMP_DUPLICATE_LIB_OK=TRUE` downgrades from an abort to "may crash").

Full mice suite on the composite tree: 254 passed / 24 skipped (MPS machine),
with the polr separation/draw gates and the logreg complete-separation gates
also run explicitly (30 passed / 1 skipped) — the polr damped-update
convergence check and its line-search interaction are unchanged by the floor.
