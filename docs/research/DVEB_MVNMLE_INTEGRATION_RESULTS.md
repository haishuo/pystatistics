# DVEB forward-Cholesky MVN-MLE integration result

**Outcome: GO for continued research-branch use.**

The explicit `solver='dveb'` route is numerically admitted, works without
PyTorch, is not materially dominated, and was faster than the incumbent
CPU-PyTorch forward-Cholesky route in every one of the 18 prospectively frozen
full-fit lanes. Its geometric-mean full-fit speedup was **2.326x**. This admits
the route for continued work on `research/dveb-pystatistics`; it does not make
DVEB the default, remove PyTorch, authorize a release, or permit a merge to
`main` during the publication freeze.

## Frozen identities

- released base: PyStatistics `v6.1.5`;
- research version: `6.2.0.dev0+dveb`;
- protocol commit: `030bea4`;
- adapter commit: `84a0535`;
- pre-timing qualification commit: `a4a4e13`;
- campaign-harness commit: `f81ddac`;
- v1 invalidation and input-order correction commit: `8cd02ba`;
- byte-identical qualification commit: `73a08f2`;
- corrected v2 campaign-freeze commit: `393087d`;
- DVEB source commit: `a4b0286`;
- native artifact SHA-256:
  `a96e410282fae692f532185ba9c6d0377885f9f7f2b696ff772eda13b1fb102f`.

The six input arrays byte-match DVEB's frozen E1--E6 generators. The immutable
code, protocol, artifact, qualification, environment, thresholds, and input
hashes are in `evidence/dveb_mvnmle/campaign-freeze.json`. No v2 full-fit
timing result existed when that freeze was committed. The excluded v1 evidence
is retained separately and contributes no value to this result.

## Correctness admission

Before timing, all 18 E1--E6 by thread-count lanes passed one untimed end-to-end
fit with both implementations. During the campaign, all **540 paired
observations (1,080 fits)** converged and passed the frozen comparison:

- maximum absolute log-likelihood difference: `1.834e-9` (bound `1e-5`);
- maximum mean-vector element difference: `2.602e-11`;
- maximum covariance-matrix element difference: `2.872e-11`;
- mean and covariance comparisons: `rtol=atol=1e-3`;
- optimizer iteration and evaluation counts: equal in every lane.

The complete MVN-MLE suite had already passed at `a4a4e13` (272 passed, 2
skipped). The whole default suite had 4,475 passed, 94 skipped, and the same one
unrelated multinomial failure reproduced at exact pre-integration commit
`a9c7e4d`; therefore the frozen no-new-failure gate passed. Two initial
fit-qualification attempts stopped before six-thread numerical work because an
OpenMP binding hint narrowed the main thread's reported affinity. Both attempts
are preserved. Removing those misleading hints while retaining explicit
`taskset` confinement and fixed thread counts produced the admitted 18/18 run;
this correction occurred before campaign timing.

The first complete campaign was also invalidated after its verifier found that
PyTorch thread initialization before input generation changed six E2-T12 input
values by one ULP. Both solvers had received the same array and all fits passed,
but its bytes did not match the freeze. Commit `8cd02ba` preserves and excludes
that evidence. V2 generates and hashes every input before PyTorch initialization
and verifies it again after all fits; all 18 lanes match the frozen hashes and
remain unchanged. Thresholds, cases, repetitions, solvers, and decision rules
were not changed.

## Full user-visible fit result

Each lane used three untimed warmups and 30 paired, randomized observations in
a fresh lane process. Times below are medians of the complete `mlest` call in
milliseconds. The interval is the frozen 10,000-resample paired-bootstrap 95%
interval for `PyTorch/DVEB` and is descriptive; the median rule decides.

| Case | Threads | PyTorch ms | DVEB ms | PyTorch/DVEB | Bootstrap 95% | DVEB schedule |
|---|---:|---:|---:|---:|---:|---|
| E1 | 1 | 4.181 | 1.139 | **3.671x** | 3.630--3.707 | serial |
| E1 | 6 | 4.360 | 1.233 | **3.535x** | 3.499--3.551 | parallel |
| E1 | 12 | 5.861 | 1.719 | **3.409x** | 3.385--3.428 | serial |
| E2 | 1 | 23.596 | 13.719 | **1.720x** | 1.715--1.724 | serial |
| E2 | 6 | 20.808 | 6.881 | **3.024x** | 2.953--3.036 | parallel |
| E2 | 12 | 29.253 | 9.134 | **3.203x** | 3.197--3.215 | parallel |
| E3 | 1 | 930.512 | 562.828 | **1.653x** | 1.624--1.676 | serial |
| E3 | 6 | 385.967 | 142.896 | **2.701x** | 2.615--2.738 | parallel |
| E3 | 12 | 481.527 | 138.256 | **3.483x** | 3.443--3.545 | parallel |
| E4 | 1 | 13.742 | 8.869 | **1.549x** | 1.542--1.559 | serial |
| E4 | 6 | 13.328 | 7.543 | **1.767x** | 1.759--1.771 | parallel |
| E4 | 12 | 18.916 | 10.999 | **1.720x** | 1.716--1.725 | parallel |
| E5 | 1 | 198.002 | 189.730 | **1.044x** | 1.038--1.053 | serial |
| E5 | 6 | 94.564 | 73.838 | **1.281x** | 1.270--1.292 | parallel |
| E5 | 12 | 128.819 | 93.439 | **1.379x** | 1.362--1.393 | parallel |
| E6 | 1 | 4,209.948 | 2,125.861 | **1.980x** | 1.920--2.077 | serial |
| E6 | 6 | 1,904.158 | 515.432 | **3.694x** | 3.600--3.788 | parallel |
| E6 | 12 | 2,346.511 | 430.221 | **5.454x** | 5.344--5.612 | parallel |

DVEB was faster in **18/18** lanes and met the preferred, nonmandatory 1.5x
threshold in **15/18**. The three sub-1.5x lanes were all E5; none was a loss.
No lane met the catastrophic `PyTorch/DVEB <= 0.10` rule. PyTorch was faster in
zero lanes, so systematic domination is false. The geometric mean of
`DVEB/PyTorch` was `0.4300`, hence **GO** under the committed rule.

## Where the gain occurs

The timer sections show that objective construction is broadly comparable and
the material gain is in optimization, where the fused native value-and-gradient
body executes repeatedly. For example:

| Lane | PyTorch setup ms | DVEB setup ms | PyTorch optimization ms | DVEB optimization ms |
|---|---:|---:|---:|---:|
| E1-T1 | 0.348 | 0.467 | 3.478 | 0.476 |
| E3-T6 | 11.979 | 13.533 | 365.317 | 125.968 |
| E5-T1 | 22.386 | 22.683 | 173.450 | 164.463 |
| E6-T12 | 91.596 | 111.436 | 2,164.045 | 289.950 |

The resident fused-call probe agrees with that attribution. DVEB was faster in
18/18 resident lanes, from 1.83x to 15.42x. These probes are operational
explanation, not a second decision endpoint.

Automatic scheduling selected serial execution for all one-thread lanes,
parallel execution for every six-thread lane, and parallel execution for all
12-thread lanes except the tiny E1 case, where it selected serial. This
consumer campaign did not race forced schedules and therefore does not claim a
new selector-regret result; it consumed the already-qualified Trunk 008 policy.
Scratch storage ranged from 165,056 bytes (E1-T1) to 2,090,544 bytes at 12
threads for the p=30 cases.

## Deployment and operational result

The copied native library is **38,768 bytes**. Its dynamic dependencies are
only `libstdc++`, `libm`, `libgomp`, `libgcc_s`, `libc`, and the system loader;
there is no PyTorch, CUDA, or NVIDIA dependency and no RPATH/RUNPATH.

Across 30 fresh processes:

- research-package import plus DVEB objective construction and first answer:
  median `232.078 ms`, p05 `231.171 ms`, p95 `234.513 ms`;
- research-package import through a complete E1 DVEB fit with importing
  `torch` forcibly rejected: median `232.329 ms`, p05 `230.935 ms`, p95
  `234.516 ms`;
- all 30 torch-blocked fits converged, reported the DVEB backend, and left
  `torch` absent from `sys.modules`.

Those startup numbers include importing the research package and are not a
claim that dynamic loading alone takes 232 ms. Campaign peak RSS was recorded
at the shared worker-process level after both implementations had run, so it
must not be attributed to either implementation individually. A separately
qualified release wheel and isolated per-implementation memory study remain
future work.

## Interpretation and next step

This is the first PyStatistics consumer result for DVEB, not a general PyTorch
replacement result. It establishes a genuine use case: the same
forward-Cholesky estimator and SciPy optimizer can use a compact CPU-only
native numerical body, remain fully torch-free when explicitly selected, and
run materially faster on this admitted grid. PyTorch remains the default and
continues to cover workloads and devices outside this artifact's contract.

The next justified work is packaging research, not another performance trunk:
define and qualify a genuinely CPU-only Linux artifact/wheel boundary, decide
the supported x86-64 baseline, and test installation on disjoint systems. That
work must remain on the research branch and cannot alter `v6.1.5`, PyPI, the
paper replication material, or `main` until the publication freeze is lifted.

## Evidence

- frozen protocol: `docs/research/DVEB_MVNMLE_INTEGRATION_PROTOCOL.md`;
- pre-timing qualification:
  `docs/research/DVEB_MVNMLE_PRETIMING_QUALIFICATION.md`;
- campaign freeze and fit qualification:
  `docs/research/evidence/dveb_mvnmle/`;
- raw 540-pair observations: `campaign-raw.json`;
- mechanical analysis: `campaign-analysis.json`;
- operational probes: `operational.json`;
- verifier: `benchmarks/dveb_mvnmle_integration/verify_evidence.py`.

No result-driven solver, artifact, optimizer, grid, threshold, or schedule
change was made. Nothing in this result authorizes merge, release, or default
routing.
