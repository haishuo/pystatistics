# DVEB portable-wheel MVN-MLE performance result

**Outcome: STRONG GO.**

The exact installed CPU-only CPython 3.11 wheel passed every correctness gate,
was faster than the optimized CPU-PyTorch implementation in all 18 frozen
full-fit lanes, and achieved a **2.975x geometric-mean advantage**. It met the
preferred 1.5x per-lane threshold in 17/18 lanes. The remaining lane, E5/T1,
was a near tie that still favored portable DVEB by 1.045x.

This establishes more than the minimum “must not be dominated” requirement
for this artifact and workload. It does not make DVEB a general PyTorch
replacement or authorize a release/default-route change.

## Protocol integrity

Portable campaign v1 stopped before timing because NumPy 2.2.6 regenerated
different array bytes from the older NumPy 2.3.5 campaign. Its 18 untimed
numerical passes are retained but excluded. V2 committed the six evaluation
arrays, froze both file and raw-array hashes, and changed no grid, solver,
environment, tolerance, repeat, or decision rule. V2 was committed at
`4d79343` before fresh qualification or timing.

The timed artifact is the exact wheel payload admitted by CPU-wheel protocol
v2:

- wheel SHA-256: `92fa398c...44caa4a`;
- native payload SHA-256: `65438a0d...229fe38`;
- dense CPU ABI v1, x86-64-v2, IEEE float64;
- contraction disabled, no fast-math;
- wheel-local `libgomp`, no CUDA/PyTorch/NVIDIA native dependency.

Workers ran from `/tmp`, required imports to resolve below the installed wheel
site, and rejected artifact or ABI drift. No build, download, fallback,
calibration, or device work occurred during qualification or timing.

## Correctness

Fresh untimed admission passed 18/18 lanes. All **540 timed pairs (1,080
fits)** converged and passed the frozen comparison. Across all qualification
and timed observations, the largest solver differences were:

- log-likelihood: `4.453e-9` (bound `1e-5`);
- mean element: `1.347e-11`;
- covariance element: `2.656e-11`;
- mean/covariance acceptance: `rtol=atol=1e-3`.

Every lane consumed its committed fixture bytes, the arrays remained
unchanged, and optimizer iteration/evaluation behavior was recorded rather
than normalized away.

## Full user-visible fit result

Each lane used three untimed warmups followed by 30 paired, randomized complete
`mlest` calls. Times are medians in milliseconds; intervals are the frozen
10,000-resample paired-bootstrap 95% intervals for `PyTorch/portable DVEB`.

| Case | Threads | PyTorch ms | Portable DVEB ms | Speedup | Bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| E1 | 1 | 4.225 | 1.345 | **3.142x** | 3.121--3.165 |
| E1 | 6 | 4.313 | 1.377 | **3.133x** | 3.119--3.153 |
| E1 | 12 | 4.785 | 1.585 | **3.020x** | 3.002--3.039 |
| E2 | 1 | 23.342 | 12.575 | **1.856x** | 1.853--1.861 |
| E2 | 6 | 728.657 | 162.631 | **4.480x** | 1.957--4.738 |
| E2 | 12 | 572.177 | 161.747 | **3.537x** | 3.214--3.910 |
| E3 | 1 | 889.953 | 545.402 | **1.632x** | 1.582--1.691 |
| E3 | 6 | 2,274.406 | 402.486 | **5.651x** | 5.184--6.010 |
| E3 | 12 | 2,502.886 | 467.736 | **5.351x** | 4.610--5.947 |
| E4 | 1 | 13.762 | 8.947 | **1.538x** | 1.531--1.547 |
| E4 | 6 | 393.506 | 100.629 | **3.910x** | 3.593--4.356 |
| E4 | 12 | 259.548 | 98.928 | **2.624x** | 2.241--4.005 |
| E5 | 1 | 193.639 | 185.220 | **1.045x** | 1.036--1.055 |
| E5 | 6 | 1,046.468 | 251.334 | **4.164x** | 2.865--4.607 |
| E5 | 12 | 1,296.330 | 240.628 | **5.387x** | 4.436--6.301 |
| E6 | 1 | 4,171.380 | 2,090.051 | **1.996x** | 1.948--2.022 |
| E6 | 6 | 2,872.911 | 991.997 | **2.896x** | 2.744--2.961 |
| E6 | 12 | 3,344.029 | 926.044 | **3.611x** | 3.316--4.087 |

The mechanical rule reports:

- PyTorch faster: 0/18 lanes;
- portable DVEB faster: 18/18;
- preferred wins of at least 1.5x: 17/18;
- catastrophic lanes: 0;
- systematic domination: false;
- geometric mean `PyTorch/portable DVEB`: **2.975x**;
- formal result: **STRONG GO**.

## Deployment characterization

Thirty fresh-process E1/T1 pairs measured import-through-complete-fit behavior:

| Implementation | Median | p05--p95 | Median max RSS |
|---|---:|---:|---:|
| CPU PyTorch | 1.087 s | 1.071--1.096 s | 617,260 KiB |
| Portable DVEB, torch blocked | 0.236 s | 0.234--0.240 s | 101,982 KiB |

All 30 DVEB fits converged with `torch` absent. These are operational,
nondecision measurements, but they directly support the intended CPU-only
deployment story.

The compressed wheel is 3.53 MB and its installed site is 16.52 MB, including
the full PyStatistics package. The incremental native pieces are a 47,977-byte
DVEB library and a 2.11 MB vendored OpenMP runtime. In contrast, this specific
CUDA-enabled comparator environment contains a 1.64 GB `torch` package tree
and 4.57 GB `nvidia` package tree. Those comparator sizes describe this
environment only; they are not a claim about every CPU-only PyTorch build.

The paired timing process loaded both PyTorch's and the wheel's OpenMP runtime.
That is an honest property of coexistence testing. The intended torch-free
product loads only the wheel-local runtime.

## Historical native comparison

Comparing portable medians with the older Forge-native campaign gives a 2.53x
geometric portable/native runtime ratio. This is **not an isolated portability
penalty**: the campaigns differ in NumPy, SciPy, PyTorch, committed input
bytes, and session state, and several multi-thread PyTorch lanes also changed
dramatically. The cross-campaign numbers are retained only as a diagnostic.
They do not affect STRONG GO and do not justify compiler optimization.

A same-environment native-versus-portable attribution study would require its
own prospective protocol if product decisions ever depend on that question.

## Interpretation and next step

The first distributable DVEB consumer artifact preserves the language's
central value for direct MVN-MLE: it is compact, works without PyTorch, and is
not merely non-dominated—it wins this complete admitted grid. This closes the
CPU-only MVN-MLE artifact milestone on the research branch.

The next justified step is product/API planning, not post-result tuning:
decide whether the research branch should expose an experimental wheel build
to controlled SGCX testing, and separately choose the next general DVEB
capability or PyStatistics workload. Nothing here changes the publication
firewall, PyPI 6.1.5, `main`, or the default PyTorch route.

## Evidence

- v2 protocol: `docs/research/DVEB_PORTABLE_MVNMLE_PERFORMANCE_PROTOCOL_V2.md`;
- freeze, invalidated v1, fixtures, qualification, raw data, analysis,
  operational data, footprint, and exact artifact:
  `docs/research/evidence/dveb_mvnmle_portable/`;
- mechanical verifier:
  `benchmarks/dveb_mvnmle_portable/verify_evidence.py`.
