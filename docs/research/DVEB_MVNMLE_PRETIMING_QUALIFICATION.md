# DVEB MVN-MLE consumer integration — pre-timing qualification

**Date:** 2026-09-01

**Adapter commit:** `84a0535`

**Status:** FUNCTIONAL GATES PASS; prospective timing remains unstarted

## Artifact and route

- Bundled research artifact SHA-256:
  `a96e410282fae692f532185ba9c6d0377885f9f7f2b696ff772eda13b1fb102f`.
- ABI version 1 and every required symbol load successfully.
- Dynamic audit reports only the ordinary C/C++/math/OpenMP runtime; no
  PyTorch, CUDA, NVIDIA, RPATH, or RUNPATH dependency.
- Explicit `solver='dveb'` works only with direct CPU fitting and refuses
  incompatible methods/backends.
- The default `backend='cpu'` route remains
  `cpu_cholesky_fp64` (PyTorch) and never selects DVEB.
- A subprocess that rejects every attempted `torch` import fits the apple
  data successfully through DVEB and leaves `torch` absent from
  `sys.modules`.

## Numerical qualification

The exact Trunk 008 Q1--Q4 generators and parameter points were reused from
the read-only DVEB repository. For every case:

- four deterministic parameter points;
- threads 1, 6, and 12;
- automatic, forced-serial, and forced-work-item-parallel schedules;
- two repeated calls per lane;
- comparison against the direct NumPy authority and incumbent CPU PyTorch.

All 144 DVEB case/point/thread/schedule combinations passed the frozen value
and gradient tolerances, chose the expected schedule, and were bitwise stable
on repeated calls for a fixed schedule/thread count. Incumbent PyTorch passed
all 16 case/point checks. Q1, Q3, and Q4 finite-difference checks passed.

Serial and parallel schedules are not asserted bit-identical to each other.
On the apple initial point their deterministic reduction orders differed by
`6.22e-15` in one gradient element, far inside the frozen numerical contract.
No tolerance was changed; the first draft test had incorrectly imposed a
stronger cross-schedule identity requirement than the protocol.

## Test gates

- Targeted integration/routing: 30 passed, 1 expected skip.
- Complete MVN-MLE suite: 272 passed, 2 expected skips.
- Configured lint over the adapter, backend, routing, and integration tests:
  pass.
- Complete non-slow suite: 4,475 passed, 94 skipped, one inherited failure.

The inherited failure is
`tests/multinomial/test_multinom.py::TestFailureCases::test_complete_separation_vcov_fails_loud`.
It reproduces unchanged when run alone at pre-integration commit `a9c7e4d`.
The protocol's F5 correction records it; the unrelated module is untouched.

No performance observation, threshold change, default-route change, merge,
tag, release, or push occurred during qualification.
