# DVEB CPU-only Linux wheel qualification v2 results

Date: 2026-09-02

## Outcome

**Formal frozen-protocol decision: GO.**

Protocol v2 was frozen at commit `2639f19` before any fresh build or test
result. It retained every v1 portability, dependency, ISA, installed-wheel,
and numerical gate. Its sole prospective correction was to judge regressions
against the branch's frozen pre-packaging baseline `da75578`, with exactly two
predeclared inherited test failures and no post-result allowlist expansion.

Every wheel-specific gate passed in three fresh environments. The complete
suite produced 4,478 passes, 94 skips, 27 deselections, and exactly the two
allowlisted failures with the same qualitative signatures as v1. There were
no new failures. The focused DVEB MVN-MLE integration suite passed 18/18.

This GO does not revise v1. V1 remains a formal NO-GO under its absolute
full-suite rule; v2 answers a separately frozen no-new-regressions question.

## Fresh build

Both wheels were rebuilt from the committed protocol tree in the pinned PyPA
manylinux image at digest
`sha256:0536c364004fa2a3c5041120b6fe35d84fc5bfe31f04c6a6304f13eac4a67b63`.
The compiler flags, generated inputs, ABI, x86-64-v2 requirement, and
auditwheel procedure were unchanged from v1.

| ABI | Final wheel size | Fresh SHA-256 |
|---|---:|---|
| CPython 3.11 | 3,529,467 bytes | `92fa398c...44caa4a` |
| CPython 3.12 | 3,481,511 bytes | `7fb9c35c...5f03565` |

Wheel archive hashes differ from v1, as expected for non-byte-reproducible
wheel archives. The payload identity that matters did not drift:

- pre-auditwheel DVEB artifact: 34,968 bytes,
  `69e93f9f...4c2e2a00`;
- installed auditwheel-repaired artifact: 47,977 bytes,
  `65438a0d...229fe38`;
- wheel-local OpenMP runtime: `libgomp-e985bcbb.so.1.0.0`;
- base requirements: NumPy and SciPy only;
- forbidden PyTorch, CUDA, and NVIDIA base/native dependencies: none.

Auditwheel again found both artifacts eligible for the stronger
`manylinux_2_24_x86_64` tag. The product claim remains at the frozen
`manylinux_2_28_x86_64` plus x86-64-v2 boundary.

## Fresh installed-wheel qualification

| Environment | Python | Network during check | Result |
|---|---:|---:|---|
| AlmaLinux 8.10, glibc 2.28 | 3.11.16 | disabled | PASS |
| Debian 13, glibc 2.41 | 3.12.13 | disabled | PASS |
| Forge Ubuntu 24.04, glibc 2.39 | 3.12.3 | host check | PASS |

In every environment, PyTorch was absent before and after fitting; all native
dependencies resolved; the complete x86-64-v2 feature set was established
before loading; ABI and artifact identity matched; the Apple fit converged at
`-74.21747613185018`; the frozen R-derived result tolerances passed; the input
bytes remained unchanged; `datasets.missvals` returned a finite result; and
serial, work-item-parallel, and automatic schedules agreed within the frozen
threshold. Serial and automatic were bit-identical. The parallel objective
value was bit-identical and its gradient differed only by admitted reduction
order.

The focused 18-test suite also exercised the injected pre-load ISA and
platform refusals, missing and corrupted artifact refusals, no-fallback
behavior, boundary ownership checks, schedule equivalence, public result
identity, torch-blocked operation, and illegal route refusal.

## Baseline-relative regression result

The fresh complete-suite result was:

```text
2 failed, 4478 passed, 94 skipped, 27 deselected
```

The only failed nodes were the exact frozen allowlist:

1. `tests/descriptive/test_gpu.py::TestGPUvsCPU::test_describe_kurtosis`
   remained a small GPU/CPU kurtosis tolerance discrepancy.
2. `tests/multinomial/test_multinom.py::TestFailureCases::test_complete_separation_vcov_fails_loud`
   remained the expected-exception-not-raised failure.

Those match the v1/baseline failure character. Collection completed normally;
no test was removed, weakened, newly skipped, or deselected; and the branch
changes from `da75578` do not touch either affected test or implementation.
The frozen v2 regression gate therefore passes.

## Preserved process deviations

The first build attempt used an incompatible read-only worktree mount and
stopped before producing a wheel. A clean second output directory was used.
On Forge, the numerical qualifier passed but the compound command's final
inventory step initially failed because the intentionally pip-less venv could
not run `python -m pip freeze`; inventory was collected through the host
installer's `--python` interface. Neither event changed a threshold or
substituted old evidence. Full details are in `process-attempts.txt`.

## Interpretation

The branch can produce the admitted self-contained CPU-only Linux wheels
without requiring PyTorch or CUDA and without adding a regression to the
baseline it inherited. This is evidence for the SGCX deployment objective: a
small native CPU artifact can retain the admitted DVEB MVN-MLE path without
shipping an ML/GPU runtime stack.

It does not prove universal Linux portability, qualify x86-64-v1/musl/Arm/
macOS/Windows/CPython 3.13, transfer the earlier Forge-native performance
claim to this portable binary, make DVEB a blanket PyTorch replacement,
authorize release, merge to `main`, publication, or any change to PyPI 6.1.5.

## Evidence

- Protocol: `docs/research/DVEB_CPU_ONLY_WHEEL_PROTOCOL_V2.md`
- Machine-readable decision:
  `docs/research/evidence/dveb_cpu_only_wheel_v2/qualification.json`
- Runtime, auditwheel, suite, inventory, and process records:
  `docs/research/evidence/dveb_cpu_only_wheel_v2/`
- Mechanical verifier:
  `packaging/dveb_mvnmle/verify_cpu_only_evidence_v2.py`

Fresh wheels remain local under
`/tmp/pystatistics-dveb-cpu-wheel-v2-2639f19-attempt2/final/`. They were not
published, merged, or installed into the frozen release.
