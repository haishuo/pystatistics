# DVEB CPU-only Linux wheel qualification results

Date: 2026-09-02

## Outcome

**Engineering result: PASS. Formal frozen-protocol decision: NO-GO.**

The engineering question was answered positively: this research branch now
builds self-contained, CPU-only PyStatistics wheels for CPython 3.11 and 3.12.
They require only NumPy and SciPy by default; neither PyTorch nor CUDA is
installed or loaded; the OpenMP runtime is carried inside the wheel; the
loader checks x86-64-v2 before opening native code; and the admitted MVN-MLE
fit passes in three clean Linux environments.

The formal decision remains NO-GO because the frozen protocol also required
the **complete branch test suite to pass absolutely**. It produced 4,478
passes, 94 skips, and two failures outside DVEB/MVN-MLE. Both failures reproduce
unchanged from the untouched pre-task commit `da75578`, and this task changed
none of their tests or implementations. That proves the packaging work did not
introduce them, but it does not make the literal predeclared gate true. The
threshold is not rewritten after results.

This distinction is deliberate:

- the CPU-only packaging mechanism is demonstrated and usable for research;
- release qualification is not granted under this protocol;
- a fresh protocol could prospectively use a no-new-regressions gate, but that
  requires a separate authorization and fresh qualification run.

## Built wheels

Both wheels were built from commit `7ec391c` in the official PyPA
`manylinux_2_28_x86_64` image pinned at digest
`sha256:0536c364...4a67b63`, using glibc 2.28, GCC 14.2.1, and auditwheel
6.8.2.

| ABI | Final wheel size | SHA-256 |
|---|---:|---|
| CPython 3.11 | 3,529,411 bytes | `31aa61e7...9ba877f8` |
| CPython 3.12 | 3,481,527 bytes | `7e9426ae...30bf0218` |

Auditwheel found both wheels eligible for `manylinux_2_24_x86_64` as well as
the requested 2.28 tag. The support claim remains at the prospectively frozen
boundary: mainstream glibc 2.28+ Linux on x86-64-v2 CPUs. The stronger observed
tag is reported, not adopted post hoc.

The wheel-local native pieces are:

- repaired DVEB ABI: 47,977 bytes,
  SHA-256 `65438a0d...229fe38`;
- vendored `libgomp`: 2,114,857 bytes;
- no DVEB compiler/runtime installation;
- no PyTorch, CUDA, or NVIDIA native dependency;
- relative `$ORIGIN` search path into the wheel-local library directory, with
  no absolute build-host path.

The source-tree artifact before auditwheel is 34,968 bytes,
SHA-256 `69e93f9f...4c2e2a00`. Auditwheel changes the ELF dependency name and
relative search path, so the final manifest is refreshed and the wheel RECORD
regenerated. The installed loader verifies the resulting hash before loading.

## Runtime qualification

| Environment | Python | Network during check | Result |
|---|---:|---:|---|
| AlmaLinux 8.10, glibc 2.28 | 3.11.16 | disabled | PASS |
| Debian 13, glibc 2.41 (`sgcbio-worker:0.6.6`) | 3.12.13 | disabled | PASS |
| Forge Ubuntu 24.04, glibc 2.39 | 3.12.3 | host check | PASS |

In every environment:

- `torch` was absent before and after the fit;
- the only base requirements were NumPy and SciPy;
- all ELF dependencies resolved, including wheel-local `libgomp`;
- all x86-64-v2 features were established before loading;
- ABI v1 and the artifact manifest agreed;
- serial, work-item-parallel, and automatic schedules agreed within the
  existing `1e-12` integration threshold;
- the Apple public fit converged at log-likelihood
  `-74.21747613185018`, passed the committed R-derived result thresholds, and
  left caller-owned bytes unchanged;
- `datasets.missvals` returned finite output.

Serial and automatic schedules were bit-identical. The parallel objective
value was bit-identical; its gradient stayed within the predeclared threshold
but was not bit-identical, consistent with reduction order. No threshold was
changed.

## CPU and platform refusal boundary

The loader now requires Linux x86-64 and the complete psABI x86-64-v2 feature
set: CX16, LAHF-SAHF, POPCNT, SSE3, SSSE3, SSE4.1, and SSE4.2. It refuses
before `ctypes.CDLL` when the platform is wrong, features are absent or cannot
be established, the artifact is missing, or the hash is wrong. There is no
silent fallback.

This is intentionally a bounded product promise. No support is claimed for
old x86-64-v1 CPUs, musl/Alpine, Arm, macOS, Windows, CPython 3.13, or every
Linux distribution. The three runtime environments share Forge's physical
CPU, so the result tests distribution/glibc/Python portability, not distinct
CPU hardware. Negative ISA behavior is covered by injected pre-load tests.

## Preservation of the earlier performance result

The earlier 18-lane MVN-MLE campaign timed the Forge-native artifact
`a96e4102...1fb102f`; the portable wheel carries a different binary. The
original ELF is now preserved at
`docs/research/evidence/dveb_mvnmle/baseline_artifacts/` and the existing
evidence verifier passes against that exact artifact: 540 campaign pairs,
decision GO, geometric-mean PyTorch-over-DVEB ratio 2.325837.

That timing is **not transferred** to the portable wheel. This packaging study
ran no performance campaign and makes no speed claim for the x86-64-v2 build.

## Failures and deviations preserved

1. Static `libgomp.a` could not be linked into a shared object because the
   builder archive is not position-independent. Standard auditwheel bundling
   was used instead.
2. The first R1 checker run rejected optional `gpu`/`all` extras as if they
   were base requirements. The environment itself contained no PyTorch; the
   checker was corrected and the fresh offline run passed.
3. Two committed-tree build attempts stopped before producing wheels: one
   lacked linked-worktree metadata inside the container; one hit Git's
   safe-directory guard. The final pipeline snapshots with an explicit
   per-command safe-directory setting and does not mutate global Git config.
4. The complete suite's two failures were reproduced from `da75578`:
   `test_describe_kurtosis` and
   `test_complete_separation_vcov_fails_loud`. They are outside this task and
   were not modified.

## Artifacts and verification

- Protocol: `docs/research/DVEB_CPU_ONLY_WHEEL_PROTOCOL.md`
- Machine-readable result:
  `docs/research/evidence/dveb_cpu_only_wheel/qualification.json`
- Per-runtime evidence and auditwheel reports:
  `docs/research/evidence/dveb_cpu_only_wheel/`
- Mechanical verifier:
  `packaging/dveb_mvnmle/verify_cpu_only_evidence.py`
- Reproducible build:
  `packaging/dveb_mvnmle/build_wheels.sh`

The final wheels remain local under
`/tmp/pystatistics-dveb-cpu-wheel-final3/final/`. They are not committed,
published, or installed into the frozen PyStatistics release. Nothing was
pushed.
