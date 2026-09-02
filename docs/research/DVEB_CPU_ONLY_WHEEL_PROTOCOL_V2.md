# DVEB CPU-only Linux wheel qualification protocol v2

Status: **FROZEN BEFORE FRESH QUALIFICATION**

Date frozen: 2026-09-02

This protocol belongs only to the unpublished
`research/dveb-pystatistics` branch. It does not change the PyPI release,
does not authorize a merge to `main`, and does not authorize publication.
The v1 protocol and its formal NO-GO at commit `e043cf9` are permanent
negative evidence. This protocol does not reinterpret or overwrite them.

## 1. Question and decision boundary

The question is whether the branch can produce a genuinely CPU-only
PyStatistics wheel containing the admitted DVEB MVN-MLE implementation:

- no PyTorch package requirement;
- no CUDA, NVIDIA, or PyTorch native-library dependency;
- no dependency on a preinstalled DVEB compiler or runtime;
- no dependency on a host-provided OpenMP runtime outside the manylinux
  contract;
- an explicit, checked CPU ISA requirement;
- correct execution in clean, torch-free Linux environments; and
- no new branch regression relative to the frozen pre-packaging baseline.

This is a packaging and portability qualification, not a performance
campaign. The previously committed MVN-MLE campaign remains the performance
authority. No timing result can change this protocol.

**GO** requires every mandatory build, inspection, installation, refusal,
and numerical check in this document to pass, plus the baseline-relative
regression rule in section 8.

**NO-GO** follows if any required wheel cannot be built and audited, if a
forbidden dependency is present, if the artifact cannot be loaded in an
admitted runtime environment, if the CPU requirement is not checked before
loading, if any numerical check fails, or if the full branch suite contains
any failure outside the exact baseline allowlist in section 8. Failures are
recorded; thresholds and the allowlist are not relaxed after observation.

## 2. Frozen relationship to v1

Protocol v1 remains authoritative for all requirements except its absolute
full-suite gate. In particular, v2 adopts without change v1 sections 2--7,
the installed-wheel correctness checks in section 8, the evidence rules in
section 9, and the interpretation limits in section 10.

The sole prospective decision-rule correction is this:

> A packaging change is qualified against the branch state it actually
> inherited. The complete suite must introduce no new failure relative to
> frozen baseline commit `da75578`; two failures already present there do not
> become attributable to DVEB packaging merely because the qualification is
> comprehensive.

This correction is frozen before rebuilding wheels or observing any v2 test
or runtime result. It does not retroactively turn v1 into a GO.

## 3. Frozen identities

- repository: `sgcx-org/pystatistics`
- branch: `research/dveb-pystatistics`
- inherited baseline commit: `da75578`
- v1 evidence commit and v2 pre-protocol HEAD: `e043cf9`
- package version: `6.2.0.dev0+dveb`
- builder image:
  `quay.io/pypa/manylinux_2_28_x86_64@sha256:0536c364004fa2a3c5041120b6fe35d84fc5bfe31f04c6a6304f13eac4a67b63`
- runtime environments: v1 R1, R2, and R3, unchanged
- DVEB source and generated-input identities: v1 section 3, unchanged
- build flags, wheel ABIs, manylinux policy, feature guard, and dependency
  prohibitions: v1 sections 4--6, unchanged

The v2 wheels must be rebuilt from the protocol commit in a fresh output
directory. Reusing v1 wheels or runtime reports is forbidden. Build products
need not be committed; their hashes and the complete qualification evidence
must be.

## 4. Fresh wheel and runtime qualification

The campaign must:

1. build CPython 3.11 and 3.12 wheels from the committed protocol tree in the
   pinned builder image;
2. repair and inspect both with auditwheel under the frozen v1 rules;
3. install and run the CPython 3.11 wheel in R1 with network disabled;
4. install and run the CPython 3.12 wheel in R2 with network disabled;
5. install and run the CPython 3.12 wheel in a fresh R3 environment;
6. run all v1 numerical, identity, dependency, input-ownership, schedule,
   platform/ISA, missing-artifact, and corrupt-artifact checks unchanged; and
7. preserve build attempts, failures, exclusions, and deviations.

Ordinary execution may not download, compile, calibrate, or search for an
alternative library. A failed runtime or static check is a NO-GO; it cannot
be substituted with v1 evidence.

## 5. Baseline-relative regression gate

The complete branch suite is run from the protocol commit after the fresh
wheel/runtime qualification. Its frozen allowlist is exactly:

1. `tests/descriptive/test_gpu.py::TestGPUvsCPU::test_describe_kurtosis`
2. `tests/multinomial/test_multinom.py::TestFailureCases::test_complete_separation_vcov_fails_loud`

Both were observed under v1 and independently reproduced at untouched
baseline commit `da75578`. The packaging task changed neither their tested
implementations nor their tests.

The gate passes only if:

- there are zero failed node IDs outside this exact allowlist;
- an allowlisted node either passes or fails for the same qualitative reason
  recorded under v1; a new exception type, crash, collection failure, or
  materially worse failure is not allowed;
- collection completes normally;
- no test is deleted, weakened, newly skipped, or deselected to manufacture
  the result; and
- the focused DVEB MVN-MLE integration suite passes completely.

An allowlisted test becoming fixed is permitted and is reported. The
allowlist is not expanded after the run. This protocol does not authorize
fixing either inherited failure because doing so would mix unrelated product
work into the packaging qualification.

## 6. Evidence and mechanical decision

V2 evidence is committed separately under
`docs/research/evidence/dveb_cpu_only_wheel_v2/`. V1 evidence is never edited
or overwritten. At minimum v2 records:

- builder, input, artifact, and wheel identities;
- final auditwheel reports;
- R1/R2/R3 machine-readable runtime reports;
- the complete-suite command, counts, failed node IDs, and qualitative
  comparison with the v1/baseline failure signatures;
- the focused integration command and counts;
- every failure, deviation, and exclusion;
- an explicit v2 GO/NO-GO decision; and
- a mechanical evidence verifier that rejects drift in committed files and
  in the decision rule.

## 7. Interpretation limits

A v2 GO would establish that the research branch can ship the admitted DVEB
MVN-MLE CPU backend in a bounded, self-contained Linux wheel without adding a
regression to the branch it inherited. It would not assert that the two
inherited failures are acceptable product behavior, would not erase v1's
NO-GO, would not prove universal Linux portability, would not transfer the
Forge-native performance result to the portable artifact, would not make
DVEB a wholesale PyTorch replacement, and would not authorize release,
publication, merge to `main`, or modification of PyPI 6.1.5.
