# DVEB portable-wheel MVN-MLE performance qualification v2

Status: **FROZEN BEFORE V2 QUALIFICATION OR TIMING**

Date frozen: 2026-09-02

V2 incorporates
`docs/research/DVEB_PORTABLE_MVNMLE_PERFORMANCE_PROTOCOL.md` in full except
for the input-byte authority corrected here. V1 is permanently invalid and
retained under `docs/research/evidence/dveb_mvnmle_portable/`.

## Correction made before timing

V1 required current E1--E6 arrays to equal hashes produced by an older NumPy
version. Its untimed qualification established that the generator remained
deterministic within the current environment and that both implementations
were numerically admitted, but all six cross-version hashes differed. V1
halted before any timed observation.

V2 freezes the evaluation values directly as six committed, non-pickled NumPy
arrays under `benchmarks/dveb_mvnmle_portable/fixtures/`. They were generated
once from the unchanged E1--E6 specifications and seeds using the comparator
environment frozen by v1. Each file has both a file SHA-256 and a raw
contiguous-array SHA-256 in the v2 freeze manifest. Workers load, validate,
copy, and re-hash the fixtures; they never call the random generator.

This correction prevents library-version-dependent regeneration and makes
the evaluated values reviewable and exactly reproducible. The same fixture
bytes go to PyTorch and portable DVEB at every thread count. The v1
qualification values were not selected for favorable timing—no timing
existed—and no array is changed after this freeze.

## Unchanged contract

Everything else is unchanged from v1:

- exact installed wheel and artifact identities;
- comparator environment;
- E1--E6 shapes, missingness designs, and 1/6/12 thread lanes;
- both numerical implementations and full `mlest` endpoint;
- correctness tolerances;
- three warmups and 30 paired randomized repetitions;
- no observation exclusion;
- catastrophic and systematic domination rules;
- GO, STRONG GO, and preferred-win thresholds;
- nondecision operational characterization;
- publication firewall and interpretation limits.

V2 qualification is fresh. V1's 18 numerical passes do not substitute for
it. No timing begins unless all fixture hashes, installed identities, and
correctness lanes pass under the corrected freeze.
