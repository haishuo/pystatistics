# DVEB MVN-MLE controlled evaluation handoff

**Date:** 2026-09-02

**Status:** technically admitted for controlled internal evaluation; not a
release, default, merge, or publication authorization

**Research branch:** `research/dveb-pystatistics`

**Qualified source commit:** `003e723`

## Decision

The branch-only CPU DVEB implementation of direct forward-Cholesky MVN-MLE is
ready for controlled SGCX evaluation on machines inside its qualified platform
contract.

This decision follows three completed prospective qualifications:

1. the PyStatistics integration passed its correctness, refusal, ABI, and
   non-domination gates;
2. a self-contained CPU-only Linux wheel passed the corrected portability and
   regression protocol; and
3. that exact installed portable artifact produced a **STRONG GO** against the
   optimized CPU-PyTorch implementation: DVEB was faster in 18/18 lanes, at
   least 1.5x faster in 17/18, with a 2.975x geometric-mean speedup.

The admitted artifact also has a measured operational benefit: the fresh
torch-blocked DVEB process used about 100 MiB peak RSS and reached an answer in
0.236 seconds, while the comparison PyTorch process used about 617 MiB and
took 1.087 seconds in the frozen environment.

These results justify evaluation, not generalization. They establish this
solver, artifact, platform boundary, and admitted case grid only.

## Publication firewall

The following rules are absolute until the MVN-MLE paper danger period ends:

- PyPI 6.1.5, tag `v6.1.5`, and `main` remain unchanged.
- This research branch must not be merged to `main`.
- No public PyStatistics release, public package index upload, release tag, or
  change to the paper replication environment is permitted.
- A research wheel must identify itself as `6.2.0.dev0+dveb`.
- The existing PyTorch route remains the default. DVEB is selected only by the
  explicit `solver="dveb"` request.
- Failure to load or execute DVEB is a loud error. It must never silently run
  another solver.

Publishing the branch to GitHub is compatible with this firewall. Publishing
a package or merging the branch is not.

## Qualified artifact contract

The controlled lane is intentionally narrow:

| Property | Qualified value |
|---|---|
| Operating system | glibc Linux |
| Architecture | x86-64-v2 |
| Wheel policy claim | manylinux_2_28_x86_64 |
| CPython | 3.11 and 3.12 |
| Numerical type | IEEE float64, contraction disabled |
| DVEB ABI | dense CPU ABI v1 |
| Native dependencies | wheel-local `libgomp`; no CUDA, NVIDIA, PyTorch, RPATH, or RUNPATH dependency |
| Python dependencies | the ordinary NumPy/SciPy base requirements |
| Statistical scope | direct forward-Cholesky MVN-MLE only |
| Selection | explicit `solver="dveb"` with CPU backend |

The final installed native payload used in the performance campaign has
SHA-256
`65438a0d742257acc0dc7bd2ff13d2017669fd21dc07f64add41b18fb229fe38`.
The admitted CPython 3.11 wheel has SHA-256
`92fa398c2785eac02d824c4bb589cabc2a9a966b4c87ab915a622df5b44caa4a`;
the admitted CPython 3.12 wheel has SHA-256
`7fb9c35c17ac169556cf0529ee5f3c67d05f1a173bfeefa1c6263fd185f03565`.

No support is claimed for x86-64-v1, musl, Arm Linux, macOS, Windows, CPython
3.13, or an arbitrary Linux distribution merely because it can import a wheel.
Unsupported machines must be refused before the native library is loaded.

## Controlled distribution procedure

The preferred internal handoff is a wheel file plus its machine-readable
qualification summary, transferred directly to the evaluation machine. It is
not uploaded to PyPI and is not represented as a supported public release.

For an exact-evidence evaluation, use the already qualified wheel whose hash is
listed above. If it must be rebuilt, check out commit `003e723` and use the
pinned manylinux builder and existing `packaging/dveb_mvnmle/` machinery. A
rebuild is a new candidate: verify its wheel and installed-payload hashes and
run the complete installed-wheel qualification before giving it to an
evaluator. Do not assume archive-level byte reproducibility from source-level
identity.

Every internal handoff must include:

- the wheel and SHA-256;
- the branch, source commit, research version, and build date;
- `docs/research/evidence/dveb_cpu_only_wheel_v2/qualification.json`;
- `docs/research/evidence/dveb_mvnmle_portable/qualification-summary-v2.json`;
- this support boundary; and
- an explicit statement that the artifact is experimental and cannot be used
  as the paper's published package.

## Evaluation procedure

An evaluation machine is admitted only if it is glibc Linux, has an x86-64-v2
CPU, and runs CPython 3.11 or 3.12. Installation and testing must occur in a
fresh isolated environment.

The evaluator must record:

1. operating-system, glibc, Python, and CPU-feature identity;
2. wheel and installed native-payload hashes;
3. the result of the artifact's pre-load ISA/platform guard;
4. confirmation that importing and fitting do not import PyTorch;
5. the full focused DVEB MVN-MLE integration suite;
6. at least the Apple reference fit and one representative local incomplete
   dataset, with caller inputs checked for mutation;
7. selected DVEB schedule and thread count; and
8. any crash, refusal, convergence difference, warning difference, or result
   discrepancy, including the complete input needed to reproduce it when data
   policy permits.

Performance measurements are optional in this lane and are descriptive. New
hardware results must not be pooled with the frozen Forge campaign or used to
retune the artifact after observation.

## Support and rollback boundary

The controlled lane is designed to avoid turning SGCX into general Linux IT
support:

- support covers only the declared wheel/platform contract and the explicit
  DVEB solver;
- unsupported distributions, CPUs, Python versions, or locally rebuilt
  binaries receive a clear refusal rather than best-effort debugging;
- the evaluator can return immediately to the unchanged PyTorch default by
  omitting `solver="dveb"` or by uninstalling the research wheel; and
- no evaluation finding changes the released 6.1.5 package.

A defect in the admitted lane is fixed on the research branch and requalified
as a new artifact. It is never patched into the frozen release line.

## Graduation criteria after the publication freeze

No automatic graduation occurs. A later integration decision requires all of:

1. the paper firewall has been explicitly lifted;
2. controlled evaluations show no unresolved correctness, ABI, or platform
   failures;
3. a new integration branch reconciles DVEB, ALLOY, and intervening main-line
   changes deliberately;
4. the public API, fallback policy, support matrix, and release packaging are
   reviewed as product decisions;
5. a fresh complete regression and public-data applied example pass; and
6. a new explicit merge and release authorization is given.

Until then, the correct status is: **qualified research artifact, explicit
opt-in, controlled internal evaluation only**.

## Evidence

- Integration result: `docs/research/DVEB_MVNMLE_INTEGRATION_RESULTS.md`
- Wheel result: `docs/research/DVEB_CPU_ONLY_WHEEL_RESULTS_V2.md`
- Portable performance result:
  `docs/research/DVEB_PORTABLE_MVNMLE_PERFORMANCE_RESULTS.md`
- Mechanical portable-evidence verifier:
  `benchmarks/dveb_mvnmle_portable/verify_evidence.py`

