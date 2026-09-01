# DVEB CPU-only Linux wheel qualification protocol

Status: **FROZEN BEFORE IMPLEMENTATION OR QUALIFICATION**

Date frozen: 2026-09-01

This protocol belongs only to the unpublished
`research/dveb-pystatistics` branch. It does not change the PyPI release,
does not authorize a merge to `main`, and does not authorize publication.

## 1. Question and decision boundary

The question is whether the branch can produce a genuinely CPU-only
PyStatistics wheel containing the admitted DVEB MVN-MLE implementation:

- no PyTorch package requirement;
- no CUDA, NVIDIA, or PyTorch native-library dependency;
- no dependency on a preinstalled DVEB compiler or runtime;
- no dependency on a host-provided OpenMP runtime outside the manylinux
  contract;
- an explicit, checked CPU ISA requirement;
- correct execution in clean, torch-free Linux environments.

This is a packaging and portability qualification, not a performance
campaign. The previously committed MVN-MLE campaign remains the performance
authority. No timing result can change this protocol.

**GO** requires every mandatory build, inspection, installation, refusal,
and numerical check in this document to pass.

**NO-GO** follows if any required wheel cannot be built and audited, if a
forbidden dependency is present, if the artifact cannot be loaded in either
admitted runtime environment, if the CPU requirement is not checked before
loading, or if any numerical check fails. Failures are recorded; thresholds
are not relaxed after observation.

## 2. Scope and nonclaims

The admitted product is deliberately narrow:

- Linux on x86-64;
- glibc 2.28 or newer;
- CPython 3.11 and 3.12;
- CPUs satisfying the complete x86-64-v2 feature level;
- IEEE float64 with contraction disabled and no fast-math;
- the branch-only DVEB direct MVN-MLE solver and dense CPU ABI v1.

The study does **not** claim support for musl/Alpine, 32-bit systems, Arm,
macOS, Windows, CPython 3.13, x86-64-v1 CPUs, unusual Linux distributions, or
every possible combination of system libraries. It does not make DVEB the
default solver and does not remove PyTorch from other PyStatistics backends.
It does not qualify an sdist. Those are separate decisions.

The wheel's platform promise follows PEP 600: a `manylinux_2_28_x86_64` tag
promises compatibility with mainstream x86-64 glibc distributions at glibc
2.28 or newer. The wheel adds the stricter x86-64-v2 ISA requirement because
wheel tags do not express x86-64 microarchitecture levels.

## 3. Frozen inputs and provenance

### 3.1 Consumer repository

- repository: `sgcx-org/pystatistics`
- branch: `research/dveb-pystatistics`
- pre-protocol HEAD: `da75578`
- package version: `6.2.0.dev0+dveb`
- existing consumer artifact SHA-256:
  `a96e410282fae692f532185ba9c6d0377885f9f7f2b696ff772eda13b1fb102f`
- existing artifact status: Forge-known-machine research baseline, retained
  in history but not admitted as the portable wheel artifact

### 3.2 DVEB authority (read-only for this task)

- repository commit: `a4b0286`
- branch: `codex/trunk-008-dense-cpu`
- generated source:
  `build/mvnmle_cpu_x86_64_v2/mvnmle_cpu_dense.cpp`
- generated source SHA-256:
  `407e77b0ad55a92719a3a70139901bdfde0576bb1da74054824e242ec3213879`
- ABI header SHA-256:
  `7c4871bb51894f588773a1142c282dd370dcf5b5eedd374209f01a47db846be2`
- dense runtime header: `dvebrt/dense/dense_rt.h`
- dense runtime header SHA-256:
  `1347391466b645fa2132c8621c279944ccf17d262392402d2e52500cac74d071`
- already qualified Forge-built x86-64-v2 artifact SHA-256:
  `4ceb3800340611eac3d2c3308646fe25787cc423dc5c9c1d48843922e70aad21`
- qualification authority: `evidence/trunk008/qualification.json`, whose
  overall result is PASS and which separately admits the portable-ISA cases

The generated C++, ABI header, and runtime header will be copied verbatim
with hashes and provenance into a branch-local packaging input directory.
They are generated consumer inputs, not a PyStatistics-specific change to
the DVEB compiler. DVEB and GradFlow remain unmodified.

## 4. Frozen build environments and commands

### 4.1 Builder

Official PyPA image:

`quay.io/pypa/manylinux_2_28_x86_64@sha256:0536c364004fa2a3c5041120b6fe35d84fc5bfe31f04c6a6304f13eac4a67b63`

Observed preflight identity:

- AlmaLinux 8.10;
- glibc 2.28;
- GCC 14.2.1;
- auditwheel 6.8.2;
- `/opt/python/cp311-cp311` and `/opt/python/cp312-cp312`.

The generated DVEB source is compiled once in this image with:

```text
-O3 -std=c++17 -fPIC -shared -fopenmp
-ffp-contract=off -march=x86-64-v2
```

Forbidden flags include `-march=native`, `-mtune=native`, `-ffast-math`, and
`-Ofast`. The exact command, compiler identity, input hashes, output hash,
ELF dependencies, and ELF version requirements are recorded.

The OpenMP runtime may be bundled by `auditwheel`; it may not remain an
unresolved host requirement. A failed preflight showed that the builder's
static `libgomp.a` is not position-independent and therefore cannot be linked
directly into this shared object. That failed feasibility check is not a
qualification result.

### 4.2 Wheel production

For CPython 3.11 and 3.12 independently:

1. build a platform wheel from a clean source checkout in the pinned image;
2. repair it with `auditwheel` to the `manylinux_2_28_x86_64` policy;
3. update the bundled artifact manifest to the repaired artifact's SHA-256;
4. regenerate wheel `RECORD` through standard wheel tooling;
5. run `auditwheel show` again on the final wheel.

The loader reads the bundled manifest and verifies the artifact hash before
`ctypes` loads it. The wheel's `RECORD` authenticates both loader and manifest
as installed from the built wheel. This is accidental-corruption detection,
not a claim of resistance to an attacker able to rewrite installed Python
code.

No source tree outside the approved PyStatistics research worktree is
modified. Wheels and scratch environments are generated under ignored build
or evidence paths. Nothing is uploaded.

## 5. CPU feature guard

Before opening the shared library, the loader must establish Linux x86-64
and all x86-64-v2 additions defined by the x86-64 psABI:

- CMPXCHG16B (`cx16`);
- LAHF-SAHF (`lahf_lm` in Linux flags);
- POPCNT;
- SSE3 (`pni` or `sse3` in Linux flags);
- SSSE3;
- SSE4.1;
- SSE4.2.

The feature probe is deterministic and occurs outside numerical timing. If
the flags cannot be established or any required flag is absent, the loader
refuses before `ctypes.CDLL` and explicitly states that no fallback was
attempted. Unit tests inject complete, incomplete, and unavailable flag sets;
the real admitted environments must also report a complete set.

## 6. Mandatory static inspections

Each final wheel must satisfy all of the following:

1. filename and WHEEL metadata carry CPython ABI and
   `manylinux_2_28_x86_64` compatibility;
2. METADATA has no required dependency whose normalized name is `torch` or
   begins with `nvidia-` or `cuda`;
3. the wheel contains the DVEB artifact, its manifest, and any auditwheel
   vendored runtime library;
4. the installed artifact hash equals the manifest value;
5. `auditwheel show` accepts the final wheel at the promised policy;
6. all ELF `NEEDED` entries resolve in each admitted runtime;
7. no resolved library path or ELF dependency contains `torch`, `cuda`, or
   `nvidia`;
8. any RUNPATH is relative and contained within the installed wheel; no
   absolute build-host path is permitted;
9. the generated artifact has no unresolved DVEB runtime dependency;
10. wheel and installed sizes are recorded, not thresholded.

## 7. Mandatory runtime environments

### R1: policy-floor environment

The CPython 3.11 wheel is installed into a clean virtual environment inside
the pinned manylinux builder (AlmaLinux 8.10, glibc 2.28). Only the final
wheel and its declared NumPy/SciPy dependencies may be installed. PyTorch is
absent.

### R2: disjoint application environment

The CPython 3.12 wheel is installed into a clean environment in the existing
local image `sgcbio-worker:0.6.6` (Debian 13, glibc 2.41, Python 3.12.13,
NumPy 2.4.6, SciPy 1.18.0). The preflight found no PyTorch installation.
The image is run with its service entrypoint disabled. The image is a runtime
qualification target only; it is not a build authority.

### R3: Forge host

The CPython 3.12 wheel is installed into a new, ignored virtual environment
on Forge (Ubuntu 24.04, glibc 2.39). The environment is created without
system site packages and must not contain PyTorch.

R1 and R2 are mandatory GO environments. R3 is mandatory as a regression
against the development host. All share the same physical CPU, so this study
does not claim hardware-diverse ISA validation. The injected refusal tests
cover the negative ISA branch without pretending to emulate an old CPU.

After installation, the numerical smoke test is rerun with network access
disabled where container tooling permits it. Ordinary execution must not
download, compile, calibrate, or search for another library.

## 8. Correctness gates before admission

The source-tree regression must pass first. Then each wheel/runtime pair must
run a standalone installed-wheel checker that does not import the source
checkout and establishes:

1. `importlib.util.find_spec("torch") is None`;
2. `torch` is absent from `sys.modules` before and after the fit;
3. the artifact and manifest identities are internally consistent;
4. ABI version is 1;
5. serial, work-item-parallel, and automatic schedules produce the same
   admitted value and gradient on the Apple dataset within the existing
   integration thresholds (prefer bit identity and report it);
6. the public call
   `mlest(datasets.apple, method="direct", backend="cpu", solver="dveb")`
   converges;
7. its log-likelihood, mean, and covariance pass the already committed
   `apple_reference.json` thresholds: `1e-7` absolute log-likelihood and
   `1e-3` relative mean/covariance;
8. the result reports the DVEB backend, ABI, and installed artifact hash;
9. caller-owned input bytes remain unchanged;
10. a second dataset (`datasets.missvals`) returns finite results;
11. invalid platform/ISA, missing artifact, and corrupted-artifact probes
    refuse clearly before numerical execution and do not silently fall back.

The complete branch unit/integration suite must pass after implementation.
Tests requiring actual CUDA hardware are not part of the wheel qualification,
but existing non-CUDA regressions may not be weakened or deleted.

## 9. Evidence and reporting

Committed evidence must include:

- builder image digest and tool versions;
- source and generated-input hashes;
- exact artifact and wheel build commands;
- pre-repair and final artifact hashes;
- final wheel hashes and sizes;
- `auditwheel show`, ELF dependency, and symbol-version reports;
- environment identities and installed package inventories;
- machine-readable results for every static and runtime check;
- failures, exclusions, and deviations;
- an explicit GO/NO-GO decision under section 1;
- remaining limitations and exact nonclaims.

Build products need not be committed if they are reproducible and their
hashes are recorded. The generated source snapshot, build scripts, manifest,
checker, and evidence are committed. The worktree must finish clean. No push
is authorized by this protocol.

## 10. Interpretation limits

A GO proves only that this research branch can ship its admitted DVEB
MVN-MLE CPU backend in a bounded, self-contained Linux wheel without making
PyTorch or CUDA part of the installation. It does not prove universal Linux
portability, does not make DVEB a wholesale PyTorch replacement, and does not
authorize release. A future product decision can choose a broader or narrower
support envelope using this evidence.
