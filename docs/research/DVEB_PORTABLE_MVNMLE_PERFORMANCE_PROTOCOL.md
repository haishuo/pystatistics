# DVEB portable-wheel MVN-MLE performance qualification

Status: **FROZEN BEFORE QUALIFICATION OR TIMING**

Date frozen: 2026-09-02

Branch: `research/dveb-pystatistics`

This protocol belongs only to the unpublished research branch. It does not
authorize a merge to `main`, a PyPI change, publication, or release. PyPI
6.1.5 and the paper-facing branch remain frozen.

## 1. Question

Does the exact CPython 3.11 CPU-only wheel admitted by CPU-wheel protocol v2
retain the MVN-MLE implementation's required performance property when used
as an installed artifact: correct forward-Cholesky results without material
end-to-end domination by the competent CPU-PyTorch implementation?

The previous 2.326x geometric-mean result belongs to the Forge-native artifact
`a96e4102...1fb102f`. It is permanent historical evidence, but it cannot be
transferred to the portable artifact. The formal comparison here is only:

1. installed portable DVEB wheel; and
2. the existing optimized CPU-PyTorch forward-Cholesky path.

The old native result is reported later only as a cross-campaign diagnostic.
It never enters this protocol's decision.

## 2. Frozen identities and installation boundary

- protocol pre-commit HEAD: `746e846`;
- inherited baseline: PyStatistics `v6.1.5`;
- research version: `6.2.0.dev0+dveb`;
- portable-wheel qualification: formal GO at `746e846`;
- exact CPython 3.11 wheel SHA-256:
  `92fa398c2785eac02d824c4bb589cabc2a9a966b4c87ab915a622df5b44caa4a`;
- wheel filename:
  `pystatistics-6.2.0.dev0+dveb-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`;
- repaired native payload SHA-256:
  `65438a0d742257acc0dc7bd2ff13d2017669fd21dc07f64add41b18fb229fe38`;
- ABI: dense CPU ABI v1;
- ISA and floating point: x86-64-v2, IEEE float64,
  `-ffp-contract=off`, no fast-math;
- wheel-local runtime: `libgomp-e985bcbb.so.1.0.0`;
- comparator environment:
  `/home/haishuo/miniconda3/envs/gradflow/bin/python`, Python 3.11.13,
  NumPy 2.2.6, SciPy 1.16.0, PyTorch
  `2.9.0.dev20250705+cu128` used on CPU;
- machine: AMD Ryzen 5 7600X, 6 physical cores / 12 logical CPUs, Forge
  Ubuntu 24.04, glibc 2.39.

The wheel is installed with `--no-deps --target` into a fresh ignored `/tmp`
site directory. Campaign workers run from outside the checkout with that
directory first on `PYTHONPATH`. Every worker refuses unless
`pystatistics.__file__` is below the installed site and the loaded artifact
hash is the frozen repaired-payload hash. The research checkout must not
satisfy package imports. No wheel rebuild, compiler invocation, or artifact
substitution is allowed during qualification or timing.

The PyTorch CUDA build's GPU libraries are irrelevant to the CPU numerical
path but are recorded when deployment footprint is described. Package size
is descriptive and must distinguish the active comparator environment from a
universal claim about every PyTorch distribution.

## 3. Correctness admission

Before timing, all 18 case/thread lanes run one untimed full fit through both
installed implementations. The gates are unchanged from the prior campaign:

- both fits converge;
- absolute log-likelihood difference is at most `1e-5`;
- `muhat` and `sigmahat` agree with `rtol=atol=1e-3`;
- the caller-owned input is byte-identical before and after both fits;
- the exact E1--E6 input hashes equal the prior corrected campaign's frozen
  hashes;
- the portable result reports ABI 1 and artifact hash
  `65438a0d...229fe38`;
- no fallback, compile, download, calibration, or device use occurs.

The already committed CPU-wheel v2 verifier, focused 18-test integration, and
full-suite baseline-relative gate must still pass before timing. A failed
candidate is not timed and thresholds are not relaxed after observation.

## 4. Evaluation grid and execution policy

Use the exact prior E1--E6 generators and bytes:

| Case | n | p | Missingness |
|---|---:|---:|---|
| E1 | 150 | 4 | MCAR 15%, seed 0 |
| E2 | 178 | 13 | MCAR 15%, seed 1 |
| E3 | 569 | 30 | MCAR 15%, seed 2 |
| E4 | 5000 | 13 | structured, at most 32 patterns, seed 3 |
| E5 | 5000 | 30 | structured, at most 64 patterns, seed 4 |
| E6 | 5000 | 30 | MCAR 15%, seed 5 |

Each runs at process affinities of 1, 6, and 12 logical CPUs. Inputs are
generated and hashed before PyTorch thread initialization.

For each of 18 lanes:

- one fresh worker process;
- three untimed full-fit warmups per implementation;
- 30 paired full user-visible `mlest` observations;
- within-pair order randomized with seed `0xD0EB1009`;
- `taskset` confinement and fixed OMP/MKL/OpenBLAS/PyTorch thread counts;
- no observation exclusion;
- one whole-lane rerun only for explicit interruption or process failure;
- medians decide; p05, p95, MAD, and 10,000-resample paired-bootstrap 95%
  intervals are descriptive.

Record complete fit time, timing sections, iterations/evaluations, final
state, schedule, artifact/ABI identity, input hashes, affinity, versions,
execution order, timestamps, and shared-process maximum RSS. Shared maximum
RSS is not attributed to either implementation.

## 5. Frozen decision

Define each lane's speedup as `PyTorch median / portable-DVEB median`.

**NO-GO / dominated** if either:

1. any lane has `PyTorch/DVEB <= 0.10`; or
2. PyTorch is faster in at least 12 of 18 lanes and the geometric mean of
   `DVEB/PyTorch` is at least `2.0`.

**GO / not dominated** if every correctness gate passes and neither domination
condition holds.

**STRONG GO** is a descriptive subclass of GO, frozen in advance: portable
DVEB is faster in at least 12 of 18 lanes and the geometric mean of
`PyTorch/DVEB` is at least `1.25`. Per-lane `PyTorch/DVEB >= 1.5` remains a
preferred, nonmandatory win.

No result-dependent compiler, artifact, schedule, optimizer, grid, repeat,
or threshold change is allowed. A weak GO is still a GO under DVEB's stated
“must not be dominated” requirement; it must not be reported as superiority.

## 6. Operational characterization

After the formal campaign, run 30 randomized fresh-process E1/T1 observations
for complete import-to-fit execution of each implementation. Compilation and
downloads are forbidden. Report active installed-package sizes, wheel size,
portable native payload and vendored-runtime sizes, and cold-path medians and
dispersion. These measurements explain deployment behavior and decide
nothing.

## 7. Evidence and interpretation limits

Commit the freeze manifest, untimed qualification, raw observations,
mechanical analysis, operational evidence, exact commands, failures, and a
binary report. Raw data is never replaced. The original native campaign and
CPU-wheel v1 NO-GO remain unchanged.

A GO establishes only that this exact portable x86-64-v2 wheel is suitable
for continued branch research on the admitted Linux/CPU boundary. It does not
make DVEB the default, remove PyTorch elsewhere, establish another platform,
prove universal Linux compatibility, authorize release, or establish a result
for CUDA, ALLOY, GradFlow, or another statistical workload.
