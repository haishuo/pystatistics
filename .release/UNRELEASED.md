# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Apple-Silicon bootstrap of the mean now runs on ALLOY-generated Metal.**
  `boot(data, statistic, backend='gpu', gpu_statistic='mean')` on an
  Apple-Silicon machine routes the eligible design (ordinary method,
  `statistic_type='index'`, no strata, 1-D data) to a packaged ALLOY artifact
  instead of PyTorch. `backend_name` reports `gpu_mps_alloy_bootstrap`, and
  `result.info` carries `implementation='alloy'` with the ALLOY commit the
  binaries were built from. CUDA is unchanged and still uses PyTorch; the CPU
  path is untouched.

  What this buys a user: the complete `boot(...)` call is 1.10-1.72x faster
  than the PyTorch/MPS path across n=100..10,000 and 2,000..100,000 resamples
  (medians of 10 timed runs in alternating order,
  `benchmarks/montecarlo/bench_alloy_bootstrap.py`, Apple M2 Max). A fixed seed
  now reproduces the replicate means *exactly* on repeated runs, and the drawn
  indices are identical on the CPU and the GPU — the PyTorch path documents its
  GPU stream as statistically equivalent to the CPU's but not identical.

  Unsupported configurations are unchanged: a declared statistic that is not
  the mean still raises, and balanced/parametric bootstrap, frequency and
  weight statistics, strata and 2-D data are still refused on the GPU.

- **`backend='gpu'` no longer requires PyTorch on Apple Silicon** for the
  eligible bootstrap-of-the-mean. That request is now decided before the shared
  backend resolver, which reaches for PyTorch only to learn that a Metal device
  exists -- something ALLOY's own runtime reports directly. The resolver itself
  is unchanged, so `backend='auto'` still resolves to the CPU on Apple Silicon,
  CUDA still routes through PyTorch, and every refusal for an unsupported
  bootstrap configuration is untouched. If the packaged ALLOY backend cannot
  run, the call fails with that reason rather than quietly using PyTorch or the
  CPU.

- **New package data: a vendored ALLOY runtime and bootstrap bundle** under
  `pystatistics/montecarlo/backends/_alloy/artifacts/` (macOS arm64 only,
  ~348 KB). They are loaded from the installed package and from nowhere else —
  no environment variable, no source-tree search, no system location — and
  every file's SHA-256 is verified before use, so a partial or corrupted
  install fails by name rather than crashing in a dynamic loader.
  `PROVENANCE.md` beside them records the ALLOY commit, compiler and
  bundle-format versions, source and binary digests, and the rebuild command.
  On any other platform the loader raises with the reason and nothing else is
  affected.

  **PyStatistics remains MIT-licensed. The vendored ALLOY runtime and
  generated artifacts are separately identified and distributed under
  Apache-2.0**, with the notice Apache-2.0 requires shipped beside them as
  `LICENSE-ALLOY`. The MIT licence at the repository root does not extend to
  those files, and their terms do not extend to anything else.
