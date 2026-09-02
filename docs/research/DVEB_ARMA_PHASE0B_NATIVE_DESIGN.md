# Exact-ARMA Phase-0B native-diagnostic design freeze

**Date:** 2026-09-02

**Status:** FROZEN BEFORE NATIVE IMPLEMENTATION OR PERFORMANCE OBSERVATIONS

**Authority:** `DVEB_ARMA_PHASE0B_PROTOCOL.md` plus the separate native-
diagnostic authorization recorded there

## Purpose and boundary

NC and NG answer one question only: does the frozen exact batched ARMA
recurrence have useful native CPU or CUDA headroom over competent existing
implementations? They are separately authored diagnostics, not PyStatistics
backends and not DVEB output. Their source may not be copied into DVEB or used
to claim a language result.

The authorship direction is mathematical contract to native diagnostic. The
author has necessarily seen the existing NumPy/Cython and ordinary-PyTorch
implementations while establishing the oracle; that cannot be undone and this
is not described as clean-room independence. No DVEB-generated ARMA source
exists, and no DVEB source or generated artifact will be inspected or changed
during this campaign.

## Fixed ABI and layouts

The diagnostic is a research shared library with a C ABI. Every entry point
returns zero on success and a nonzero documented refusal/error code otherwise.
There is no CPU fallback from CUDA and no shape coercion.

Inputs are caller-owned, writable or read-only C-contiguous float64 arrays:

- observations `z[K,n]`;
- AR coefficients `phi[K,r]`;
- innovation loading `loading[K,r]`;
- an L3 proposal trace `phi_trace[100,K,r]` and
  `loading_trace[100,K,r]`.

Outputs are caller-owned `nll[K]`, `sigma2[K]`, and byte-valued `status[K]`.
Native code never mutates inputs. Supported frozen limits are `K>=1`, `n>=1`,
`1<=r<=25`, and exactly 100 L3 proposals. Unsupported shapes refuse.

CPU contexts own reusable scratch sized by maximum OpenMP threads, not by
series count. CUDA contexts own device copies of inputs, outputs, and proposal
traces. L1 begins after resident upload and ends after device completion; L2
includes upload, execution, synchronization, and output download. L3 keeps `z`
and the complete committed proposal trace resident, invokes one likelihood per
proposal through the same public ABI, and materializes all 100 result vectors
after the region. This preserves 100 distinct call boundaries for both native
and PyTorch paths rather than hiding the trace in one special fused kernel.

## NC: CPU diagnostic

NC uses C++17 plus OpenMP and parallelizes the independent `K` dimension with
static scheduling. Each OpenMP worker owns one reusable aligned workspace.
The recurrence within a series remains sequential in time and retains the
contract's scalar accumulation order. Matrix work uses explicit loops, not a
BLAS library; this avoids changing the estimator through an uncontrolled
reduction order.

Forced schedules are `{1,6,12}` OpenMP threads. Every schedule is qualified and
reported. The calibration cells may select a deterministic per-cell best
thread count for descriptive automatic scheduling, but the decision campaign
reports all three and uses the fastest admitted existing/native result per
cell without fitting a hidden selector.

Compile policy: `g++ 13.3`, C++17, `-O3`, `-march=native`, OpenMP,
`-ffp-contract=off`, no fast-math. The portable PyStatistics artifact is not
being evaluated here; NC is intentionally a Forge-native headroom diagnostic.

## NG: CUDA diagnostic

NG assigns one CUDA block to each independent series. Threads cooperate over
the `r*r` covariance elements while the observation-time recurrence remains
sequential. Stationary doubling and filtering use dynamic shared storage for
the live matrices and state vectors. No per-series global scratch is read in
the numerical loop, and no device-side allocation occurs.

Forced block sizes are `{32,64,128,256}`. All are qualified first. Calibration
selects one frozen block size per state dimension `r` by the geometric mean of
L1 medians on C01--C04 cells having that `r`; when only one calibration cell
exists for an `r`, that cell alone selects it. Ties within 1% choose the smaller
block. Evaluation cells never influence selection. The mapping is persisted
before decision timing.

Each likelihood invocation is one kernel launch. No persistent kernel, CUDA
Graph, custom PTX, tensor core, mixed precision, or cross-proposal fusion is
allowed. These exclusions make NG a straightforward native diagnostic rather
than an unlimited hand-tuning project.

Compile policy: CUDA 13.0 update 2, C++17, `-O3`, `sm_120`, IEEE float64,
`--fmad=true`, no fast-math. CUDA properties are queried once at context
creation and recorded. A block-size override must be obeyed or refused.

## Existing comparators

- C1 is the existing serial Cython workspace path.
- C2 is a persistent Python thread-pool wrapper around independent existing
  Cython workspaces. The Cython numerical core releases the GIL; threads
  `{1,6,12}` are retained and reported. It changes scheduling, not arithmetic.
- T1 is the admitted eager public-op PyTorch formulation.
- T2 is retained as a compile refusal because its literal Python time loop did
  not finish full-graph compilation at C01 within three minutes.
- T3 is the admitted public `torch.while_loop` formulation compiled with
  `torch.compile(fullgraph=True)`. Its prototype API status remains prominent.

CPU comparisons are NC versus the fastest admitted C1/C2/T1/T3 CPU lane at
the same cell and endpoint. CUDA comparisons are NG versus the fastest admitted
T1/T3 CUDA lane. Results are not pooled across devices.

## L3 proposal trace

The 100 proposals are a deterministic, committed perturbation of each frozen
cell's base coefficients. For proposal `j`, all series receive a bounded
coefficient-wise perturbation derived from the master seed, cell ID, proposal
index, series index, and coefficient index. AR perturbations are rejected and
redrawn until every series has minimum root modulus at least `1.0001`.
Loading-vector perturbations are bounded independently. Proposal zero is the
unmodified base point.

The generator commits every trace hash and root minimum before calibration or
timing. The same trace bytes are consumed by all implementations. There is no
optimizer and no adaptive proposal generation.

## Correctness and tolerance freeze

Before timing, NC and every forced NG block size run all 16 frozen cells and
all correctness-only edge cases. NC must be bit-identical to Cython for NLL,
sigma2, and status; failure stops rather than relaxing that requirement. NG
uses the protocol's independently derived relative bounds and exact status.

Tolerance derivation uses only newly generated correctness-only inputs absent
from C01--C04 and E01--E12. The derived NLL and sigma2 bounds, every source and
binary hash, environment identity, proposal-trace hashes, calibration results,
and a zero-decision-observation declaration are committed before campaign
timing.

## Timing and decision

The protocol's 10 warmups, 30 observations, randomized paired order, fixed
affinity, thermal exclusions, 100 ms region batching, and 10,000-resample
paired bootstrap apply unchanged. L1--L4 remain separate. Calibration results
cannot count as decision evidence.

Because T1 may be impractically slow and T2 has already refused, calibration
may mark a lane infeasible only after one bounded pilot exceeds a projected
eight-hour total for that lane. The observed pilot and projection are retained;
the lane is not silently omitted. T3 and the relevant native diagnostic remain
load-bearing.

The mechanical Case-1/Case-2 rule is exactly section 12.1 of the protocol. No
threshold, grid cell, proposal, implementation source, or selected CUDA block
may change after evaluation results are visible.

## Intended files

- `benchmarks/dveb_arma_phase0b/native/arma_native.h`;
- `benchmarks/dveb_arma_phase0b/native/arma_cpu.cpp`;
- `benchmarks/dveb_arma_phase0b/native/arma_cuda.cu`;
- build/loading, proposal-freeze, admission, calibration, campaign, analysis,
  and verification scripts under `benchmarks/dveb_arma_phase0b/`;
- source/binary freezes and evidence under
  `docs/research/evidence/dveb_arma_phase0b_native/`.

Generated shared libraries and profiler scratch outside the evidence directory
remain ignored. Exact admitted binaries are force-added to the evidence record.
