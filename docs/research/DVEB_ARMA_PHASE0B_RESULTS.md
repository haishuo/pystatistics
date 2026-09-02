# DVEB batched exact-ARMA Phase-0B result

**Date:** 2026-09-02

**Binary outcome:** **CASE 2 — useful native headroom exists; DVEB does not yet
express the required general CUDA execution model.**

This is not a DVEB performance GO. It authorizes consideration of a bounded
general compiler trunk, followed by a separately frozen DVEB-inclusive
non-domination campaign. No DVEB source was read during native authorship, no
DVEB compiler or runtime was changed, and no PyStatistics API/backend changed.

## Decision result: L3 100-proposal fit trace

Each entry is the median time for the same committed sequence of 100 exact
likelihood proposals. `T3` is ordinary public-op PyTorch 2.13 CUDA using
`torch.while_loop` and `torch.compile(fullgraph=True)`. `NG` is the separately
authored exact CUDA diagnostic. Ratios are `T3 / NG`; values above one favor
native code.

| Cell | `(K,n,r)` | T3 (s) | NG (s) | T3 / NG | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| E04 | `(64,2000,3)` | 6.776748 | 0.418738 | 16.184x | [16.166, 16.194] |
| E05 | `(256,500,13)` | 1.841773 | 0.185244 | 9.942x | [9.927, 9.954] |
| E06 | `(256,2000,13)` | 7.082906 | 0.688514 | 10.287x | [10.274, 10.300] |
| E07 | `(1024,2000,3)` | 9.312872 | 1.188788 | 7.834x | [7.829, 7.839] |
| E08 | `(1024,500,13)` | 2.551121 | 0.649549 | 3.928x | [3.923, 3.932] |
| E09 | `(64,10000,3)` | 34.835580 | 2.088499 | 16.680x | [16.663, 16.690] |
| E11 | `(256,500,25)` | 2.369842 | 0.623729 | 3.799x | [3.798, 3.802] |
| E12 | `(1024,120,1)` | 0.318288 | 0.064400 | 4.942x | [4.937, 4.950] |

The frozen Case-2 threshold required at least 1.5x in six of eight primary
cells, every winning interval above 1.0, and no cell below 0.80x. The result is
**eight of eight qualifying wins**, no floor failure, and a geometric-mean
speedup of **7.950x**. The classification is therefore mechanical, not a
post-result judgment.

## Descriptive endpoint checks

The ancillary L1 resident and L2 transfer-inclusive results use all twelve
evaluation cells. Both implementations perform the same shared repeat count;
10 warmups and 30 randomized paired observations remain. A duration-only cap
of eight repeats was frozen after two fully invalidated planning attempts. The
cap changes no decision evidence.

| Cell | `(K,n,r)` | L1 T3 / NG | L2 T3 / NG |
|---|---:|---:|---:|
| E01 | `(1,120,1)` | 13.335x | 11.989x |
| E02 | `(1,2000,13)` | 16.179x | 15.978x |
| E03 | `(16,500,3)` | 16.399x | 15.651x |
| E04 | `(64,2000,3)` | 16.296x | 15.830x |
| E05 | `(256,500,13)` | 10.169x | 9.591x |
| E06 | `(256,2000,13)` | 10.769x | 10.340x |
| E07 | `(1024,2000,3)` | 5.909x | 5.630x |
| E08 | `(1024,500,13)` | 3.947x | 3.822x |
| E09 | `(64,10000,3)` | 16.455x | 16.068x |
| E10 | `(16,2000,25)` | 13.012x | 12.953x |
| E11 | `(256,500,25)` | 4.429x | 4.407x |
| E12 | `(1024,120,1)` | 4.802x | 4.294x |

Transfers therefore do not erase the observed native headroom. L4 tells the
same story at E07: median fresh-process launch-to-answer is **4.878 s for T3**
and **0.316 s for NG**, a **15.427x** ratio. Median peak RSS is 1,026,376 KiB
for T3 and 629,400 KiB for NG (1.631x). L4 includes import/load, input creation
and device upload, ordinary compile-cache recovery/specialization, one exact
likelihood, host materialization, serialization, and process shutdown.

## Correctness and artifact qualification

Before timing, NC at threads `{1,6,12}` and NG at blocks `{32,64,128,256}`
passed all 16 frozen cells, persistent/diffuse/invalid cases, eight refusal
cases, deterministic repeat and immutability gates, and auto/forced schedule
agreement. The selected CUDA mapping, frozen on disjoint calibration cells,
is `r=1 -> 32`, `r=3 -> 32`, `r=13 -> 256`, and `r=25 -> 256`.

All **1,980 stored timed observations** pass finite-output, exact-status, and
independent-compiler tolerance checks. Worst relative differences across L3
are `4.11e-15` for NLL and `1.16e-14` for sigma2. The derived prospective
relative floor is `1e-14`, combined with the frozen scale-aware term.

The exact frozen PyStatistics time-series regression was rerun after the
campaign: **153 passed, 0 failed, 0 skipped**, with one existing PyTorch API
deprecation warning.

The CPU diagnostic is 25,432 bytes and built in 0.347 s. The CUDA diagnostic
is 1,059,352 bytes and built in 0.656 s for `sm_120`; its one likelihood kernel
uses 39 registers/thread, no local-memory spill, 1,072 bytes static shared
memory plus shape-dependent dynamic shared memory. These are diagnostic
artifacts, not distributable PyStatistics backends.

## What the result does and does not attribute

PyTorch successfully represented the exact recurrence: one captured graph,
zero graph breaks, and two public structured while loops. Thus this is not an
expressibility failure. The separately authored CUDA path assigns one block to
each independent series, keeps the sequential time recurrence and local
matrix state inside one kernel, cooperates across each `r x r` matrix in
shared memory, and launches once per likelihood. That execution shape is
consistent with the measured gap.

The campaign did not collect a kernel-launch trace for T3, so it does not claim
a fully isolated microarchitectural cause. It establishes the performance
fact and a credible native construction, not that every nanosecond has been
attributed. A later profiler pass may explain the framework-generated path but
cannot revise this frozen Case-2 decision.

## General DVEB capability diagnosis

The current DVEB implementation has relevant pieces but not their required
combination:

- the scalar/stencil CUDA frontend requires one straight-line `foreach` body,
  f64 scalar parameters, and statically offset field access;
- the dense-objective frontend supports integer indices, nested loops,
  reductions, and shared/per-worker scratch, but emits CPU-only artifacts and
  has no conditional/early-exit statement in its bounded body language;
- existing portable ABIs already demonstrate explicit residency, reusable
  CUDA contexts, deterministic placement, and forced launch controls.

A general next trunk would therefore need a CUDA-capable stateful dense or
batched-recurrence execution model—not an `arma` recognizer. The minimum
mechanisms indicated by NG are:

1. an outer parallel map over independent work items with a sequential,
   loop-carried inner recurrence;
2. bounded per-work-item matrices/vectors indexed by runtime loop variables;
3. block-scoped shared scratch, cooperative loops, and barriers;
4. bounded conditional termination and explicit status propagation;
5. deterministic one-block-per-work-item CUDA lowering with legal forced block
   controls and a calibration-backed automatic block policy; and
6. a reusable device-resident context able to update proposal parameters and
   materialize results without hidden allocation or fallback.

These are general compiler/runtime features useful to state-space filters,
dynamic programs, small-matrix recurrences, and batched simulations. ARMA
mathematics must remain in a DVEB example/consumer layer. Existing GradFlow
and dense-CPU behavior remains an immutable regression surface.

## Scope boundary and next decision

Phase-0B stops here at Case 2. No additional CPU performance campaign was run
after the CUDA diagnostic mechanically satisfied the stopping rule; NC remains
a fully admitted correctness/artifact diagnostic, not performance evidence.
No DVEB compiler work is authorized by this result alone.

The scientifically valid next step is a separately approved DVEB trunk with a
prospective language/compiler protocol, general non-ARMA proof programs, full
GradFlow and dense-CPU regression, and a new DVEB-inclusive campaign applying
section 12.2's non-domination rule. Only that later campaign can award a DVEB
ARMA suitability GO.

## Evidence

- protocol: `docs/research/DVEB_ARMA_PHASE0B_PROTOCOL.md`;
- native design: `docs/research/DVEB_ARMA_PHASE0B_NATIVE_DESIGN.md`;
- native admission/calibration: `docs/research/evidence/dveb_arma_phase0b_native/`;
- raw and analyzed campaign: `docs/research/evidence/dveb_arma_phase0b_campaign/`;
- implementation/harness history: `benchmarks/dveb_arma_phase0b/`;
- offline verifier: `benchmarks/dveb_arma_phase0b/verify_campaign.py`.

Run from the repository root:

```bash
python3 benchmarks/dveb_arma_phase0b/verify_campaign.py
```
