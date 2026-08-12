> Non-binding, hard-won engineering knowledge about GPU behavior across CUDA
> and MPS. The **binding** API/backend rules live in
> `pystatistics/CONVENTIONS.md`; this file is notes, not law.

# GPU Backend Notes

Hard-won knowledge about GPU behavior across CUDA and MPS backends.
Written because people won't know this unless they hit the specific
issue in the specific way.

---

## CUDA vs MPS: When Metal Falls Off a Cliff

### The Short Version

NVIDIA CUDA and Apple MPS are not interchangeable GPU backends. Certain
memory access patterns that are fast on CUDA are **catastrophically slow**
on MPS — not 2x slower, but 1000x slower. If your algorithm uses
`scatter_add_` with sparse, irregular bucket IDs, MPS is the wrong backend.

The pystatistics and pystatsbio libraries detect this and fail fast rather
than silently delivering a 15-minute wait.

---

### The Pattern That Breaks MPS: `scatter_add_` with Sparse Targets

**What the code needs to do** (example: midrank computation for batch AUC
with 5,000 biomarkers × 500 patients):

1. Sort each column — MPS is fine at this.
2. Find groups of tied values — MPS fine.
3. **For each tie group, sum up the ranks of its members and count them**
   — this is `scatter_add_`.
4. Divide to get the average rank per group — MPS fine.

Step 3 is the problem. `scatter_add_` says: "I have 2.5 million items.
Each one has a bucket ID. Add each item's value into its bucket." The
bucket IDs are sparse and irregular — each column has different tie
patterns.

### Why CUDA Handles This Fine

NVIDIA GPUs have **atomic operations in shared memory**. When 1,000
threads simultaneously try to add values to the same bucket, CUDA
serializes them at the hardware level using atomic compare-and-swap — a
few clock cycles per conflict. The GPU's memory controller is designed
for this pattern.

**Measured**: ~0.05 ms per `scatter_add_` call on RTX 5070 Ti.

### Why MPS Is Catastrophically Slow

Apple's Metal GPU architecture was designed for **graphics** — rendering
pixels, texture mapping, vertex shading. These workloads are *regular*:
every pixel does the same work, reads from the same textures, writes to
a predictable framebuffer location.

`scatter_add_` is the opposite: thousands of threads writing to *random,
unpredictable* memory locations with *read-modify-write* semantics.
Metal has to:

1. **Serialize conflicting writes** — Metal's atomic support is weaker
   than CUDA's. When two threads hit the same bucket, one stalls.
2. **The memory access pattern defeats the cache** — buckets are spread
   across a 2.5M-element array. Each access is essentially a cache miss.
3. **Metal's command encoding adds overhead** — Metal batches GPU
   commands through a command buffer; every scatter operation requires a
   full encode-dispatch-wait cycle, unlike CUDA's inline kernel execution.

**Measured**: ~150 ms per `scatter_add_` call on M2 Max — **3,000x
slower** than the same operation on CUDA.

### The Cascading Effect

The midrank algorithm calls `scatter_add_` 3 times (sum ranks, count
members, map back) inside `_midranks_vectorized`, which is itself called
3 times (pooled ranks, case within-ranks, control within-ranks). The
flattening trick that makes CUDA fast (process all 5,000 columns in one
scatter) makes MPS *worse* because it creates a single huge sparse array.

**Result**: Batch AUC for 5,000 markers takes ~0.015s on CUDA, ~20s on
MPS, vs ~0.9s on CPU. MPS is 22x slower than CPU.

> **Update (2026-08-12, torch 2.13.0):** this section's numbers are from the
> torch 2.10/2.11 MPSGraph-routed `scatter_add_`. PyTorch 2.13 (July 2026)
> replaced scatter/gather with hand-written native Metal kernels, and the
> re-measurement (M2 Max vs RTX 5070 Ti, 2.5M items, sparse random buckets)
> gives **0.45 ms/call on MPS vs 0.060 ms on CUDA — 7.6x, not 3,000x — and
> ~8x FASTER than CPU `np.add.at` (3.9 ms)**. The end-to-end batch-AUC
> conclusion above (and pystatsbio's `backend='gpu'`-on-MPS refusal built on
> it) is based on a number that no longer exists.
> Full re-measurement table: `docs/ACCELERATION_AUDIT_2026-08.md` §1.4.
>
> **Confirmed end-to-end (2026-08-12, Powerhouse M2 Max, torch 2.13.0):**
> batch AUC 5,000 markers × 500 samples = **0.012 s on MPS vs 0.67 s CPU
> (~55x)**, and 2.6–67x faster than CPU across the full historical grid
> (100–20,000 markers × 500–1,155 samples), passing the `GPU_FP32` check
> against the CPU fp64 reference in every cell. pystatsbio has lifted the
> refusal, version-gated on torch >= 2.13 (`_mps_native_kernels()` in
> `pystatsbio/diagnostic/_batch.py`), and `'auto'` now picks MPS there.
> Forensic caveat: on this machine/macOS 26.5 the workload is equally fast
> on torch 2.9–2.12 (35–38 ms warm), so the old ~20 s pathology lived in
> the April 2026 OS/MPSGraph stack, not the torch version alone — and
> since "Mainframe" was this same physical machine (Powerhouse's original
> name), hardware is a controlled variable: the OS/framework stack is the
> only thing that changed. The >= 2.13 gate is kept because it is the
> first torch whose scatter speed is guaranteed by bundled native kernels
> rather than the host OS. Grid table: pystatsbio
> `docs/GPU_BACKEND_NOTES.md`.

---

## What We Did About It

### pystatsbio `batch_auc` (updated 2026-08-12)

- `backend='gpu'` on CUDA: uses the vectorized `scatter_add_` kernel.
  49-63x faster than CPU.
- `backend='gpu'` on MPS with torch >= 2.13: supported — 2.6–67x faster
  than CPU across the benchmark grid (see the update note above).
- `backend='gpu'` on MPS with torch < 2.13: **raises `RuntimeError`**
  with an actionable message (upgrade torch, or `backend='cpu'`). Fail
  fast, fail loud (Coding Bible Rule 1).
- `backend='auto'` on MPS: picks MPS (float32) on torch >= 2.13, CPU
  below — auto means "best available", and the version gate decides
  which one that is.

### Could It Be Fixed for MPS?

Yes, but it requires a **completely different algorithm**:

- **Don't scatter.** Use `torch.unique_consecutive` on the sorted data
  to group ties *in-place* without random memory access. Then use
  `cumsum` to compute group sizes and rank sums. All operations are
  sequential/streaming, which Metal handles well.
- **Or process columns one at a time** — 5,000 iterations of a
  500-element sort + midrank is fast even in Python because each
  operation is small and cache-friendly.
- **Or use the CPU.** `scipy.stats.rankdata` is Cython running in L1
  cache on a single core. For 500 elements it takes ~0.13ms. Even
  looping 5,000 times: 650ms. The CPU wins not because it's "faster"
  than the GPU, but because the problem isn't parallel enough to
  justify GPU dispatch overhead for this workload shape.

The fundamental issue isn't Apple Silicon's compute power — it's that
Metal's programming model doesn't expose the low-level atomic memory
operations that make scatter patterns efficient on CUDA. This is a
deliberate Apple design choice (simpler programming model, optimized for
graphics/ML inference, not scientific computing). It may improve in
future Metal versions, but today, if your algorithm requires
`scatter_add_` into sparse targets, MPS is the wrong backend.

---

## General Rules for GPU Backend Selection

Based on validation across Mac Studio M2 Max and Linux RTX 5070 Ti:

### Operations That Are Fast on Both CUDA and MPS

- Matrix multiply (`X.T @ X`, `X @ beta`)
- Cholesky decomposition and triangular solves
- Element-wise operations (add, multiply, exp, log)
- Reductions (sum, mean, max along a dimension)
- `argsort` (used for ranking)
- `torch.rand`, `torch.randint` (random number generation)

### Operations That Are Fast on CUDA but Slow on MPS

- **`scatter_add_` with sparse/irregular indices** — the specific killer
- `scatter_` in general with non-contiguous write patterns
- Any operation that requires atomic read-modify-write to random locations
- Operations that create very large intermediate tensors with irregular
  access patterns

### When GPU Wins Over CPU

- **Large matrix operations**: n × p with n > 10,000 and p > 50.
  The GPU amortizes transfer overhead. Below this, CPU is often faster.
- **Embarrassingly parallel tasks**: R=50,000 permutation tests where
  each permutation is independent. GPU computes all R at once.
- **Batch operations**: fitting 5,000 dose-response curves simultaneously,
  computing AUC for 20,000 genes at once.

### When CPU Wins Over GPU

- **Small problems**: n < 1,000. GPU launch overhead dominates.
- **Sequential algorithms**: iterative methods where each step depends
  on the previous (e.g., Cox PH Newton-Raphson with few iterations).
- **Sparse scatter patterns on MPS**: see above.
- **User-supplied Python callbacks**: bootstrap/permutation with arbitrary
  statistic functions — the function runs on CPU regardless.

---

## Benchmark Reference

All benchmarks measured on Forge (RTX 5070 Ti, CUDA 12.0) and Mainframe
(Mac Studio M2 Max — the same machine since renamed Powerhouse) during
the April 2026 Linux/NVIDIA validation. Historical note: these numbers
were long mis-attributed to an "M2 Ultra"; no such machine ever existed
in the fleet.

### Regression (pystatistics)

| Problem | CPU | GPU (CUDA) | Speedup |
|---------|-----|------------|---------|
| OLS 500K × 200 | 5.4s | 0.13s | **42x** |
| OLS 1M × 100 | 5.4s | 0.12s | **44x** |
| GLM binomial 50K × 100 | 0.4s | 0.08s | **5x** |

### Batch AUC (pystatsbio)

| Markers × Samples | CPU | GPU (CUDA) | GPU (MPS) | CUDA Speedup |
|--------------------|-----|------------|-----------|--------------|
| 100 × 1,155 | 0.018s | 0.9s | N/A | 0.02x (CPU wins) |
| 1,000 × 1,155 | 0.18s | 0.003s | N/A | **63x** |
| 20,000 × 1,155 | 3.6s | 0.074s | N/A | **49x** |

### Permutation Test (pystatistics)

| n samples | R perms | CPU | GPU (CUDA) | Speedup |
|-----------|---------|-----|------------|---------|
| 1,000 | 50,000 | 1.4s | 0.28s | **5x** |
| 10,000 | 50,000 | 6.7s | 0.29s | **23x** |
| 50,000 | 50,000 | 33s | 1.4s | **23x** |

---

## The MPS small-n floor (why MPS can't fully match CUDA there)

Established while optimizing MICE (per-op MPS-vs-CUDA timing + controlled
in-sweep runs). Two distinct causes, both defensible:

1. **A few torch ops have pathologically slow MPS kernels** —
   `solve_triangular` ~250x slower than CUDA, `searchsorted` ~1136x (n=20k),
   `cholesky_solve` / `eigh` unimplemented. These ARE recoverable by
   rebuilding from MPS-fast primitives (matmul / cholesky / sort): MICE
   replaced `searchsorted`→merge-rank and `solve_triangular`→matmul-series
   inverse, closing most of the *mid/large-n* gap to ~the raw FP32 silicon
   ratio (~3-4x).
2. **Per-op dispatch overhead is the small-n floor** — but the "~0.5-1 ms/op,
   NOT recoverable, no graph capture" characterization below is **obsolete as
   of torch 2.12/2.13** (re-measured 2026-08-12; the original text is kept
   for the record, struck through in spirit):

   > *Original (torch 2.10/2.11):* In a sequential per-step sweep, MPS pays
   > ~0.5-1ms of command-encode overhead *per op*, and — unlike CUDA — has no
   > graph-capture to amortize it… This is an intrinsic platform limitation.

   **What the ~0.5-1 ms actually was:** a per-op *command-buffer commit +
   CPU-GPU sync* regime (MPSGraph per-op scheduling, plus our own per-step
   `.item()` syncs before 3.14.0) — not a Metal hardware floor. The measured
   decomposition (2026-08-12): encoding one more dispatch into an open command
   buffer costs ~1-6 µs; a PyTorch native-Metal-shader op dispatches in
   ~1.2-1.4 µs; an MPSGraph-routed op ~20 µs; the expensive event is the
   command-buffer **commit + sync** at ~0.1-1 ms — which is what per-op-synced
   sweeps paid on every op.

   **Measured now (M2 Max):** torch 2.12/2.13 eager elementwise chains run
   at **~4.2-4.5 µs/op** (parity with CUDA eager at 3.3 µs), and
   **`torch.compile` works on MPS** (Inductor-Metal), fusing a 1000-op chain
   to **~0.023 µs/op** — the graph-capture-equivalent that "does not exist"
   now ships in-framework. PyTorch 2.13 also migrated scatter/gather, cumsum,
   sort, reductions, RNG, and copy/cast off MPSGraph onto hand-written Metal
   kernels. The remaining levers, in cost order: (1) re-audit which ops still
   route through MPSGraph, (2) `torch.compile` the sweep step, (3)
   `torch.mps.compile_shader` for hand-fused kernel chains (one command
   buffer per solver iteration). The *conclusion* of this section — reduce op
   count per step / fuse — was correct; the floor it fights is ~100-200x
   lower than documented, and the empirically-tuned crossovers calibrated
   against the old floor (e.g. `_SERIES_INV_MIN_NOBS = 3000`) need re-tuning.
   See `docs/ACCELERATION_AUDIT_2026-08.md` for the full audit.

### MPS dense factor-and-solve: fast vs slow/absent

Apple MPS executes batched **matmul** and **cholesky** fast, but its small
dense **factor-and-solve** kernels are slow or absent:

| op | torch 2.10/2.11 (original) | torch 2.13.0 (re-measured 2026-08-12, M2 Max) |
|---|---|---|
| `matmul`, `cholesky_ex`, `sort`, `cumsum`, `gather` | fast | fast (confirmed; cumsum 2.5M = 0.10 ms, cholesky_ex B=1000 p=20 = 0.23 ms) |
| `scatter_add_` (sparse targets) | ~150 ms / 2.5M items (~3000× CUDA) | **0.45 ms (7.6× CUDA; 8× faster than CPU) — FIXED** |
| `searchsorted` | ~1136× CUDA at n=20k | **37 µs at n=20k (7.4× CUDA; 38× faster than CPU) — FIXED** |
| `solve_triangular` | slow (~4 ms for a 20×20 batch, n-independent) | still slow: 3.8 ms @ B=100 p=20 → 352 ms @ B=10⁴ (12-190× the cholesky cost) — **workaround stays** |
| `linalg.solve`, `linalg.inv` | slow (~100-300× matmul) | still slow: ~465 ms @ B=1000 p=20 (~3300× matmul) — **workaround stays** |
| `pinv` | slow | 29.6 ms via **CPU-fallback SVD** (now emits an explicit UserWarning) |
| `cholesky_solve` | **unimplemented** (errors on MPS) | implemented but **~1300× CUDA** (66 ms @ B=1000 p=20) — worse than failing loud; **workaround stays** |
| `linalg.eigh` | **unimplemented** (errors on MPS) | still unimplemented |
| `linalg.lstsq` | unsupported (CPU detour) | still unimplemented (CPU detour stays) |

### The in-house remedy

`pystatistics/mvnmle/_objectives/_batched_cholesky.py` solves the
factor-and-solve gap with a matmul-only path:

- `_tri_inv_blocked(torch, L)` — matmul-only block-recursive inverse of a
  batched lower-triangular factor. No `solve_triangular` / `inv` kernel
  touched.
- `_use_blocked(L, method)` — device dispatch: matmul-inverse on MPS,
  `solve_triangular` on CUDA/CPU (fast there). One shared path, one device
  bridge.

Several modules with `linalg.solve` / `cholesky_solve` / `eigh` / `inv` on an
MPS hot path can be reformulated the same way (chol + matmul-inverse, or
SVD-based PCA in place of `eigh`). Each such fix is its own task with R-fidelity
plus FP32 re-validation on **both** MPS and CUDA, since the shared-primitive
change touches both devices.
