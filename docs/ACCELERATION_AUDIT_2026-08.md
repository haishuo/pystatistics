# PyStatistics Lower-Level Acceleration Viability Report

**Date:** 2026-08-12 (rev 2 — torch 2.13 re-benchmark folded in; upgrade-path recommendation added as §6)
**Scope:** Fact-finding only — no library code changes. Would Mojo, Triton, raw CUDA C++, or raw Metal capture efficiency gains that PyStatistics' current stack (NumPy/SciPy + PyTorch + Cython) is leaving on the table? And now: what is the recommended path to closing the measured gap?
**Evidence base:** 5 repo auditors (file:line evidence), 5 external researchers (primary sources), 4 adversarial verifiers, plus microbenchmarks run 2026-08-12 on Forge (RTX 5070 Ti, torch 2.11-dev cu128) and Powerhouse (Mac Studio M2 Max, torch 2.12 **and 2.13** MPS, MLX 0.32).

---

## 0. Executive summary

**The one-sentence answer:** there is real headroom, but almost none of it is where the question pointed — it is not "a lower-level *language*," it is (a) **kernel fusion / dispatch amortization on Apple silicon**, which has quietly become available *inside PyTorch itself*, and (b) **batched formulations** of workloads that currently have no batch axis in the API. Mojo, Triton, and raw CUDA each fail to buy anything the current stack cannot already reach — for reasons that are documented below with math and measurements.

**Corrections to the question's premises:**

1. **PyStatistics does not use CuPy anywhere.** The stack is NumPy/SciPy (fp64 CPU reference), PyTorch (GPU via CUDA fp32/fp64 and MPS fp32), and 17 native CPU kernels (Numba → Cython migration, **complete** as of 2026-07-22, numba removed from deps). Also for the record: CuPy is a *GPU* (CUDA/ROCm) library, not a CPU one — the CPU-side native layer here is Cython.
2. **"We couldn't find a good-enough accelerated architecture" is only true for specific modules** — and the repo preserves an unusually complete audit trail of exactly which formulations were tried and rejected (§4). Most modules *do* have GPU backends with 5–450× wins.
3. **The "did not have a speed advantage" conclusions split into three different kinds of wall** — one mathematical, one physical, one that was an artifact and has since **fallen** (§2). Only the artifact one is actionable.

**Report card at a glance:**

| Candidate attack | Verdict | Why (one line) |
|---|---|---|
| **Mojo** (CPU kernels) | ❌ Not now | Parity with Cython at best (verified, multiple 2026 benchmarks); no wheel-distribution story; no Windows; bit-identity unverified |
| **Mojo** (GPU kernels) | ❌ Not now | GPU APIs moved into proprietary `max` package (Mojo 1.0, Aug 2026); Apple tier is fp32-only + untested-in-CI; Qualcomm acquisition risk |
| **Triton** | ❌ | No Metal backend (upstream rejected one, Mar 2026); **no fp64 `tl.dot`** — disqualifying; Linux-only wheels; its launch overhead needs CUDA Graphs anyway |
| **Raw CUDA C++** | ❌ (niche ✓) | Everything it recovers (launch overhead, fusion, batching) is already reachable via torch.compile/CUDA Graphs/torch batched linalg; the niche is MAGMA-style p<16 register-tiled batched factorizations |
| **Raw Metal / custom kernels** | ✅ **Real headroom** | The "unrecoverable ~0.5–1 ms/op MPS dispatch floor" is **refuted** — it was a torch-2.10/2.11 MPSGraph+sync artifact; fused kernels get ~1–6 µs/op; `torch.mps.compile_shader` ships this with zero new deps |
| **torch.compile** (the null hypothesis) | ✅ **Cheapest win** | Measured today: 0.014 µs/op CUDA, **0.021 µs/op MPS** — a ~200× dispatch reduction on both platforms without leaving PyTorch |
| **Sequential recursions on GPU (any language)** | ❌ **Mathematically settled** | Measured: a fused single-dispatch Metal kernel is still 19–24× slower than CPU `lfilter`; a single GPU core loses to a single CPU core on a dependency chain, full stop (§3.1) |

**Algorithm-level verdicts** (details in §5):

| Module | Current rationale | Is the rationale *mathematically* correct? | Does the *decision* stand? |
|---|---|---|---|
| `anova` (SS chains) | launch overhead dominates tiny fits | Partially — batching dissolves launch overhead, but a single call has no batch axis | ✅ Stands |
| `survival.coxph` | "inherently sequential" | ❌ **False** — risk-set sums are prefix sums; GPU Cox is peer-reviewed at 42–52× (n=10⁵–10⁶) | ✅ Stands *at typical n*, rationale should be rewritten |
| ARIMA exact-ML / ETS | "inherently sequential" | ❌ False for additive forms (associative scan, O(log T) span) — but scan loses to Cython below T≈10⁶ (measured) | ✅ Stands |
| Multiplicative ETS | sequential | ✅ **True** — state-dependent coefficients are non-associative; no known scan form | ✅ Stands |
| STL/LOESS | sequential | Mostly false (per-point fits are independent) but robustness outer loop is sequential | ✅ Stands (112 µs total runtime) |
| `factor_analysis` | tiny p×p eigh chain | ✅ True at these shapes | ✅ Stands |
| acf/stationarity | O(n·lag) trivia | ✅ True | ✅ Stands |
| Concordance (Fenwick) | "inherently sequential" | ❌ False — inversion counting parallelizes (merge-rank, Cole 1988) | ✅ Stands below n≈10⁶ |
| MICE/MVNMLE small-n MPS sweeps | "dispatch floor NOT recoverable" | ❌ **False — this is the finding of the audit.** Floor already collapsed on torch ≥2.12 (measured); fusion buys the rest | ⚠️ **Revisit — §6 Phase 1–2** |
| pystatsbio `batch_auc` on MPS | scatter_add ~3000× slower | ❌ **Now false** — re-measured on torch 2.13: 0.45 ms vs CUDA 0.06 ms (7.6×), and **8× faster than CPU** | ⚠️ **Prohibition obsolete** (sibling repo — flagged, not touched) |
| MICE merge-rank bridge (MPS `searchsorted` workaround) | searchsorted ~1136× slower than CUDA | ❌ **Now false** — re-measured on torch 2.13: 37 µs vs CUDA 5 µs (7.4×), 38× faster than CPU | ⚠️ **Retire-able on torch ≥2.13** |

---

## 1. What we measured today (your hardware, not the literature's)

### 1.1 Dispatch overhead per op — the numbers that reframe everything

1000-op chain of `y = y*0.999 + 1e-6` on a 32-element tensor:

| Path | Per-op cost | vs CUDA eager |
|---|---|---|
| CUDA eager (RTX 5070 Ti, torch 2.11-dev) | 3.25 µs | 1× |
| CUDA Graphs replay (same chain) | 0.99 µs | 3.3× better |
| **CUDA torch.compile** | **0.014 µs** | **~230× better** |
| MPS eager (M2 Max, torch 2.12) | 4.5 µs | ≈ parity with CUDA eager (!) |
| **MPS torch.compile** | **0.021 µs** | **~215× better than MPS eager** |
| MLX lazy (1 eval per 1000 ops) | 6.0 µs | — |
| MLX `mx.compile` | 0.39 µs | — |
| MLX forced eval per op (the "sync-per-op" regime) | 81.5 µs | — |

Three things fall out of this table:

- **The GPU_NOTES "~0.5–1 ms per op, not recoverable, no graph capture" claim is dead on torch 2.12.** MPS eager is now ~4.5 µs/op for elementwise chains — within noise of CUDA eager — and `torch.compile` works on MPS and fuses the whole chain to one dispatch. The old number was real but was a *MPSGraph-per-op + command-buffer-commit + sync* artifact of torch 2.10/2.11 (the adversarial verifier found PyTorch issue #148219 measuring 1.2–1.4 µs dispatch for native-shader MPS ops vs ~20 µs for MPSGraph-routed ones, on an M1 Pro; and PyTorch 2.13's release notes *explicitly* migrated scatter/gather, cumsum, sort, reductions, RNG etc. off MPSGraph because its per-dispatch overhead "can dominate execution time").
- **`torch.compile` is the ~free 100–200× dispatch lever on both platforms.** Caveats: Inductor-Metal still has codegen bugs on some multistage reductions (pytorch #152155); `reduce-overhead` mode had a 40–55% regression in torch 2.10 vs 2.9 (#174575) — pin and benchmark, don't assume.
- **The fp32 wall on Apple is untouched by all of this** (§3.2). Every µs saved on MPS is saved in a non-reference fp32 tier.

### 1.2 Batched tiny linear algebra (the ANOVA-shaped question)

Batched SPD factor+solve, B problems of size p×p:

| p | B | GPU fp64 (CUDA) | GPU fp32 (CUDA) | CPU NumPy batched | CPU Python loop |
|---|---|---|---|---|---|
| 5 | 10,000 | 0.31 ms | 0.20 ms | 2.6 ms | 48 ms |
| 20 | 10,000 | 1.6 ms | 0.50 ms | 36 ms | 84 ms |
| 50 | 10,000 | 6.5 ms | 2.6 ms | **204 ms** | 213 ms |

And the K=30 small-OLS chain (n=500, p=10 — an ANOVA-like shape):

| Path | Time |
|---|---|
| CPU loop of 30 fits | 0.32 ms |
| GPU loop of 30 fits (eager, per-fit) | 5.9 ms — **18× slower than CPU** |
| GPU **one batched** call (warm) | 0.17 ms |
| GPU batched incl. H2D transfer | 0.29 ms |

**Reading:** batching tiny problems is emphatically GPU-shaped (22–31× over NumPy's own batched path at p=20–50, B=10⁴) — *and* PyTorch already routes batched `torch.linalg.cholesky` to cuSOLVER `potrfBatched` under the hood, so no raw CUDA is needed to get this. The launch-overhead objection to "many small fits" dissolves **if and only if the fits arrive as one batch**. A single `anova()` call has 4–7 fits — there is nothing to batch; the CPU does the whole call in microseconds. The decision stands; the *reformulation* is sitting there if a batch-of-ANOVAs API (mass-univariate, permutation-F) ever exists.

On MPS: batched `cholesky` is fine (8.4 ms vs 103 ms CPU at p=50, B=10⁴), but `solve_triangular` remains catastrophic (114 ms–1.1 s, n-independent) and `cholesky_solve` — newly *implemented* in torch 2.12 — is implemented **badly** (208 ms–2.1 s, ~250× the factorization cost). It went from "fails loud" to "silently slow," which validates keeping the in-house matmul-series inverse. `eigh` still raises.

### 1.3 The sequential recursion (Kalman/ETS proxy) — the mathematical wall, measured

First-order linear recurrence `x_t = a·x_{t-1} + b_t` (this is the *easiest possible* member of the family — the real Kalman filter carries an r-vector and r×r covariance):

| T | CPU `lfilter` (compiled seq.) | CUDA log-depth scan | MPS scan | Winner |
|---|---|---|---|---|
| 1,000 | 0.011 ms | 0.29 ms | 1.2 ms | **CPU by 26×–110×** |
| 10,000 | 0.040 ms | 0.40 ms | 1.4 ms | **CPU by 10×–36×** |
| 100,000 | 0.35 ms | 0.49 ms | 1.8 ms | **CPU by 1.4×–5×** |
| 1,000,000 | 3.3 ms | 1.3 ms | 2.4 ms | GPU by 1.7–2.5× |

(Scan is numerically exact: max rel. error 1.9e-15 vs the sequential fp64 reference.)

And the three ways of running the recurrence *sequentially* on GPU, T=1000:

| Path | Time | vs CPU (19.6 µs) |
|---|---|---|
| CUDA eager per-step loop | 6.6 ms | 340× slower |
| CUDA Graph replay of the fully unrolled 1000-step chain | 1.95 ms | 100× slower |
| **Custom fused Metal kernel, ONE dispatch, whole loop in-kernel** | **342 µs** | **23× slower** |

That last row is the decisive experiment for the "dropping down" hypothesis: even with *zero* framework overhead — one hand-written Metal kernel, one dispatch, the entire recurrence executed inside the GPU — the GPU loses by ~20× and stays ~20× behind as T grows (8.1 ms vs 0.41 ms at T=100k, i.e. ~81 ns/step vs ~4 ns/step). No language fixes this; it is the silicon (§3.1).

Independent replication: the adversarial verifier ran its *own* benchmarks (numba `@njit` and statsmodels' production Cython SARIMAX filter vs a full Särkkä 5-tuple Kalman-combine scan in PyTorch) and got the same shape — statsmodels' Cython loglike beats the scan 7–9× everywhere in T=100…50,000 (e.g. 3.8 ms vs 35.5 ms at T=10⁴ on CPU), and the scan on MPS was 3–4 orders of magnitude behind. Even the published literature's own ICASSP paper concedes the parallel form is slower than sequential *on CPU*.

### 1.4 The torch 2.13 re-benchmark of the GPU_NOTES op table (run 2026-08-12)

Identical script on both machines: Powerhouse MPS (torch **2.13.0**) vs Forge CUDA (torch 2.11-dev). Every claim in the GPU_NOTES op-status table, re-measured:

| Claim (GPU_NOTES, torch 2.10/2.11) | torch 2.13 MPS measured | CUDA measured | New ratio | Status |
|---|---|---|---|---|
| `scatter_add_` sparse ~150 ms/call, **~3000×** vs CUDA | **0.45 ms** (random-sparse, 2.5M items); 0.49 ms (tied) | 0.060 ms | **7.6×** | **FIXED** (~300× improvement) — also 8× faster than CPU `np.add.at` (3.9 ms) |
| `searchsorted` **~1136×** vs CUDA @ n=20k | **37 µs**/call | 5.0 µs | **7.4×** | **FIXED** (~150× improvement) — 38× faster than CPU (1.4 ms) |
| `solve_triangular` ~4 ms n-independent, ~250× | 3.8 ms @ B=100 p=20 → 352 ms @ B=10⁴ (now scales with B) | 0.01–0.23 ms | **12–190× the Cholesky cost** | Still bad — matmul-series workaround stays |
| `cholesky_solve` unimplemented | Implemented: 66 ms @ B=1000 p=20 | 0.05 ms | **~1300×** | Implemented-but-terrible — workaround stays |
| `linalg.eigh` unimplemented | Still `NotImplementedError` | 0.23 ms | — | Unchanged |
| `linalg.solve` / `inv` ~100–300× matmul | 464 / 469 ms @ B=1000 p=20 | 0.08 / 0.15 ms | **~3300× matmul** | Still terrible (worse ratio — matmul got faster) |
| `pinv` | 29.6 ms — but via **silent CPU-fallback SVD** (explicit warning now) | 1.0 ms | — | Fallback now *warns* (was silent) |
| `lstsq` unsupported | Still `NotImplementedError` | 0.28 ms | — | Unchanged (CPU detour stays) |
| `svd` silent CPU fallback | Still CPU fallback, now with explicit `UserWarning` | 0.98 ms | — | Improved diagnosability only |
| fast set (matmul/cholesky_ex/sort/cumsum/gather) | Confirmed fast: cumsum 2.5M = 0.099 ms, gather 0.140 ms, cholesky_ex B=1000 = 0.23 ms | all fast | — | Confirmed |
| dispatch floor ~0.5–1 ms/op, "no graph capture" | **4.16 µs/op** eager; **0.023 µs/op** torch.compile | 3.27 µs / 0.0115 µs | ≈ parity | **COLLAPSED** (~100–200× improvement in eager; compile ≈ CUDA-class) |

**Direct consequences:**

1. **The pystatsbio `batch_auc` MPS prohibition is obsolete.** The op it banned MPS over is now ~8× *faster* than the CPU path it routes to. The old end-to-end numbers (CUDA 0.015 s / CPU 0.9 s / MPS ~20 s at 5,000 markers) imply MPS should now land far *below* CPU — needs one end-to-end confirmation run in pystatsbio. *(Sibling repo — flagged per Rule 8, not touched.)*
   **→ Confirmed and done (2026-08-12, user-authorized):** end-to-end on Powerhouse (M2 Max, torch 2.13.0), 5,000×500 = 0.012 s MPS vs 0.67 s CPU (~55×), 2.6–67× across the full grid, `GPU_FP32`-tier match in every cell. pystatsbio lifted the ban, version-gated on torch >= 2.13 (local predicate on the `mps_native_kernels()` pattern), and `'auto'` picks MPS there. Forensic note: on this machine the same workload is also fast on torch 2.9/2.10/2.11/2.12 (35–38 ms warm; `scatter_add_` 2.5M random buckets ≤ 1.2 ms) — the ~150 ms/call pathology was a property of the April 2026 OS/MPSGraph stack, not torch version alone, so the >= 2.13 gate stands as the only Python-visible predicate that *guarantees* the fast kernels.
2. **The MICE merge-rank bridge (~10 ops replacing 1 `searchsorted`) is retire-able on torch ≥2.13** — native searchsorted is fast now, and retiring the bridge also removes ~10 ops/step from the dispatch budget.
3. **`_SERIES_INV_MIN_NOBS = 3000` was tuned against the old dispatch floor** — with dispatch ~100–200× cheaper, that crossover has moved and needs re-tuning (possibly the series inverse now wins at all sizes; possibly the single `solve_triangular` never does).
4. **The dispatch component of the MICE sweep (~15–60 ms/step of pure dispatch at 30–60 ops/step under the old floor) is now ~0.13–0.25 ms/step in eager** — before any fusion. The small-n MPS regime the docs wrote off is back in play.

---

## 2. The MPS dispatch floor: refuted, with a decomposition

GPU_NOTES.md's claim under test:

> "MPS pays ~0.5–1ms of command-encode overhead per op, and — unlike CUDA — has no graph-capture to amortize it… This is an intrinsic platform limitation."

**Verdict (adversarial verifier: CONFIRMED-recoverable; my measurements agree): it is a framework-regime artifact, not a hardware floor.** The correct cost decomposition on Apple silicon:

| Event | Actual cost |
|---|---|
| Encoding one more compute dispatch into an open command buffer | ~1–6 µs |
| PyTorch native-Metal-shader op dispatch | ~1.2–1.4 µs |
| PyTorch MPSGraph-routed op dispatch (the old default path) | ~20 µs |
| Command buffer **commit + GPU wakeup + CPU sync** | ~0.1–1 ms |

The ~0.5–1 ms/op regime is what you get when each op (or each step) pays a commit+sync round trip — which is exactly what the MICE sweep's per-step `.item()` calls did before 3.14.0, and what MPSGraph's per-op scheduling approximated. Apple's own guidance (tech talk 10580) says precisely this: batch more encoders per command buffer before committing.

**What changed under the library since the notes were written:**
- torch 2.12 (measured today): eager elementwise ~4.5 µs/op; `torch.compile` works on MPS → 0.021 µs/op.
- torch 2.13 (July 2026): mass migration of ops (scatter/gather, cumsum, sort, reductions, RNG, copy/cast) from MPSGraph to hand-written Metal kernels; `torch.mps.compile_shader()` (since 2.7) lets a library JIT MSL kernel strings at runtime with **no build toolchain and no new dependency**; `torch.mps.load_metallib` for precompiled libraries.
- Metal 4 (shipped fall 2025): reusable command buffers (`MTL4CommandBuffer`) — a CUDA-Graphs-replay analogue below the framework; not yet adopted by torch/MLX.

**What did NOT change:** no fp64 (hardware, permanent — MSL has no `double`; the only emulation lib is archived-incomplete at ~1/32–1/64 fp32 throughput); `solve_triangular`/`cholesky_solve`/`solve`/`inv` still terrible (measured on 2.13, §1.4); `eigh`/`lstsq` still missing; Apple atomics genuinely weaker than CUDA's (32-bit device atomics only) — but the *practical* scatter_add gap is now 7.6×, not 3000× (§1.4), so "atomics-weak" no longer translates into "unusable."

**MLX is not the answer** despite the beautiful `mx.compile` number: its entire `linalg` is CPU/LAPACK (zero Metal decomposition kernels — verified against the source tree), fp64 is CPU-only by design, and a native-Metal runtime (BaseRT) beats *well-batched* MLX by only 1.01–1.35× — so the 10–100× is specifically vs per-op-synced eager, not vs a good baseline. Adding MLX would add a third numerics stack for tens of percent.

**Where this lands:** the documented "only lever left is reducing op count per step" was exactly right — and the lever now exists in three grades: (1) upgrade torch + re-audit which of the ~30–60 ops per MICE step still pay MPSGraph/sync costs, (2) `torch.compile` the step, (3) `torch.mps.compile_shader` a fused kernel chain committing one command buffer per solver iteration. Grade 3 is the "drop below PyTorch" experiment worth running; MVNMLE's objective evaluation (one L-BFGS eval = a fixed op DAG, already sync-minimized to one round trip) is the ideal first target.

---

## 3. The three walls, with the math

### 3.1 Wall 1 — serial dependency chains (mathematical; no language escapes it)

A recurrence x_t = f_t(x_{t-1}) has a critical path of length T: computing x_T requires T dependent evaluations of f. On any machine, the time is bounded below by T × (latency of one f-step). The relevant silicon numbers: a modern CPU core retires a dependent FMA chain at ~4–5 GHz with 4–5-cycle latency and full superscalar reordering around it (~1–4 ns/step for small f); a GPU "core" is a ~1–2 GHz in-order lane with no branch prediction and high-latency memory — **10–50× slower on a serial chain**. Measured today: 81 ns/step (single-thread Metal kernel) vs 4 ns/step (CPU `lfilter`) — 20×. Raw CUDA, Mojo, or Metal cannot repeal this; they all run on the same lane.

**The only mathematical escape is reformulation.** If f_t is *affine* — x_t = A_t x_{t-1} + b_t — then the maps compose associatively:

    (A₂,b₂) ∘ (A₁,b₁) = (A₂A₁, A₂b₁ + b₂)

so the T-fold composition is a *scan* over a monoid, computable in O(log T) span (Blelloch 1990). Kalman filtering — and hence the ARIMA exact-ML innovations likelihood — is covered by the deeper result of Särkkä & García-Fernández (IEEE TAC 2021): the per-step Bayesian update elements a_k = (A_k, b_k, C_k, η_k, J_k) with combine

    A_ij = A_j(I + C_iJ_j)⁻¹A_i
    b_ij = A_j(I + C_iJ_j)⁻¹(b_i + C_iη_j) + b_j
    C_ij = A_j(I + C_iJ_j)⁻¹C_iA_jᵀ + C_j    (etc.)

form an associative operation; Yaghoobi et al. (SIAM J. Sci. Comput. 2023) add square-root variants for fp32 stability and derive the *likelihood itself* in O(log T). Additive-error ETS is the affine case directly: x_t = (F − gwᵀ)x_{t-1} + g·y_t. **So "inherently sequential" is mathematically false for these** — that phrasing in CONVENTIONS.md does not survive scrutiny.

But the scan buys *span*, not *work*: it costs 2–4× the flops (the combine is ~6–10 matmuls + an inverse vs ~4 matmuls for a sequential step) plus ~2·log₂T kernel launches. Result (measured, both platforms, twice independently): **crossover vs compiled sequential CPU is T ≈ 10⁶** for the scalar case — and worse for real Kalman (the combine's matrix inverses). R-typical series are T = 10²–10³. The decision stands by three orders of magnitude of margin; only the stated *reason* needs correcting.

**Genuinely sequential (no known escape):** multiplicative ETS (update coefficients depend on the running state → composition non-associative; log-transform only linearizes the fully multiplicative subcase), and *across-iteration* dependence of every outer loop (Newton, EM fixed-point θ_{k+1}=M(θ_k), Nelder-Mead) — though each is only ~5–20 dependent steps around a parallelizable inner evaluation.

**Cox PH is the case where the "sequential" label was simply wrong** (verifier: CONFIRMED). Sort by event time; the risk set R(t_i) is a suffix of the sorted array; therefore S0/S1/S2 = Σ_{j∈R(t_i)} w_j·{1, x_j, x_jx_jᵀ} are suffix cumulative sums — *literal prefix sums*, the canonical GPU primitive. The library's own CPU implementation already exploits this (single reverse cumsum, `_cox.py:177-185`)! Efron ties decompose into segmented scans (production GPU code exists: Novartis TorchSurv computes Efron denominators via `exp_hz.flip(0).cumsum(0).flip(0)` — shipped, FDA-catalogued). Peer-reviewed GPU Cox (Suchard group, JCGS 2024): 42–52× at n=10⁵–10⁶ with P≈1000 sparse covariates; a 434,866-patient study went from ~2 days (8 cores) to 3.87 h. The honest rationale for PyStatistics: dense, small-p Cox at n ≲ 10⁴ finishes in tens of ms on CPU (already at R parity: n=20,000 tied fit ≈ 0.93× R) — below the 100 ms threshold. It is an engineering-threshold decision, not an impossibility, and at n ≳ 10⁵ the math is sitting there.

### 3.2 Wall 2 — precision and parity (permanent on Apple; structural everywhere)

- **Apple GPUs have no fp64.** Not in MSL, not in hardware, M1–M5 inclusive. Emulation: archived prototype at ~1/32–1/64 fp32 throughput whose fast variant (double-single, ~47-bit mantissa, 8-bit exponent) cannot reproduce IEEE-double rounding. **The R-parity fp64 reference path can never move to Apple GPU**, whatever the language. Every Apple-GPU gain lives in the fp32 accelerator tier the library already treats as non-reference.
- **Consumer NVIDIA fp64 runs at 1/64 the fp32 rate** (RTX 5070 Ti included) — hardware, not software; another reason `gpu_fp64` wins only when the CPU alternative is far off the pace.
- **The bit-identity contract on the 17 Cython kernels bans GPU by construction.** The parity tests assert exact equality (atol=0) vs numpy references; the kernels are compiled `-ffp-contract=off` (zero FMA, verified in assembly), explicit scalar loops with pinned accumulation order (BLAS banned after discovering Numba's matmul already differed from `@` by ~1.4e-17), `x*x` never `pow`. Any parallel reduction reorders sums; fp32 is out entirely; even a CUDA fp64 port would have to serialize its accumulations — forfeiting the parallelism that was the point. GPU versions of these kernels are *contractually* impossible under the current test regime, independent of the performance question. (GPU paths elsewhere are tolerance-tested — the GPU_FP32 tier — which is why GPU backends exist at all.)

### 3.3 Wall 3 — dispatch and transfer floors for tiny work (physical, but *bounded* and now mostly amortizable)

- CUDA launch: ~4–6 µs eager, ~2.5 µs + ~1 ns/node for a whole CUDA-Graph replay (a 1000-kernel graph replays for ~3.5 µs of submission). PyTorch eager op overhead ~3–7 µs (measured 3.25 µs today). `torch.compile` fuses + graphs it away (measured 0.014 µs/op).
- PCIe: ~7–10 µs per transfer *floor* regardless of size; ~8 ms to move a 160 MB fp64 design matrix on Gen4. One-shot small fits are transfer-bound before they are compute-bound — the threshold rule's "transfer-bound" exclusion is correct and stays.
- MPS: see §2 — the floor was 20–1000 µs depending on regime and is now ~1–6 µs when fused.
- **Stan Math's fused-GPU-GLM data point** (the best published analogue of "hand-fuse the whole IRLS step in one kernel"): even with likelihood+gradient in ONE kernel, logistic regression crosses speedup=1 only around n≈10⁴–10⁵. Fusion widens GPU wins where the GPU already wins; it does not move the crossover much. The library's `>100 ms CPU with >80% BLAS-mappable` rule is *quantitatively* vindicated by the external evidence.

---

## 4. What was actually tried and rejected (the audit trail you half-remembered)

The changelog and docs preserve every formulation experiment. The ones matching your memory of "too slow or unacceptably inaccurate":

| Attempt | Fate | Evidence |
|---|---|---|
| **GAM GPU backend** (built! 4.8–28.6× measured) | **Removed at 4.6.0** — fp32 silently wrong (LU vs Cholesky pick different null-space representatives → smooth-term χ² off 8×; cond ~1e17 at small λ), fp64 slower than CPU. The exact "too slow AND inaccurate" pair | CHANGELOG 4.6.0; gam/backends/ now empty |
| **MVNMLE autodiff-through-Cholesky gradient** | Rejected — Cholesky backward pathologically slow on Metal (100-var fit >30 min); replaced by closed-form gradient (3.10.0, ~20×) | CHANGELOG 3.10.0 |
| **R's inverse-Cholesky parameterization on GPU** | Rejected — needs a matrix inverse per unpack; forward-Cholesky (Σ=LLᵀ) adopted as *required* for GPU viability, later became the CPU default too (100 s → 0.1 s) | parameterizations.py; CONVENTIONS.md:108-111 |
| **Device-resident MVNMLE EM (first attempt)** | Slower than CPU on every shape that matters; also: the pre-2.1.0 "GPU EM" **never actually ran on GPU** (constructor flag set, numpy paths executed) | CHANGELOG 2.1.0 |
| **BoundedCholeskyParameterization** (fp32 stability) | Built, shelved (`use_bounded=False`) | parameterizations.py:159-252 |
| **montecarlo GPU statistic auto-detection** | Rejected as silently-wrong (detected "mean" from one resample); replaced by explicit declaration | CHANGELOG 4.6.7 |
| **PCA Gram path** | Kept but demoted: 1.1–2.2× measured vs projected 30–100× (GPU-SVD was already bandwidth-bound) | CHANGELOG |
| **MoM MCAR GPU** | Never beat CPU on any shape; feature later removed | CHANGELOG 2.1.0/3.0.0 |
| **CGLS/LSMR iterative solvers** | Planned, never built (`iterative.py` doesn't exist); ROADMAP says compelling only at n>10M | ROADMAP:190-197 |
| **solve_triangular on MPS** → matmul-Neumann-series inverse; **searchsorted** → merge-rank; **eigh/svd** → randomized HMT SVD + CholeskyQR2; **topk** → windowed argmin | The complete "reformulate onto MPS-fast primitives" playbook — this is the library's signature move, and it worked (mid/large-n MPS gap closed to the ~3–4× raw fp32 silicon ratio) | triangular.py, _gpu_methods.py, gpu_pca_randomized.py |

**The meta-pattern:** every successful remedy stayed inside PyTorch's op vocabulary; custom kernels (CUDA C, Metal, Triton, torch.compile) appear *nowhere* in the history. The op-level lever set is exhausted — which is exactly why the remaining headroom (§2) is below the op level, and why it went unexplored: until torch 2.12/2.13, the tooling to reach it cheaply didn't exist.

---

## 5. Technology deep-dives

### 5.1 Mojo — the direct answer to your question

**Status (verified against primary sources, Aug 2026):** Mojo 1.0.0 shipped **yesterday** (Aug 11, 2026) alongside MAX 26.5. Modular was acquired by **Qualcomm** (~$3.9B, closed July 2026). GPU support is real and shipped: NVIDIA (Turing→Blackwell; B200 in CI), AMD (MI300X/MI355X in CI; RDNA consumer parts), Apple Silicon M1–M5 via a Metal/AIR backend ("known compatible" tier only — *no* Apple device in Modular's CI; no GPU debugger; no in-kernel print). An independent ORNL/UTK SC'25 study measured 87–100% of CUDA/HIP for memory-bound kernels, 0.54 portability for compute-bound, inconsistent atomics.

**Why it fails for PyStatistics today (verifier: REFUTED on all three load-bearing predicates):**

1. **Distribution.** There is no supported way to ship a prebuilt Mojo `.so` in a wheel to users who don't have the Modular runtime — compiled artifacts hard-link `libKGENCompilerRTShared.so` under `site-packages/modular/lib`; the forum thread asking exactly this (Aug 2025) is unresolved. Python interop is officially **Beta** (≤6 `PythonObject` args, no native kwargs). Cython emits self-contained C extensions; Mojo cannot replicate that on *any* platform. And there is **no Windows target at all** (WSL only) — the library just added Windows wheels to its build matrix.
2. **Licensing/governance.** As of Mojo 1.0 the GPU APIs moved *out* of the (Apache-2.0) stdlib into the proprietary-licensed `max` package. The compiler is still closed-source (open-sourcing promised by end of 2026, teased for ModCon '26 — possibly next week). The roadmap is now owned by a chip vendor.
3. **No performance case.** CPU: every credible 2026 benchmark puts Mojo in the same tier as Cython/Numba for float64 scalar loops (n-body: Cython 10 ms, Mojo 11 ms); Modular staff concede CPU "has been a little neglected." And §3.2's bit-identity contract would demand verifying Mojo's codegen (FMA contraction, accumulation order, its own non-numpy libm) to the last bit per platform. GPU: everything §3 says applies to Mojo kernels identically — fp32-only on Apple (hardware), serial chains still serial. The *one* genuinely differentiated capability — single-source portable GPU kernels including Metal, with compiled command buffers that would dodge the dispatch floor — is real but arrives at the same destination as `torch.mps.compile_shader`, at the cost of a proprietary toolchain dependency instead of zero new dependencies.

**Re-evaluation triggers** (worth a calendar note, genuinely): compiler open-sourced AND Python interop out of Beta AND a wheel-distribution story (any two of three would justify a fresh look). The ROI would still be portability, not speed.

### 5.2 Triton

No Metal backend — upstream **rejected** a working community Apple/AIR backend (PR #9701, Mar 2026) as out of scope; the designated out-of-tree home doesn't contain one yet. So Triton cannot touch the one platform with real headroom. On CUDA: **`tl.dot` has no fp64** (supported types top out at fp32) — fp64 matmul in Triton means manual FMA loops, guaranteed slower than the cuBLAS DGEMM torch already calls; batched tiny factorizations are a known structural weak spot (power-of-2 blocks, 16×16 minimum MMA tiles); Triton's own Python launch path costs tens of µs (open RFC #10140) so small-kernel sweeps need CUDA-Graph wrapping anyway — at which point `torch.compile(mode="reduce-overhead")` gets you there without writing a kernel. Wheels are Linux-only. No numerical-accuracy contract (nothing about ULP/rounding in the semantics doc). **Verdict: dominated by torch.compile for this library's needs on every axis.** (If a hand-fused fp64 elementwise/reduction pipeline is ever needed on CUDA where Inductor fuses poorly: Helion 1.0 — PyTorch Foundation, GA Apr 2026, compiles to Triton with autotuning — is the sane entry point, not raw Triton.)

### 5.3 Raw CUDA C++

Decomposes into exactly three recoverable costs, all measured above: launch overhead (recoverable in-framework: graphs/compile), intermediate memory traffic (recoverable: Inductor fusion), and tiny-matrix throughput (torch already calls `potrfBatched`/`gemmStridedBatched`; the residual niche is MAGMA-style register-tiled kernels for p ≲ 16, where cuBLAS craters — 105 GFlop/s at size 15 — and custom kernels get up to 6×). What raw CUDA can *never* buy: serial latency (§3.1). The RAPIDS cuML precedent is instructive: their hand-optimized batched ARIMA is *slower than statsmodels for a single series* and wins only via many-series batching (1000 series ≈ 12× the cost of one). **Verdict: not worth a build-toolchain and maintenance surface for currently-shipped workloads; becomes interesting only if a batched-tiny-fits API ships and profiling shows the p<16 regime matters.**

### 5.4 Custom Metal kernels (the winner, such as it is)

Three shipping Python paths, ranked: (1) **`torch.mps.compile_shader`** — MSL source strings JIT-compiled at runtime, invoked on ordinary MPS tensors; zero build toolchain, zero new dependencies; the natural fit for fusing MVNMLE's objective eval or a MICE step into 1–3 dispatches per iteration. (2) `mx.fast.metal_kernel` — same idea, requires adopting MLX arrays (not worth it, §2). (3) Full C++/ObjC torch extension — most control, per-torch-version ABI churn, platform wheels; only if (1) proves out and needs hardening. Expected gain, honestly bounded: 10–100× on *per-op overhead* in the small-n dispatch-bound regime (i.e., exactly the documented MICE small-n floor and the ~15–60 ms/step of pure dispatch it implies); tens of percent at most vs an already-well-batched baseline (BaseRT vs MLX: 1.01–1.35×). All of it fp32-tier only.

---

## 6. The recommended upgrade path

This is the answer to "we've left performance on the table — how do we close the gap?" Four phases, each with a measurable gate, ordered so every phase's result decides whether the next is worth its cost. All the leverage is on the **MPS fast paths of MICE and MVNMLE** — precisely the two modules with papers attached — plus a workaround-retirement sweep enabled by torch 2.13.

**Where the gap actually is (calibrating expectations honestly):** the collapsed dispatch floor and the fixed kernels help exactly where the library was *dispatch-bound* — the small-n MPS sweep regime the docs wrote off — and help little where it was *compute-bound* (mid/large-n already sits at the ~3–4× raw fp32 silicon ratio vs CUDA, which no software touches). So: order-of-magnitude end-to-end gains are plausible in the small-n MPS corner (where ~15–60 ms/step of dispatch just became ~0.2 ms); expect meaningful-but-not-10× (roughly 1.5–3×) at the paper-scale shapes; expect ~nothing at large compute-bound shapes. The gates below measure exactly this, and the paper decision (§6.5) keys off what they return.

### Phase 0 — Re-baseline on torch 2.13 (days; zero algorithmic risk)

1. Validate the library's GPU test suite on torch 2.13.0, both devices (MPS here, CUDA on Forge). The MPS solver paths are documented as validated *per torch version* — 2.13 becomes the new validated tier.
2. Re-run the existing committed benchmarks (`benchmarks/mvnmle_bench.py`, `benchmarks/mvnmle_gpu_per_eval.py`, the MICE timings from CHANGELOG 3.13/3.14) on 2.13 to establish the new baseline **before** touching anything — otherwise Phase 1–3 gains can't be attributed.
3. Doc corrections (Rule 11, wording only): GPU_NOTES' dispatch-floor section gets the commit-vs-encode decomposition and the §1.4 table; the op-status table gets torch-version columns; CONVENTIONS' "inherently sequential… (Cox PH partial likelihood)" becomes the threshold statement (Cox = prefix sums, Suchard JCGS 2024; Kalman/additive-ETS = associative scans, Särkkä & García-Fernández 2021; measured crossovers n≈10⁵ / T≈10⁶, far above R-parity sizes).

**Gate:** suite green on 2.13 + fresh baseline numbers in hand.

> #### Phase 0 — EXECUTED 2026-08-12. Results:
>
> **Test suite on torch 2.13.0** (fresh venvs, package built from HEAD):
> - **CUDA** (Forge, Linux, 2.13.0+cu130): **4407 passed / 92 skipped / 0 failed** ✅
> - **macOS MPS** (Powerhouse, 2.13.0): **4389 passed / 109 skipped / 1 failed** — the failure is
>   `test_ets_selection.py::TestDampedConvergence::test_free_phi_dominates_fixed_phi_probes`:
>   CPU-only (no torch on that path), deterministic, and invariant across numpy 2.4.6/2.5.2 and
>   scipy 1.17.1/1.18.0 (proven by isolation) — the free-phi ETS(M,Ad,M) fit converges to a local
>   optimum 0.25 loglik below the fixed-phi probe on macOS while passing on Linux. This is the
>   platform-libm optimizer-trajectory class already documented on Windows in
>   CYTHON_MIGRATION_PROPOSAL.md §17. Left unfixed deliberately: whether to robustify the test
>   (same-outcome assertion, as done for the Windows case) or the phi-probe logic is a design
>   decision to make, not a torch-2.13 blocker.
>
> **New benchmark baselines** (all commands recorded; synthetic MICE data seed=42):
>
> | Benchmark | CUDA (torch 2.13) | MPS (torch 2.13) |
> |---|---|---|
> | MVNMLE per-eval, p=100 n=50k (obj+grad → fused) | 1.42 → **1.26 s** (1.13×) | 4.76 → **2.53 s** (1.88×) — matches the historical 4.7→2.5 s |
> | MVNMLE EM, breast (569×30): CPU → GPU | 1.99 s → **0.168 s** (11.8×, matches historical 14.6× class) | GPU = documented deliberate refusal |
> | `little_mcar_test`, breast: CPU → GPU | 2.02 s → **0.214 s** | refusal propagates (via EM) |
> | MICE p=20 m=100 maxit=5, GPU @ n=2k/8k/20k | 1.08 / 0.26 / **0.63 s** (first incl. CUDA init) | 1.65 / 3.06 / **4.17 s** |
> | MICE CPU @ n=2k / 20k | 3.30 / 34.5 s | 3.48 / 39.2 s |
> | → MICE GPU speedup @ n=20k / n=2k | **~53×** / ~3× | **~9.4× / ~2.1×** — the small-n MPS corner is exactly Phase 1–2's target |
>
> **Defects found and fixed in passing** (Rule 11; in the working tree, uncommitted):
> - `benchmarks/mvnmle_bench.py` was **broken against the 4.0+ API** — still passed the pre-4.0
>   `algorithm=` kwarg (renamed `method=` in 4.0); every case failed with TypeError. Also its
>   `_gpu_available()` checked CUDA only, silently skipping all GPU rows on Apple Silicon. Both
>   fixed. (Not in CI — that's how it rotted.)
> - `mvnmle/solvers.py`'s MPS-EM refusal message pointed to `docs/GPU_BACKEND_NOTES.md`, which
>   doesn't exist → corrected to `docs/GPU_NOTES.md`. Note: that refusal's *stated reason*
>   (Metal launch overhead) is itself now partially stale — re-evaluating the MPS-EM refusal
>   belongs on the Phase 1 list.
> - Doc corrections landed: GPU_NOTES.md (scatter_add update note, dispatch-floor correction with
>   the measured decomposition, dual-version op-status table), CONVENTIONS.md non-target wording,
>   and UNRELEASED.md entries for all of the above.

### Phase 1 — Reap the torch-2.13 free wins (days–weeks; low risk, in-PyTorch)

1. **Retire the MICE merge-rank bridge** (`_insertion_rank`'s ~10-op MPS path) in favor of native `searchsorted` on torch ≥2.13 — version-gated or via a minimum-torch bump for the MPS fast path. Saves ~10 ops/step *and* deletes workaround code.
2. **Re-tune `_SERIES_INV_MIN_NOBS`** (empirically tuned against the old dispatch floor) and re-ablate matmul-series-inverse vs `solve_triangular` on 2.13 — the `method='auto'|'solve'|'blocked'` hooks built at 3.11.0 exist for exactly this.
3. **Sync-point re-audit** of the MICE sweep and polr/logreg paths under the new floor: decisions made when a sync cost ~1 ms (quadratic-interpolation line search, once-per-sweep degeneracy guard) may now be re-balanced toward robustness where that trades better.
4. Flag to pystatsbio (separate task, its own repo): re-run `batch_auc` end-to-end on 2.13-MPS; the RuntimeError prohibition and the "auto routes to CPU" policy are both based on a 3000× number that is now 7.6×.

**Gate:** MICE MPS end-to-end at n ∈ {2k, 8k, 20k} (m=100, p=20) vs the Phase-0 baseline. Keep what measures faster; revert what doesn't. R-fidelity + GPU_FP32-tier tests must stay green — these are semantics-preserving changes, so no statistical re-validation is triggered.

> #### Phase 1 — EXECUTED 2026-08-12. Results:
>
> **Ablations (MPS, torch 2.13, MICE shapes m=100, q+1=21):**
> - Native `searchsorted` beats the merge-rank bridge **9.5× / 12.0× / 16.4×** at n=2k/8k/20k
>   (max rank difference 1 — the documented tie convention the exact window search refines away).
> - The matmul-series inverse beats `solve_triangular` at **every** n_obs — **6.75× / 3.2× / 1.95×**
>   at n_obs=1,700/6,800/17,000. The old `_SERIES_INV_MIN_NOBS=3000` picked the *slower* path by
>   6.75× exactly where it was supposed to help.
> - **MPS-EM refusal bypassed and measured** (direct `EMBackend('mps')`): fp32 EM on wine (178×13)
>   took 1,002 iterations vs the fp64 CPU's 21 (11.1 s vs 41 ms) — the fixed-point stalls at the
>   fp32 noise floor — and on breast (569×30) it **fails outright** (E-step Cholesky loses
>   positive-definiteness in fp32). **The refusal stays**; its message now gives the real
>   (precision) reason instead of the stale launch-overhead one. Notably the per-iteration cost
>   (~11 ms) is fine now — precision, not dispatch, is the wall.
>
> **Changes shipped (working tree, uncommitted):** new `core.compute.device.mps_native_kernels()`
> predicate (torch ≥ 2.13); `_insertion_rank` uses native `searchsorted` on MPS behind it (bridge
> retained for older torch); `batched_bayes_linreg_draw` bypasses the series threshold behind it
> (older torch keeps the tuned crossover); EM refusal message corrected. CUDA paths untouched.
>
> **Gate results:**
> - **A/B end-to-end** (warm, min-of-3, identical data/process — the honest protocol; Phase-0
>   single-shot numbers included per-shape MPS shader compilation and overstate steady-state):
>   MICE PMM GPU on MPS **0.92 → 0.70 s at n=2k (1.31×), 0.88 → 0.79 s at n=8k (1.11×),
>   2.04 → 1.88 s at n=20k (1.09×)**. Gains concentrate at small n, as §6's calibration predicted.
> - **CUDA regression check:** MICE gate identical to baseline (0.26 / 0.64 s at n=8k/20k).
> - **Test suites:** CUDA (Forge) **4411 passed / 0 failed / 88 skipped**; macOS MPS
>   **4393 passed / 105 skipped / 1 failed** — the same pre-existing platform-libm ETS case as
>   Phase 0; zero regressions.
> - **pystatsbio `batch_auc`** re-evaluation spun off as its own task (sibling repo, Rule 8).
>
> **Reading for Phase 2:** the end-to-end MICE profile is no longer dominated by the ops Phase 1
> touched — 1.1–1.3× was the realistic ceiling here, and it was reached. The remaining headroom
> (per-step dispatch across ~30–60 ops, cross-op fusion) is exactly what `torch.compile` on the
> sweep step targets. Phase 2's go/no-go should profile first.

### Phase 2 — `torch.compile` the hot sweeps (weeks; medium risk, the big in-framework lever)

Apply `torch.compile` to (a) the MICE per-column step, (b) the MVNMLE batched objective/value+gradient evaluation, on both devices; CUDA additionally gets `mode="reduce-overhead"` (CUDA-Graph capture) trialed on the MVNMLE L-BFGS eval and the batched-Whittle Adam step.

Constraints that make this *engineering* rather than a flag-flip:
- **Fidelity (A6):** compiled-vs-eager numerics can differ at rounding level (fusion changes intermediate rounding). GPU paths are tolerance-tested (GPU_FP32 tier), so this is admissible — but it must be *verified* per path per device, and the compile choice disclosed in solution metadata (`backend_name`/info), never silent.
- **Slimness (A7):** torch stays optional; compile availability is a runtime capability check with a **loud** eager path (documented, not silent degradation — same doctrine as everywhere else).
- **Known sharp edges:** Inductor-Metal multistage-reduction bugs (#152155), `reduce-overhead` version sensitivity (#174575), dynamic-shape recompiles (MICE column widths vary — may need per-shape compilation caches or padding). If Inductor can't handle an op mix, that's a finding, not a failure — it routes Phase 3.

**Gate:** same benchmark matrix as Phase 1. Adopt where ≥1.3× and tolerance-clean; document; feed the numbers to §6.5.

### Phase 3 — Fused custom kernels, only where Phase 2 leaves measured dispatch on the table (weeks–months; highest cost, bounded upside)

`torch.mps.compile_shader` (MSL strings, zero new dependencies, no build toolchain) fusing the MVNMLE value+gradient chain — one command buffer per L-BFGS evaluation — and, if MICE still shows dispatch residue, the per-column step's gather/draw/scatter chain. CUDA equivalent: manual CUDA-Graph capture where `reduce-overhead` fell short.

Go/no-go is strictly evidence-based: the external calibration (BaseRT vs well-batched MLX: 1.01–1.35×) says fused-vs-well-compiled gains at compute-bound shapes are tens of percent — so Phase 3 only pays if Phase 2's profiles still show launch-bound sections. Do not start here; earn it with a profile.

**Gate:** ≥1.5× over the Phase-2 result on at least one paper-relevant shape, tolerance-clean on both devices, or it doesn't ship.

### 6.5 What this means for the two papers (the material part)

- **Both papers benchmark the code paths this plan changes.** The forward-Cholesky MVNMLE paper pins `pystatistics==6.0.1`; its Apple-silicon benchmarks were run on torch 2.13 (M2 Max) — so its numbers already sit on the *new* dispatch regime and are **not** invalidated by the torch upgrade itself. What changes them is Phases 1–3: any adopted change re-runs the paper's benchmark grid, and the pin moves to the release that carries the improvements. The GPU-MICE paper is more exposed: merge-rank retirement (Phase 1) and step compilation (Phase 2) directly alter the mechanism the paper describes, not just its timings — its methods section describes the workaround architecture, so adopted Phase-1/2 changes mean **prose revisions plus re-benchmarking**, not just new numbers.
- **The re-validation cascade, by phase:** Phase 1 is semantics-preserving (same estimator, same draws given the same seeds *except* the merge-rank ±1 tie convention, which was already refined away by the exact window search — verify with the existing GPU-vs-CPU equivalence tests). Phase 2 changes numerics at rounding level → full GPU tolerance suite + R-fidelity re-run on both devices, per path. Phase 3 replaces kernels → treat as a new backend: the full mandated backend test surface, both devices, plus the validation program's adversarial/Red-Team suites for the affected modules.
- **The "full rewrite" tripwire, stated precisely:** if Phase 2/3 measurements show ≥5–10× end-to-end at paper-relevant shapes — i.e., the op-DAG architecture itself, not its dispatch, was the binding constraint — then re-architecting the MICE/MVNMLE GPU backends around fused evaluation is justified, and that *is* a rewrite: every parity, tolerance, R-fidelity, and Red-Team suite re-runs, and both papers re-benchmark on the new architecture before submission. Current evidence puts the *probability* of tripping this at low-to-moderate for MICE small-n (where the dispatch share was genuinely dominant) and low for MVNMLE (already sync-fused and compute-bound at large p) — but that is exactly what the phase gates measure, and the decision should be made on their numbers, not this prediction.
- **Sequencing with the papers:** run Phase 0–1 *before* freezing either paper's benchmark tables (cheap, and Phase 1 alone may change MICE's headline MPS numbers); gate paper submission on whichever phase the measurements say is the point of diminishing returns.

### 6.6 Parked, with named triggers

GPU Cox (trigger: users at n≳10⁵ or a pystatsbio-scale survival need — the prefix-sum formulation and TorchSurv's Efron treatment are the blueprint); batched-ANOVA/ETS/EM APIs (trigger: a vertical needs mass-univariate or many-series fitting — cuML's 12×-for-1000-series is the model); Mojo (trigger: compiler open-sourced + interop GA + wheel story — any two of three); `torch.associative_scan` (trigger: leaves prototype status — currently CUDA-only, no autograd, slower than cumsum per PyTorch's own RFC).

### 6.7 Do not pursue

Triton (dominated); an MLX backend (adds a stack, lacks linalg/fp64); GPU versions of the 17 Cython kernels (contractually impossible under bit-identity, and §3.1 says pointless anyway); any Apple-GPU fp64 ambition (hardware-impossible).

---

## Appendix A — Machines and versions

| | Forge | Powerhouse |
|---|---|---|
| Hardware | RTX 5070 Ti (sm_120, 16 GB), Ryzen 7600X | Mac Studio M2 Max |
| Stack tested | torch 2.11.0.dev+cu128, CUDA 12.8, driver 580.173 | torch 2.12.0 **and 2.13.0** (MPS), MLX 0.32.0, scipy 1.17.1 |
| GPU state | idle before/after; scratch dirs removed | benchmark venvs in session scratchpad |

## Appendix B — Key primary sources

- Särkkä & García-Fernández, *Temporal Parallelization of Bayesian Smoothers*, IEEE TAC 66(1), 2021 (arXiv:1905.13002); Yaghoobi et al., SIAM J. Sci. Comput. 2023 (arXiv:2207.00426) — square-root parallel forms + O(log T) likelihood.
- Yang, Schuemie, Ji & Suchard, *Massive Parallelization of Massive Sample-size Survival Analysis*, JCGS 33(1), 2024 (PMC11070748) — GPU Cox 42–52×; Cyclops `gpu_cox` branch; TorchSurv (Novartis) production Efron-via-cumsum.
- Blelloch, *Prefix Sums and Their Applications*, CMU-CS-90-190 — first-order recurrences as scans.
- PyTorch issue #148219 (MPS dispatch decomposition); PyTorch 2.13 release blog (MPSGraph→native-Metal migration); Apple tech talk 10580 (batch-before-commit).
- ORNL/UTK SC'25 Workshops (arXiv:2509.21039) — Mojo GPU portability measurements; mojolang.org 1.0 release notes; Modular/Qualcomm announcements.
- Stan Math GPU paper (arXiv:1907.01063) — fused-kernel GLM crossovers; cuML batched ARIMA (RAPIDS blog); MAGMA batched tiny-matrix series (Dongarra group).
- In-repo: CONVENTIONS.md (threshold rule, deliberate no-GPU list), GPU_NOTES.md (MPS measurements, torch 2.10/2.11 era), CYTHON_MIGRATION_PROPOSAL.md (bit-identity contract, kernel audit), CHANGELOG.md 1.0.2→6.0.2 (formulation history).
