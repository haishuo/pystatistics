# DVEB batched exact-ARMA Phase-0B preflight result

**Outcome: PREFLIGHT PASS; FORMAL PHASE-0B DECISION DEFERRED.**

The consumer-side workload/null preflight establishes that exact batched ARMA
likelihood evaluation is expressible in ordinary public PyTorch operations on
both CPU and CUDA, with excellent agreement to the existing Cython authority.
It does **not** establish whether PyTorch is near the native performance ceiling.
No performance observations were collected, no native diagnostic exists, and
neither Case 1 nor Case 2 can therefore be awarded under the frozen protocol.

## Scope and integrity

The protocol was frozen at `bb8b84c` before implementation or observations.
Inputs use four stable parameter families across four disjoint calibration and
twelve evaluation cells and are frozen by byte hash. The formal source freeze
at `eae1923` records zero decision observations and explicitly says that neither
a native diagnostic nor DVEB compiler work was authorized.

This phase changed no PyStatistics API or backend, no DVEB source, and no
publication-facing branch. It introduced research-only comparator and evidence
files on `research/dveb-pystatistics`.

## Numerical result

The compiled structured-loop formulation passed every frozen cell on both
devices:

| Gate | CPU | CUDA |
|---|---:|---:|
| Structured calibration + evaluation cells | 16/16 | 16/16 |
| Eager smoke cells C01--C02 | 2/2 | 2/2 |
| Persistent/diffuse/invalid/shape cases | 4/4 | 4/4 |
| Caller inputs unchanged | 18/18 | 18/18 |
| Exact status agreement | 18/18 | 18/18 |

The worst structured-loop difference from the Cython implementation occurred
at E09 (`K=64`, `n=10000`, `r=3`):

- NLL absolute: `3.8198777474462986e-11`;
- NLL relative: `2.6794973249455973e-15`;
- sigma2 absolute: `7.993605777301127e-15`;
- sigma2 relative: `7.888478041865127e-15`.

All are far inside the preflight's `1e-9` relative rejection boundary. The
focused existing time-series regression suite also passed: **153 passed**.

## PyTorch control-flow finding

The literal Python time loop works eagerly and passed the C01--C02 smoke
checks, but `torch.compile(fullgraph=True)` did not finish compiling the
smallest C01 shape within three minutes and was interrupted. That is preserved
as an operational failure, not a performance result.

The public structured-control-flow formulation required one ordinary source
correction: tensor time indexing had to use `torch.index_select` rather than
Python-style `z[:, t]`. After that correction, `torch.while_loop` captured the
whole workload on both devices:

| Device | Graphs | Graph breaks | Structured while loops | Outer graph ops |
|---|---:|---:|---:|---:|
| CPU | 1 | 0 | 2 | 45 |
| CUDA | 1 | 0 | 2 | 45 |

This is positive evidence for the ordinary-PyTorch null hypothesis. It is not
yet a production recommendation: PyTorch 2.13 explicitly labels
`torch.while_loop` a **prototype** feature with limited type support and no
training support.

The one-row input generator also exposed a NumPy ownership edge case:
`ascontiguousarray` may retain a read-only broadcast when `K=1`. The generator
was corrected to force owning writable arrays. All numerical bytes and hashes
remained unchanged. These pre-decision failures are retained in
`benchmarks/dveb_arma_phase0b/PROCESS_ATTEMPTS.md`, along with the harmless
post-preflight test-runner invocation correction.

## What the result means

This preflight rules out a simple claim that PyTorch cannot represent the exact
recurrence. It does not answer the actual native-headroom question:

- there are no L1--L4 timings;
- there is no competent multicore native batch comparator;
- there is no separately authored native CUDA implementation;
- no memory, launch, register, or kernel-structure comparison has been made;
- no Case-1/Case-2 threshold has been evaluated.

The next scientifically valid step, if separately authorized, is to implement
the protocol's NC and NG diagnostic ceilings from the frozen mathematical
contract, admit them through correctness first, and only then freeze and run
the prospective performance campaign. If native code shows no qualifying
headroom, the result is Case 1 and DVEB does not receive an ARMA trunk. If it
does, the missing general DVEB mechanisms can be identified as Case 2 without
conflating PyTorch source, generated code, and the underlying algorithm.

## Evidence and verification

- protocol: `docs/research/DVEB_ARMA_PHASE0B_PROTOCOL.md`;
- input/source freezes: `benchmarks/dveb_arma_phase0b/input-freeze.json` and
  `benchmarks/dveb_arma_phase0b/preflight-freeze.json`;
- raw CPU/CUDA evidence and summary:
  `docs/research/evidence/dveb_arma_phase0b_preflight/`;
- offline verifier: `benchmarks/dveb_arma_phase0b/verify_preflight.py`.

Run the verifier from the repository root with any Python 3.11+ interpreter:

```bash
python3 benchmarks/dveb_arma_phase0b/verify_preflight.py
```

It uses only the standard library, verifies the frozen source and input hashes,
checks every raw evidence hash, replays all admission invariants, and confirms
that the formal decision remains deferred with zero performance observations.
