# Campaign v1 invalidation record

The first consumer campaign is permanently excluded from the DVEB MVN-MLE
decision. Its raw observations, analysis, operational probes, qualification,
and freeze are preserved with `v1` in their filenames.

After all v1 timing completed, the evidence verifier found that E2-T12's
recorded data SHA-256 was
`99b114e968c790ddc3effd7d0d8fd10229cc397de2f95f3dffb950ba7ad5e7bf`
rather than the frozen E2 SHA-256
`1f080b40b4c71d17bcfa414ea7081f2ac967f94373679a42e689705d6c82fcb7`.
The other 17 lanes matched their frozen bytes.

This was not caller-input mutation. A fresh hash-after-every-fit diagnostic
showed both solvers preserve the supplied array. The divergence occurs before
fitting: importing NumPy, then configuring PyTorch for 12 threads, then calling
NumPy's `Generator.multivariate_normal` changes six of E2's 1,945 finite input
values by one ULP (maximum absolute difference
`2.220446049250313e-16`). Both implementations received the same altered array
within E2-T12, and all v1 correctness comparisons passed, but the lane did not
use the byte-identical frozen input required by the campaign freeze.

Therefore v1's apparent GO cannot be used. No threshold, evaluation point,
repeat count, solver, artifact, schedule, or decision rule changes. The sole
v2 correction is to generate and hash each frozen input before importing or
configuring PyTorch, and to verify the hash again after all fits. The entire
18-lane campaign must be frozen and rerun; no v1 timing may enter the v2
decision.
