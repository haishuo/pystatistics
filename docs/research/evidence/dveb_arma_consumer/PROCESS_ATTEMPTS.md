# Process attempts

## Qualification attempt 1 — infrastructure halt

The first launch of `benchmarks/dveb_arma_consumer/qualify.py` stopped before
loading either DVEB artifact or evaluating any numerical candidate. Executing
the file directly placed its own directory, but not the repository root, on
`sys.path`; importing the frozen Cython authority consequently raised
`ModuleNotFoundError: No module named 'pystatistics'`.

The correction adds the repository root to the harness import path. It changes
no artifact, input, oracle, tolerance, schedule, or qualification rule. No
qualification evidence file existed before the corrected restart.

## Regression attempt 1 — incomplete baseline inventory

The first launch of `benchmarks/dveb_arma_consumer/run_regression.py` completed
the focused time-series suite and both offline evidence verifiers, then ran the
complete test suite.  It stopped as designed because the observed failure set
contained two tests while the frozen runner's baseline inventory named only
one.  The complete-suite result was 2 failed, 4,485 passed, 94 skipped, and 27
deselected.  No `regression.json` was written.

The known multinomial complete-separation failure was already reproduced and
recorded at `a9c7e4d`.  The additional failure was
`tests/descriptive/test_gpu.py::TestGPUvsCPU::test_describe_kurtosis`: one of
five values differed by 6.79605395e-07 (relative difference 0.00017979) against
the test's `rtol=1e-4` and zero absolute tolerance.

Before restarting, that exact isolated test was run in a temporary detached
worktree at the frozen pre-adapter commit `f156ee5`, using the same Python,
dependencies, GPU, and ABI-compatible prebuilt Cython extensions.  It failed
with the identical actual and desired arrays, identical mismatched element,
and identical absolute and relative differences.  The same isolated test at
the current adapter commit produced the identical output.  This establishes
the kurtosis test as inherited baseline behavior rather than a DVEB-adapter
regression.

The regression runner's baseline inventory is therefore corrected to contain
both exactly named tests.  This does not relax the frozen rule: the criterion
remains no new full-suite failure relative to `f156ee5`.  No adapter,
artifact, numerical tolerance, public API, qualification result, or test was
changed before the corrected restart.
