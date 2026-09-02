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
