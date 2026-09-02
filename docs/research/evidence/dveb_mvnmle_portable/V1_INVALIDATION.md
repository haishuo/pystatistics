# Portable MVN-MLE campaign v1 invalidation

V1 stopped before timing. All 18 untimed installed-wheel lanes passed their
solver-to-solver numerical comparisons, but every regenerated E1--E6 array
failed the protocol's requirement to byte-match the older native campaign.

The cause is environmental, not numerical: the frozen generator uses NumPy's
multivariate-normal linear algebra, and the portable comparator environment
uses NumPy 2.2.6 while the older campaign used NumPy 2.3.5. Each case was
internally byte-identical across 1, 6, and 12 threads, but none matched the old
cross-version hash. No portable performance observation existed when this was
detected.

The complete invalid untimed result is retained as
`fit-qualification-v1-invalid.json`. V1 contributes no correctness or timing
evidence to the final campaign. Its protocol, freeze, thresholds, and failure
remain unchanged.

V2 changes only input materialization: it commits the six arrays produced by
the already frozen generator/specification in the current comparator
environment and loads those exact bytes for every qualification and timing
lane. Grid, implementations, environment, tolerances, warmups, repetitions,
randomization, and decision thresholds are unchanged.
