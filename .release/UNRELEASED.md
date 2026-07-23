# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Docs: fixed malformed reStructuredText in numerous docstrings so the
  Sphinx API reference renders cleanly. Example blocks, algorithm
  listings, equations, and option lists in ``pca``, ``boot``, ``multinom``,
  ``mlest``, ``mlest_monotone_closed_form``, ``adf_test``, ``pacf``,
  ``Gaussian.aic``, ``HTestSolution.summary`` and the ``timeseries``
  package overview are now proper literal blocks and lists instead of
  being silently dropped or misindented. Also removed duplicate API
  entries (``DataSource``, time-series result attributes) and ambiguous
  cross-references from shape annotations. No behavior, signature, or
  documented-value changes — docstring formatting only. The docs now
  build warning-free under ``sphinx-build -W``.
