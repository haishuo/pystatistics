"""
Initial states for ETS model fitting.

Two jobs, both about the state vector at time zero and nothing else:

* **Heuristic starting values** (``_init_level_trend``, ``_init_season``)
  — data-driven estimates of the initial level/trend/seasonal states used
  to seed the optimiser, following R ``forecast::ets``'s ``initstate``
  scheme (first-period mean for seasonal level, cross-period slope for
  trend, classical-decomposition seasonal indices normalised to
  ``sum = 0`` additive / ``sum = m`` multiplicative).
* **Expansion of the optimiser's free states** (``_assemble_init_states``)
  — seasonal models optimise ``m - 1`` free initial seasonal states (as R
  does); the remaining one is reconstructed from the normalisation so
  every optimiser iterate satisfies it exactly.

Consumed by ``_ets_fit.py``; the optimiser, likelihood, and parameter
transforms live there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.timeseries._ets_models import ETSSpec


def _assemble_init_states(free_states: NDArray, spec: ETSSpec) -> NDArray:
    """Expand the optimiser's free initial states to the full state vector.

    Seasonal models optimise ``m - 1`` initial seasonal states (as R
    forecast::ets does); the remaining one — the index used at the first
    observation — is determined by the normalisation ``sum(s) = 0``
    (additive) / ``sum(s) = m`` (multiplicative).
    """
    if spec.season == "N":
        return np.asarray(free_states, dtype=np.float64)
    n_lead = 1 + (1 if spec.trend in ("A", "Ad") else 0)
    lead = free_states[:n_lead]
    s_free = free_states[n_lead:]
    target = 0.0 if spec.season == "A" else float(spec.period)
    s_first = target - float(np.sum(s_free))
    return np.concatenate([lead, [s_first], s_free])


def _init_level_trend(y: NDArray, spec: ETSSpec) -> tuple[float, float | None]:
    """
    Estimate initial level and trend from the data.

    For non-seasonal models, uses the first 10 observations (or fewer).
    For seasonal models, uses the mean of the first complete period for
    level and a simple slope for trend.

    Parameters
    ----------
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.

    Returns
    -------
    tuple
        ``(level, trend_or_None)``
    """
    m = spec.period
    has_season = spec.season in ("A", "M")

    if has_season and len(y) >= 2 * m:
        level = float(np.mean(y[:m]))
    else:
        k = min(10, len(y))
        level = float(np.mean(y[:k]))

    trend: float | None = None
    if spec.trend in ("A", "Ad"):
        if has_season and len(y) >= 2 * m:
            # Average slope across first two periods
            trend = float(np.mean(y[m : 2 * m] - y[:m]) / m)
        else:
            k = min(10, len(y))
            if k >= 2:
                trend = float((y[k - 1] - y[0]) / (k - 1))
            else:
                trend = 0.0

    return level, trend


def _init_season(y: NDArray, spec: ETSSpec, level: float) -> NDArray | None:
    """
    Estimate initial seasonal indices via classical decomposition.

    Parameters
    ----------
    y : NDArray
        Time series.
    spec : ETSSpec
        Model specification.
    level : float
        Initial level estimate.

    Returns
    -------
    NDArray or None
        Seasonal indices of length ``period``, or ``None`` if no season.
    """
    if spec.season == "N":
        return None

    m = spec.period
    n_full = min(len(y), 3 * m)
    y_sub = y[:n_full]

    if spec.season == "A":
        # Additive: s_i = mean(y[i::m]) - level
        season = np.array([float(np.mean(y_sub[i::m])) - level for i in range(m)])
        # Centre so they sum to zero
        season -= np.mean(season)
    else:
        # Multiplicative: s_i = mean(y[i::m]) / level
        if abs(level) < 1e-15:
            season = np.ones(m)
        else:
            season = np.array(
                [float(np.mean(y_sub[i::m])) / level for i in range(m)]
            )
            # Normalise so product = 1 (equivalent: mean = 1)
            season *= m / np.sum(season)

    return season
