"""Design-identity tests frozen before Phase-II fitter implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE.parent)]

from benchmarks.dveb_arma_fit_phase2.common import (  # noqa: E402
    CALIBRATION_CELLS,
    EVALUATION_CELLS,
    FIT_FAMILIES,
    biased_yule_walker,
    expand_parameters,
    generate_cell,
    generate_starts,
    minimum_ar_root,
    public_airpassengers,
    truth_vector,
)
from benchmarks.dveb_arma_fit_phase2.coordinator_model import (  # noqa: E402
    barrier_rounds,
    consecutive_chunks,
)
from pystatistics.timeseries._arima_fit import _yule_walker_start  # noqa: E402


class ParameterIdentityTests(unittest.TestCase):
    def test_truth_expands_to_phase0_coefficients(self):
        for cell_id in CALIBRATION_CELLS + EVALUATION_CELLS:
            z, expected_phi, expected_loading = generate_cell(cell_id)
            family_id = next(
                family_id
                for family_id, family in FIT_FAMILIES.items()
                if family.state == expected_phi.shape[1]
            )
            phi, loading = expand_parameters(truth_vector(family_id), family_id)
            self.assertTrue(np.array_equal(phi[0], expected_phi[0]))
            self.assertTrue(np.array_equal(loading[0], expected_loading[0]))
            self.assertTrue(z.flags.c_contiguous)

    def test_yule_walker_matches_existing_single_series_helper(self):
        for cell_id in ("C01", "C02", "C03", "C04"):
            z, phi, _ = generate_cell(cell_id)
            actual = biased_yule_walker(z[0], phi.shape[1])
            expected = _yule_walker_start(z[0], phi.shape[1])
            self.assertTrue(np.array_equal(actual, expected))

    def test_starts_are_stationary_and_ma_is_zero(self):
        for cell_id in CALIBRATION_CELLS + EVALUATION_CELLS:
            z, phi, _ = generate_cell(cell_id)
            family_id = next(
                family_id for family_id, family in FIT_FAMILIES.items()
                if family.state == phi.shape[1]
            )
            starts = generate_starts(cell_id)
            self.assertEqual(starts.shape[0], z.shape[0])
            split = len(FIT_FAMILIES[family_id].ar_free)
            self.assertTrue(np.array_equal(starts[:, split:], np.zeros_like(starts[:, split:])))
            self.assertGreaterEqual(
                min(minimum_ar_root(row, family_id) for row in starts), 1.001
            )

    def test_public_fixture_is_exactly_transformed(self):
        values = public_airpassengers()
        self.assertEqual(values.shape, (143,))
        self.assertAlmostEqual(float(values.mean()), 0.0, places=16)
        self.assertTrue(np.isfinite(values).all())


class CoordinatorIdentityTests(unittest.TestCase):
    def test_chunks_are_consecutive_and_bounded(self):
        chunks = consecutive_chunks(1024)
        self.assertEqual(tuple(map(len, chunks)), (256, 256, 256, 256))
        self.assertEqual(tuple(item for chunk in chunks for item in chunk), tuple(range(1024)))

    def test_barrier_waits_for_every_live_worker(self):
        rounds = barrier_rounds((("a0", "a1"), ("b0",), ("c0", "c1", "c2")))
        self.assertEqual(
            rounds,
            (
                ((0, "a0"), (1, "b0"), (2, "c0")),
                ((0, "a1"), (2, "c1")),
                ((2, "c2"),),
            ),
        )

    def test_barrier_preserves_each_independent_trace(self):
        traces = tuple(tuple(f"{worker}:{step}" for step in range(worker % 5)) for worker in range(19))
        recovered = [[] for _ in traces]
        for batch in barrier_rounds(traces):
            self.assertEqual(tuple(worker for worker, _ in batch), tuple(sorted(worker for worker, _ in batch)))
            for worker, request in batch:
                recovered[worker].append(request)
        self.assertEqual(tuple(map(tuple, recovered)), traces)


if __name__ == "__main__":
    unittest.main()
