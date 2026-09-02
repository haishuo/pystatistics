"""Small implementation tests; no calibration or evaluation cell is timed."""

from __future__ import annotations

import unittest

import numpy as np

from benchmarks.dveb_arma_fit_phase2.backends import (
    DVEBFiniteDifferenceBackend,
    FD_CENTRAL,
    FD_FORWARD,
    TorchAutogradBackend,
)
from benchmarks.dveb_arma_fit_phase2.coordinator import DeterministicCoordinator
from benchmarks.dveb_arma_fit_phase2.fit import fit_coordinated


class QuadraticBackend:
    def __init__(self):
        self.traces = {}
        self.last_likelihood_rows = 0

    def value_and_gradient(self, rows, parameters):
        self.last_likelihood_rows = len(rows)
        for row, values in zip(rows, parameters, strict=True):
            self.traces.setdefault(row, []).append(tuple(values))
        residual = parameters - np.asarray([1.5, -0.25])
        return np.sum(residual * residual, axis=1), 2.0 * residual


class CoordinatorTests(unittest.TestCase):
    def test_independent_optimizers_converge_without_shared_state(self):
        starts = np.asarray([[0.0, 0.0], [3.0, 4.0], [-2.0, 1.0]])
        backend = QuadraticBackend()
        results, accounting = DeterministicCoordinator(backend).fit(starts)
        for result in results:
            self.assertTrue(result.success)
            np.testing.assert_allclose(result.x, [1.5, -0.25], atol=1e-8)
        self.assertEqual(set(backend.traces), {0, 1, 2})
        self.assertGreater(accounting.barrier_batches, 0)
        self.assertEqual(accounting.likelihood_rows, accounting.request_rows)

    def test_fd_identities_are_frozen(self):
        self.assertEqual((FD_FORWARD.policy_id, FD_FORWARD.step, FD_FORWARD.central), ("FD-F", 1e-8, False))
        self.assertEqual((FD_CENTRAL.policy_id, FD_CENTRAL.step, FD_CENTRAL.central), ("FD-C", 1e-5, True))

    def test_private_cpu_routes_fit_the_same_tiny_problem(self):
        rng = np.random.Generator(np.random.PCG64DXSM(123))
        z = np.ascontiguousarray(rng.standard_normal((2, 32)), dtype=np.float64)
        starts = np.ascontiguousarray([[0.2], [0.3]], dtype=np.float64)
        dveb = DVEBFiniteDifferenceBackend(z, "F1", FD_FORWARD, route="D1")
        torch_backend = TorchAutogradBackend(z, "F1", device="cpu", compiled=False)
        try:
            dveb_fit = fit_coordinated(z, starts, "F1", "D1", dveb)
            torch_fit = fit_coordinated(z, starts, "F1", "TC1", torch_backend)
        finally:
            dveb.close()
        self.assertTrue(dveb_fit.success.all())
        self.assertTrue(torch_fit.success.all())
        np.testing.assert_allclose(dveb_fit.parameters, torch_fit.parameters, atol=2e-7)
        np.testing.assert_allclose(dveb_fit.nll, torch_fit.nll, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
