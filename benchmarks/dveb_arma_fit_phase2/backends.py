"""Private Phase-II likelihood/gradient backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .common import expand_parameters


@dataclass(frozen=True)
class FiniteDifferencePolicy:
    policy_id: str
    step: float
    central: bool


FD_FORWARD = FiniteDifferencePolicy("FD-F", 1.0e-8, False)
FD_CENTRAL = FiniteDifferencePolicy("FD-C", 1.0e-5, True)


def expand_fit_parameters(parameters, family_id):
    if family_id != "P11":
        return expand_parameters(parameters, family_id)
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"P11 expects (*,2) free parameters, got {values.shape}")
    phi = np.zeros((values.shape[0], 2), dtype=np.float64)
    loading = np.zeros_like(phi)
    phi[:, 0] = values[:, 0]
    loading[:, 0] = 1.0
    loading[:, 1] = values[:, 1]
    return phi, loading


def fit_layout(family_id):
    if family_id == "P11":
        return 2, (0,), (0,)
    from .common import FIT_FAMILIES
    spec = FIT_FAMILIES[family_id]
    return spec.state, spec.ar_free, spec.ma_free


class DVEBFiniteDifferenceBackend:
    """Expand all live optimizers' perturbations into one DVEB ABI batch."""

    def __init__(
        self,
        z: NDArray[np.float64],
        family_id: str,
        policy: FiniteDifferencePolicy,
        *,
        route: str,
        combined_library: Path | None = None,
    ):
        from pystatistics.timeseries._dveb_arma import (
            DVEBCPUExactArma,
            DVEBCudaTransferExactArma,
        )
        from pystatistics.timeseries._dveb_arma.loader import CPU_ITEM_PARALLEL, CPU_SERIAL

        self.z = np.ascontiguousarray(z, dtype=np.float64)
        self.family_id = family_id
        self.policy = policy
        self.route = route
        self.last_likelihood_rows = 0
        if route == "D1":
            schedule = CPU_SERIAL if z.shape[0] == 1 else CPU_ITEM_PARALLEL
            self.evaluator = DVEBCPUExactArma(max_threads=12, schedule=schedule)
            self.threads = 1 if z.shape[0] == 1 else 12
        elif route == "D2":
            if combined_library is None:
                raise ValueError("D2 requires the explicit qualified combined library")
            # One request can require base plus two perturbations per parameter.
            maximum_rows = 256 * (1 + 2 * 5)
            self.evaluator = DVEBCudaTransferExactArma(
                max_items=maximum_rows,
                max_steps=z.shape[1],
                max_state=25,
                library_path=combined_library,
            )
            self.threads = None
        else:
            raise ValueError(f"unknown DVEB route {route!r}")

    def close(self) -> None:
        self.evaluator.close()

    def _evaluate(self, z, parameters):
        phi, loading = expand_fit_parameters(parameters, self.family_id)
        if self.route == "D1":
            return self.evaluator.evaluate(z, phi, loading, threads=self.threads)
        return self.evaluator.evaluate(z, phi, loading, block=0)

    def value_and_gradient(self, rows, parameters):
        row_index = np.asarray(rows, dtype=np.intp)
        source = self.z[row_index]
        count, width = parameters.shape
        if self.policy.central:
            copies = 1 + 2 * width
            expanded = np.repeat(parameters, copies, axis=0)
            expanded_z = np.repeat(source, copies, axis=0)
            for row in range(count):
                base = row * copies
                for column in range(width):
                    expanded[base + 1 + 2 * column, column] += self.policy.step
                    expanded[base + 2 + 2 * column, column] -= self.policy.step
        else:
            copies = 1 + width
            expanded = np.repeat(parameters, copies, axis=0)
            expanded_z = np.repeat(source, copies, axis=0)
            for row in range(count):
                base = row * copies
                for column in range(width):
                    expanded[base + 1 + column, column] += self.policy.step
        values, _sigma2, _status = self._evaluate(
            np.ascontiguousarray(expanded_z), np.ascontiguousarray(expanded)
        )
        base_values = values[::copies].copy()
        gradients = np.empty_like(parameters)
        for row in range(count):
            base = row * copies
            for column in range(width):
                if self.policy.central:
                    plus_index = base + 1 + 2 * column
                    minus_index = base + 2 + 2 * column
                    displacement = expanded[plus_index, column] - expanded[minus_index, column]
                    gradients[row, column] = (values[plus_index] - values[minus_index]) / displacement
                else:
                    plus_index = base + 1 + column
                    displacement = expanded[plus_index, column] - expanded[base, column]
                    gradients[row, column] = (values[plus_index] - values[base]) / displacement
        self.last_likelihood_rows = expanded.shape[0]
        return base_values, gradients


class TorchAutogradBackend:
    """Ordinary public-op PyTorch likelihood with batched independent gradients."""

    def __init__(self, z, family_id: str, *, device: str, compiled: bool):
        import torch
        from benchmarks.dveb_arma_phase0b.torch_impl import (
            likelihood_for_loop,
            likelihood_while_loop,
        )

        if device == "cpu":
            torch.set_num_threads(12)
            if torch.get_num_interop_threads() != 1:
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Phase-II requires PyTorch inter-op threads=1 before parallel "
                        "work starts; create the CPU backend before any other torch work"
                    ) from exc
        self.torch = torch
        self.family_id = family_id
        self.device = torch.device(device)
        self.z = torch.as_tensor(z, dtype=torch.float64, device=self.device)
        self.compiled = bool(compiled)
        self.function = (
            torch.compile(likelihood_while_loop, fullgraph=True)
            if compiled else likelihood_for_loop
        )
        self.last_likelihood_rows = 0

    def value_and_gradient(self, rows, parameters):
        torch = self.torch
        index = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        z = torch.index_select(self.z, 0, index)
        theta = torch.tensor(
            parameters, dtype=torch.float64, device=self.device, requires_grad=True
        )
        spec_phi, spec_loading = expand_fit_parameters(parameters, self.family_id)
        # Constant templates carry fixed zeros/ones. CopySlice preserves the
        # gradient connection from each free theta column.
        phi = torch.zeros(spec_phi.shape, dtype=torch.float64, device=self.device)
        loading = torch.zeros(spec_loading.shape, dtype=torch.float64, device=self.device)
        loading[:, 0] = 1.0
        _state, ar_free, ma_free = fit_layout(self.family_id)
        split = len(ar_free)
        for column, lag in enumerate(ar_free):
            phi[:, lag] = theta[:, column]
        for column, lag in enumerate(ma_free, start=split):
            loading[:, lag + 1] = theta[:, column]
        nll, _sigma2, _status = self.function(z, phi, loading)
        gradient = torch.autograd.grad(nll.sum(), theta)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_likelihood_rows = len(rows)
        return nll.detach().cpu().numpy(), gradient.detach().cpu().numpy()
