"""
PyTorch forward-Cholesky backend for direct MVN MLE.

Drives CUDA (FP32/FP64), MPS (FP32 only), and a CPU torch device (FP64 — the
fast default CPU path). Uses PyTorch autodiff for analytical gradients.
"""

import warnings

from pystatistics.core.compute.device import mps_misread_status
from pystatistics.core.result import Result
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle.backends._direct import run_direct_solve


# The same threshold MICE uses (mice/backends/gpu.py). The misread is
# triggered by caching-allocator state after prior large allocations, not by
# any one operation's size, so the honest predictor is problem scale rather
# than a property of the matmul itself.
_MPS_MISREAD_WARN_N = 20_000


def _warn_if_mps_misread(device: str, n: int, p: int) -> None:
    """Warn -- loudly, once -- before a survey-scale MVN MLE fit on an MPS
    build of torch carrying the allocator-state matmul misread (silently wrong
    results; docs/GPU_NOTES.md, "MPS strided-matmul buffer corruption").

    MVNMLE reaches the defect the same way MICE does. The FP32 objective forms
    ``sigma = L @ L.T`` (``_objectives/gpu_fp32.py``), whose second operand is
    a transposed -- therefore strided -- view, which is exactly the shape that
    misreads. Until this existed, MICE was guarded and MVNMLE was not, though
    both run on the same backend against the same torch builds.

    A warning rather than an error, for the reason MICE gives: moderate-n MPS
    results are validated and a user may knowingly accept the risk. Not raised
    for 'mitigated' torch builds -- installing one is the supported opt-in path
    to silence it.
    """
    if device != "mps" or n < _MPS_MISREAD_WARN_N:
        return
    if mps_misread_status() != "affected":
        return
    import torch

    warnings.warn(
        f"mvnmle on Apple Silicon (MPS) with torch {torch.__version__} at "
        f"n={n}, p={p}: this torch build can return silently wrong GPU "
        f"results at this scale (an upstream torch bug in MPS memory "
        f"management; the FP32 objective's `L @ L.T` is a strided matmul, the "
        f"shape that misreads). Results may be wrong without any error. "
        f"Remedies: use backend='cpu', run on CUDA, or install torch >= 2.14 "
        f"(pre-release: a nightly >= 2.14.0.dev20260624), where the corruption "
        f"no longer reproduces. To test your own machine/torch combination, "
        f"run pystatistics.mice.diagnostics.mps_matmul_canary().",
        UserWarning,
        stacklevel=3,
    )


class DirectMLEBackend:
    """
    PyTorch forward-Cholesky backend for direct MVN MLE.

    Device-parametrised. Despite the historical name, this backend also drives
    the fast CPU path: a CPU torch device with FP64 runs the same
    forward-Cholesky estimator and is the default CPU backend (it beats the
    numpy inverse-Cholesky reference substantially while matching R to ~1e-9).

    Precision is selected by device:
    - CPU torch device: FP64 (the fast default CPU path)
    - MPS: FP32 only (MPS doesn't support FP64)
    - CUDA consumer (RTX): FP32 by default
    - CUDA data center (A100/H100): FP64 available

    Args:
        device: 'cpu', 'cuda', 'mps', or 'auto'
        use_fp64: Force FP64 (raises on MPS, which has no FP64)
    """

    def __init__(self, device: str = 'auto', use_fp64: bool = False):
        import torch

        self._torch = torch
        self._use_fp64 = use_fp64

        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'
            else:
                from pystatistics.core.compute.backend import NO_GPU_MSG
                raise RuntimeError(NO_GPU_MSG)
        else:
            self._device = device

        # Validate MPS + FP64 conflict
        if self._device == 'mps' and use_fp64:
            from pystatistics.core.compute.backend import FP64_REQUIRES_CUDA_MSG
            raise RuntimeError(FP64_REQUIRES_CUDA_MSG)

    @property
    def name(self) -> str:
        precision = 'fp64' if self._use_fp64 else 'fp32'
        # A CPU torch device is a CPU backend, not a GPU one — report it
        # honestly so ``backend_name`` is truthful on torch-free-GPU machines.
        if self._device == 'cpu':
            return f'cpu_cholesky_{precision}'
        return f'gpu_{self._device}_{precision}'

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str | None = None,
        tol: float | None = None,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        """
        Solve MVN MLE using GPU.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper
        method : str or None
            Optimization method. Auto-selected based on precision.
        tol : float or None
            Convergence tolerance. Auto-selected based on precision.
        max_iter : int
            Maximum iterations

        Returns
        -------
        Result[MVNParams]
        """
        _warn_if_mps_misread(self._device, design.n, design.p)

        # Select objective class and defaults based on precision
        if self._use_fp64:
            from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64
            ObjectiveClass = GPUObjectiveFP64
            default_method = 'BFGS'
            default_tol = 1e-5
        else:
            from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32
            ObjectiveClass = GPUObjectiveFP32
            default_method = 'L-BFGS-B'
            default_tol = 1e-3

        method = method or default_method
        tol = tol or default_tol
        device = self._device

        return run_direct_solve(
            lambda: ObjectiveClass(design.data, device=device),
            method=method,
            tol=tol,
            max_iter=max_iter,
            backend_name=self.name,
            parameterization='cholesky',
            device=device,
            precision='fp64' if self._use_fp64 else 'fp32',
            sync_cuda=(device == 'cuda'),
        )
