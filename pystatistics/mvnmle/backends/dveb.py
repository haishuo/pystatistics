"""Research-only DVEB forward-Cholesky backend for direct MVN MLE."""

from __future__ import annotations

from dataclasses import replace

from pystatistics.core.result import Result
from pystatistics.mvnmle._dveb.objective import DVEBDenseObjective
from pystatistics.mvnmle.backends._direct import run_direct_solve
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams


class DVEBDenseBackend:
    """Explicit CPU-only consumer of the qualified DVEB dense ABI."""

    def __init__(self, *, threads: int | None = None, schedule: int = 0):
        self.threads = threads
        self.schedule = schedule

    @property
    def name(self) -> str:
        return "cpu_dveb_cholesky_fp64"

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str | None = None,
        tol: float | None = None,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        objective_holder: dict[str, DVEBDenseObjective] = {}

        def factory() -> DVEBDenseObjective:
            objective = DVEBDenseObjective(
                design.data,
                threads=self.threads,
                schedule=self.schedule,
            )
            objective_holder["objective"] = objective
            return objective

        result = run_direct_solve(
            factory,
            method=method or "BFGS",
            tol=tol if tol is not None else 1e-5,
            max_iter=max_iter,
            backend_name=self.name,
            parameterization="cholesky",
            device="cpu",
            precision="fp64",
            sync_cuda=False,
        )
        objective = objective_holder["objective"]
        info = dict(result.info)
        info.update(
            {
                "dveb_abi_version": 1,
                "dveb_artifact_sha256": objective.artifact_sha256,
                "dveb_threads": objective.threads,
                "dveb_schedule_override": objective.schedule,
                "dveb_selected_schedule": objective.last_selected_schedule,
                "dveb_scratch_bytes": objective.scratch_bytes,
            }
        )
        return replace(result, info=info)
