"""Ordinary public-op PyTorch formulations for exact ARMA likelihood batches."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

KAPPA = 1.0e6
PENALTY = 1.0e18


def _transition(phi: Tensor) -> Tensor:
    _, r = phi.shape
    dtype, device = phi.dtype, phi.device
    column_zero = torch.zeros(r, dtype=dtype, device=device)
    column_zero[0] = 1.0
    transition = phi.unsqueeze(2) * column_zero.view(1, 1, r)
    if r > 1:
        transition = transition + torch.diag(
            torch.ones(r - 1, dtype=dtype, device=device), diagonal=1
        ).unsqueeze(0)
    return transition


def stationary_for_loop(phi: Tensor, loading: Tensor) -> tuple[Tensor, Tensor]:
    """Batched doubling with masked per-series convergence."""
    k, r = phi.shape
    transition = _transition(phi)
    s = loading.unsqueeze(2) * loading.unsqueeze(1)
    a = transition
    active = torch.ones(k, dtype=torch.bool, device=phi.device)
    converged = torch.zeros_like(active)

    for _ in range(60):
        u = torch.matmul(torch.matmul(a, s), a.transpose(1, 2))
        new_s = s + u
        finite = torch.isfinite(u).all(dim=(1, 2)) & torch.isfinite(new_s).all(dim=(1, 2))
        scale = new_s.abs().amax(dim=(1, 2))
        u_max = u.abs().amax(dim=(1, 2))
        just_converged = (
            active & finite & (u_max <= 1.0e-13 * torch.maximum(torch.ones_like(scale), scale))
        )
        usable = active & finite
        s = torch.where(usable[:, None, None], new_s, s)
        converged = converged | just_converged
        active = active & finite & ~just_converged
        next_a = torch.matmul(a, a)
        finite_a = torch.isfinite(next_a).all(dim=(1, 2))
        a = torch.where((active & finite_a)[:, None, None], next_a, a)
        active = active & finite_a

    stationary = 0.5 * (s + s.transpose(1, 2))
    eye = torch.eye(r, dtype=phi.dtype, device=phi.device).expand(k, r, r)
    p = torch.where(converged[:, None, None], stationary, KAPPA * eye)
    return p, converged


def stationary_while_loop(phi: Tensor, loading: Tensor) -> tuple[Tensor, Tensor]:
    """The same doubling recurrence represented by public structured control flow."""
    k, r = phi.shape
    transition = _transition(phi)
    s = loading.unsqueeze(2) * loading.unsqueeze(1)
    active = torch.ones(k, dtype=torch.bool, device=phi.device)
    converged = torch.zeros_like(active)
    iteration = torch.zeros((), dtype=torch.int64, device=phi.device)

    def cond(it: Tensor, _s: Tensor, _a: Tensor, live: Tensor, _done: Tensor) -> Tensor:
        return (it < 60) & torch.any(live)

    def body(
        it: Tensor, current_s: Tensor, current_a: Tensor, live: Tensor, done: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        u = torch.matmul(torch.matmul(current_a, current_s), current_a.transpose(1, 2))
        new_s = current_s + u
        finite = torch.isfinite(u).all(dim=(1, 2)) & torch.isfinite(new_s).all(dim=(1, 2))
        scale = new_s.abs().amax(dim=(1, 2))
        u_max = u.abs().amax(dim=(1, 2))
        just_converged = (
            live & finite & (u_max <= 1.0e-13 * torch.maximum(torch.ones_like(scale), scale))
        )
        usable = live & finite
        selected_s = torch.where(usable[:, None, None], new_s, current_s)
        next_done = done | just_converged
        next_live = live & finite & ~just_converged
        squared_a = torch.matmul(current_a, current_a)
        finite_a = torch.isfinite(squared_a).all(dim=(1, 2))
        selected_a = torch.where((next_live & finite_a)[:, None, None], squared_a, current_a)
        next_live = next_live & finite_a
        return it + 1, selected_s, selected_a, next_live, next_done

    _, s, _, _, converged = torch.while_loop(
        cond, body, (iteration, s, transition, active, converged)
    )
    stationary = 0.5 * (s + s.transpose(1, 2))
    eye = torch.eye(r, dtype=phi.dtype, device=phi.device).expand(k, r, r)
    p = torch.where(converged[:, None, None], stationary, KAPPA * eye)
    return p, converged


def _kalman_step(
    z_t: Tensor,
    phi: Tensor,
    loading_outer: Tensor,
    state: Tensor,
    covariance: Tensor,
    sse: Tensor,
    sum_log_f: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _, r = phi.shape
    innovation = z_t - state[:, 0]
    f = covariance[:, 0, 0]
    step_valid = valid & torch.isfinite(f) & (f > 0.0)
    safe_f = torch.where(step_valid, f, torch.ones_like(f))
    gain = covariance[:, :, 0] / safe_f[:, None]
    filtered_state = state + gain * innovation[:, None]
    filtered_covariance = covariance - gain[:, :, None] * covariance[:, 0, :].unsqueeze(1)

    if r > 1:
        shifted_state = torch.cat(
            (filtered_state[:, 1:], torch.zeros_like(filtered_state[:, :1])), dim=1
        )
    else:
        shifted_state = torch.zeros_like(filtered_state)
    next_state = phi * filtered_state[:, :1] + shifted_state

    if r > 1:
        shifted_column = torch.cat(
            (
                filtered_covariance[:, 1:, 0],
                torch.zeros_like(filtered_covariance[:, :1, 0]),
            ),
            dim=1,
        )
    else:
        shifted_column = torch.zeros_like(phi)
    first_factor = phi * filtered_covariance[:, :1, 0] + shifted_column
    next_covariance = first_factor[:, :, None] * phi[:, None, :] + loading_outer

    if r > 1:
        second = phi[:, :, None] * filtered_covariance[:, :1, 1:]
        shifted_block = torch.cat(
            (
                filtered_covariance[:, 1:, 1:],
                torch.zeros_like(filtered_covariance[:, :1, 1:]),
            ),
            dim=1,
        )
        second = second + shifted_block
        next_covariance = next_covariance + torch.cat(
            (second, torch.zeros_like(next_covariance[:, :, :1])), dim=2
        )

    finite_next = torch.isfinite(next_covariance).all(dim=(1, 2))
    next_valid = step_valid & finite_next
    next_sse = sse + torch.where(
        step_valid, innovation * innovation / safe_f, torch.zeros_like(sse)
    )
    next_sum_log_f = sum_log_f + torch.where(
        step_valid, torch.log(safe_f), torch.zeros_like(sum_log_f)
    )
    return next_state, next_covariance, next_sse, next_sum_log_f, next_valid


def likelihood_for_loop(z: Tensor, phi: Tensor, loading: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Readable batched formulation with a Python time loop."""
    k, n = z.shape
    p, _ = stationary_for_loop(phi, loading)
    state = torch.zeros_like(phi)
    loading_outer = loading.unsqueeze(2) * loading.unsqueeze(1)
    sse = torch.zeros(k, dtype=z.dtype, device=z.device)
    sum_log_f = torch.zeros_like(sse)
    valid = torch.ones(k, dtype=torch.bool, device=z.device)
    for t in range(n):
        state, p, sse, sum_log_f, valid = _kalman_step(
            z[:, t], phi, loading_outer, state, p, sse, sum_log_f, valid
        )
    sigma2 = sse / n
    finite = valid & torch.isfinite(sigma2) & (sigma2 > 0.0)
    safe_sigma2 = torch.where(finite, sigma2, torch.ones_like(sigma2))
    nll = 0.5 * n * torch.log((2.0 * math.pi) * safe_sigma2) + 0.5 * sum_log_f + 0.5 * n
    finite = finite & torch.isfinite(nll)
    return (
        torch.where(finite, nll, torch.full_like(nll, PENALTY)),
        torch.where(finite, sigma2, torch.ones_like(sigma2)),
        finite,
    )


def compile_fullgraph() -> Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    return torch.compile(likelihood_for_loop, fullgraph=True)


def likelihood_while_loop(z: Tensor, phi: Tensor, loading: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Exact likelihood using public ``torch.while_loop`` for the time axis."""
    k, n = z.shape
    p, _ = stationary_while_loop(phi, loading)
    state = torch.zeros_like(phi)
    loading_outer = loading.unsqueeze(2) * loading.unsqueeze(1)
    sse = torch.zeros(k, dtype=z.dtype, device=z.device)
    sum_log_f = torch.zeros_like(sse)
    valid = torch.ones(k, dtype=torch.bool, device=z.device)
    iteration = torch.zeros((), dtype=torch.int64, device=z.device)

    def cond(
        t: Tensor,
        _state: Tensor,
        _p: Tensor,
        _sse: Tensor,
        _sum_log_f: Tensor,
        _valid: Tensor,
    ) -> Tensor:
        return t < n

    def body(
        t: Tensor,
        current_state: Tensor,
        current_p: Tensor,
        current_sse: Tensor,
        current_sum_log_f: Tensor,
        current_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        z_t = torch.index_select(z, 1, t.reshape(1)).squeeze(1)
        next_state, next_p, next_sse, next_sum_log_f, next_valid = _kalman_step(
            z_t,
            phi,
            loading_outer,
            current_state,
            current_p,
            current_sse,
            current_sum_log_f,
            current_valid,
        )
        return t + 1, next_state, next_p, next_sse, next_sum_log_f, next_valid

    _, _, _, sse, sum_log_f, valid = torch.while_loop(
        cond, body, (iteration, state, p, sse, sum_log_f, valid)
    )
    sigma2 = sse / n
    finite = valid & torch.isfinite(sigma2) & (sigma2 > 0.0)
    safe_sigma2 = torch.where(finite, sigma2, torch.ones_like(sigma2))
    nll = 0.5 * n * torch.log((2.0 * math.pi) * safe_sigma2) + 0.5 * sum_log_f + 0.5 * n
    finite = finite & torch.isfinite(nll)
    return (
        torch.where(finite, nll, torch.full_like(nll, PENALTY)),
        torch.where(finite, sigma2, torch.ones_like(sigma2)),
        finite,
    )


def compile_while_loop() -> Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    return torch.compile(likelihood_while_loop, fullgraph=True)
