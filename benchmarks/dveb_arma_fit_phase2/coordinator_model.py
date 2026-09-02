"""Executable specification of Phase-II deterministic barrier batching."""

from __future__ import annotations

from collections.abc import Sequence

from .common import MAX_ACTIVE_OPTIMIZERS


def consecutive_chunks(count: int, limit: int = MAX_ACTIVE_OPTIMIZERS) -> tuple[tuple[int, ...], ...]:
    if count < 1 or limit < 1:
        raise ValueError("count and limit must be positive")
    return tuple(tuple(range(start, min(start + limit, count))) for start in range(0, count, limit))


def barrier_rounds(requests: Sequence[Sequence[str]]) -> tuple[tuple[tuple[int, str], ...], ...]:
    """Return the only legal request grouping for independent worker traces."""
    offsets = [0] * len(requests)
    active = set(range(len(requests)))
    rounds: list[tuple[tuple[int, str], ...]] = []
    while active:
        batch = []
        completed = []
        for worker in sorted(active):
            if offsets[worker] == len(requests[worker]):
                completed.append(worker)
                continue
            batch.append((worker, requests[worker][offsets[worker]]))
            offsets[worker] += 1
        active.difference_update(completed)
        if batch:
            rounds.append(tuple(batch))
        elif active:
            raise RuntimeError("barrier model made no progress")
    return tuple(rounds)
