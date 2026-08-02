"""Exact Pareto membership for the shared Grid ranking path."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_grid_pareto_mask(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> np.ndarray:
    """Return exact positional Pareto flags for finite objective vectors.

    Exactly two objectives use the sorting path. Three or more objectives keep
    the historical direct Python all-pairs implementation.
    """

    if len(directions) == 2:
        return _compute_two_objective_mask(values, directions)
    return _compute_quadratic_mask(values, directions)


def _compute_two_objective_mask(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> np.ndarray:
    row_count = len(values)
    if row_count == 0:
        return np.zeros(0, dtype=np.bool_)

    # This copy is intentional: direction normalization must never mutate a
    # caller-owned contiguous float64 array.
    normalized = np.array(values, dtype=np.float64, order="C", copy=True)
    if normalized.ndim != 2 or normalized.shape != (row_count, 2):
        raise ValueError("Two-objective Pareto values must have exactly two columns.")
    for column, direction in enumerate(directions):
        if direction == "maximize":
            normalized[:, column] *= -1.0

    first = normalized[:, 0]
    second = normalized[:, 1]
    sorted_positions = np.lexsort((second, first))
    sorted_first = first[sorted_positions]
    sorted_second = second[sorted_positions]

    group_start = np.empty(row_count, dtype=np.bool_)
    group_start[0] = True
    group_start[1:] = sorted_first[1:] != sorted_first[:-1]
    group_ids = np.cumsum(group_start, dtype=np.intp) - 1
    group_minima = sorted_second[group_start]

    # Equal-first rows must be judged as a group: only ties at the group's
    # second minimum can survive, and that minimum must beat every strictly
    # earlier group. A plain cumulative minimum mishandles duplicates and ties.
    prior_group_minima = np.empty(len(group_minima), dtype=np.float64)
    prior_group_minima[0] = np.inf
    if len(group_minima) > 1:
        prior_group_minima[1:] = np.minimum.accumulate(group_minima[:-1])

    sorted_mask = (
        (sorted_second == group_minima[group_ids])
        & (group_minima[group_ids] < prior_group_minima[group_ids])
    )
    positional_mask = np.empty(row_count, dtype=np.bool_)
    positional_mask[sorted_positions] = sorted_mask
    return positional_mask


def _compute_quadratic_mask(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
) -> np.ndarray:
    def dominates(candidate: Sequence[float], other: Sequence[float]) -> bool:
        better_or_equal = True
        strictly_better = False
        for value, other_value, direction in zip(candidate, other, directions):
            if direction == "maximize":
                if value < other_value:
                    better_or_equal = False
                    break
                if value > other_value:
                    strictly_better = True
            else:
                if value > other_value:
                    better_or_equal = False
                    break
                if value < other_value:
                    strictly_better = True
        return better_or_equal and strictly_better

    positional_mask = np.ones(len(values), dtype=np.bool_)
    for index, candidate in enumerate(values):
        for other_index, other in enumerate(values):
            if index == other_index:
                continue
            if dominates(other, candidate):
                positional_mask[index] = False
                break
    return positional_mask
