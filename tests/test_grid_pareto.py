from itertools import product

import numpy as np
import pytest

import core.grid_pareto as grid_pareto


def _quadratic_oracle(values, directions):
    """Historical strict-dominance definition, kept independent of production."""

    def dominates(candidate, other):
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

    return np.asarray(
        [
            not any(
                index != other_index and dominates(other, candidate)
                for other_index, other in enumerate(values)
            )
            for index, candidate in enumerate(values)
        ],
        dtype=np.bool_,
    )


@pytest.mark.parametrize("directions", product(("maximize", "minimize"), repeat=2))
def test_two_objective_dense_parity_for_every_direction(directions):
    values = [list(point) for point in product((-2.0, -0.0, 0.0, 1.0, 2.0), repeat=2)]
    values.extend(([1.0, 1.0], [-0.0, 0.0], [2.0, -2.0]))

    actual = grid_pareto.compute_grid_pareto_mask(values, directions)

    np.testing.assert_array_equal(actual, _quadratic_oracle(values, directions))


@pytest.mark.parametrize(
    "values",
    [
        [[1.0, 3.0], [1.0, 2.0], [1.0, 2.0], [2.0, 4.0]],
        [[3.0, 1.0], [2.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        [[-0.0, 0.0], [0.0, -0.0], [1.0, 1.0]],
        [
            [np.finfo(np.float64).max, np.nextafter(0.0, 1.0)],
            [np.finfo(np.float64).max, np.nextafter(0.0, 1.0)],
            [-np.finfo(np.float64).max, np.finfo(np.float64).max],
        ],
    ],
)
def test_two_objective_ties_duplicates_signed_zero_and_extremes(values):
    directions = ("maximize", "minimize")

    actual = grid_pareto.compute_grid_pareto_mask(values, directions)

    np.testing.assert_array_equal(actual, _quadratic_oracle(values, directions))


def test_two_objective_empty_single_row_and_constant_columns():
    directions = ("minimize", "maximize")

    assert grid_pareto.compute_grid_pareto_mask([], directions).tolist() == []
    assert grid_pareto.compute_grid_pareto_mask([[2.0, 3.0]], directions).tolist() == [True]

    constant_first = [[1.0, 4.0], [1.0, 4.0], [1.0, 2.0]]
    constant_second = [[4.0, 1.0], [4.0, 1.0], [2.0, 1.0]]
    for values in (constant_first, constant_second):
        np.testing.assert_array_equal(
            grid_pareto.compute_grid_pareto_mask(values, directions),
            _quadratic_oracle(values, directions),
        )


def test_two_objective_does_not_mutate_caller_array():
    values = np.asarray([[3.0, 7.0], [2.0, 1.0], [1.0, 5.0]], dtype=np.float64)
    before = values.copy()

    grid_pareto.compute_grid_pareto_mask(values, ("maximize", "minimize"))

    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize(
    ("dimensions", "directions"),
    [
        (3, ("maximize", "minimize", "maximize")),
        (4, ("minimize", "maximize", "minimize", "maximize")),
    ],
)
def test_three_and_four_objective_fallback_match_oracle(dimensions, directions):
    rng = np.random.default_rng(20260802 + dimensions)
    values = rng.integers(-3, 4, size=(45, dimensions)).astype(np.float64).tolist()
    values.extend((values[0].copy(), values[1].copy()))

    actual = grid_pareto.compute_grid_pareto_mask(values, directions)

    np.testing.assert_array_equal(actual, _quadratic_oracle(values, directions))


def test_two_objective_dispatch_bypasses_quadratic_fallback(monkeypatch):
    def unexpected_fallback(values, directions):  # noqa: ARG001
        raise AssertionError("2D path called the quadratic fallback")

    monkeypatch.setattr(grid_pareto, "_compute_quadratic_mask", unexpected_fallback)

    assert grid_pareto.compute_grid_pareto_mask(
        [[1.0, 2.0], [2.0, 1.0]],
        ("maximize", "minimize"),
    ).tolist() == [False, True]


def test_three_objective_dispatch_uses_python_sequence_fallback(monkeypatch):
    calls = []

    class NoArrayConversion(list):
        def __array__(self, dtype=None, copy=None):  # noqa: ARG002
            raise AssertionError("3D path converted the input to an ndarray")

    def fallback(values, directions):
        calls.append((values, directions))
        return np.asarray([True, False], dtype=np.bool_)

    values = NoArrayConversion([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    directions = ("maximize", "minimize", "maximize")
    monkeypatch.setattr(grid_pareto, "_compute_quadratic_mask", fallback)

    assert grid_pareto.compute_grid_pareto_mask(values, directions).tolist() == [True, False]
    assert calls == [(values, directions)]
