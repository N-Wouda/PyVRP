import math
import pickle

import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

from pyvrp import CostEvaluator, Population, RandomNumberGenerator, Solution
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics
from pyvrp.diversity import broken_pairs_distance


@pytest.mark.parametrize(
    ("routes", "num_iterations", "runtime"),
    [([[1, 2], [3], [4]], 1, 1.5), ([[1, 2, 3, 4]], 100, 54.2)],
)
def test_fields_are_correctly_set(ok_small, routes, num_iterations, runtime):
    """
    Tests that ``Result``'s data properties are correctly set after
    initialisation completes.
    """
    sol = Solution(ok_small, routes)
    res = Result(sol, Statistics(), num_iterations, runtime)

    assert_equal(res.is_feasible(), sol.is_feasible())
    assert_equal(res.num_iterations, num_iterations)
    assert_allclose(res.runtime, runtime)

    if sol.is_feasible():
        assert_equal(res.cost(), CostEvaluator(0, 0, 0).cost(sol))
    else:
        assert_equal(res.cost(), math.inf)


@pytest.mark.parametrize(
    ("num_iterations", "runtime"),
    [
        (-1, 0.0),  # num_iterations < 0
        (0, -1.0),  # runtime < 0
    ],
)
def test_init_raises_invalid_arguments(ok_small, num_iterations, runtime):
    """
    Tests that invalid arguments are rejected.
    """
    sol = Solution(ok_small, [[1, 2, 3, 4]])

    with assert_raises(ValueError):
        Result(sol, Statistics(), num_iterations, runtime)


@pytest.mark.parametrize("num_iterations", [0, 1, 10])
def test_num_iterations(ok_small, num_iterations: int):
    """
    Tests access to the ``num_iterations`` property.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)
    pop = Population(broken_pairs_distance)
    stats = Statistics()

    for _ in range(num_iterations):
        stats.collect_from(pop, cost_evaluator)

    best = Solution.make_random(ok_small, rng)
    res = Result(best, stats, num_iterations, 0.0)
    assert_equal(res.num_iterations, num_iterations)


@pytest.mark.parametrize(
    "routes",
    [[[1, 2], [3], [4]], [[1, 2, 3, 4]]],
)
def test_summary_contains_essential_information(ok_small, routes):
    """
    Tests that calling ``summary()`` returns a summary of the solution.
    """
    sol = Solution(ok_small, routes)
    res = Result(sol, Statistics(), 0, 0.0)
    summary = res.summary()

    # Test that feasibility status and solution cost are presented.
    if sol.is_feasible():
        cost = CostEvaluator(0, 0, 0).cost(sol)
        assert_(str(cost) in summary)
    else:
        assert_("INFEASIBLE" in summary)

    # Test that standard statistics are always present.
    assert_(str(sol.num_routes()) in summary)
    assert_(str(sol.num_clients()) in summary)
    assert_(str(sol.distance()) in summary)
    assert_(str(sol.duration()) in summary)


@pytest.mark.parametrize(
    "routes",
    [[[1, 2], [3], [4]], [[1, 2, 3, 4]]],
)
def test_str_contains_summary_and_routes(ok_small, routes):
    """
    Tests that printing (or, in general, calling ``str(result)``) returns a
    summary of the solution and the solution's routes.
    """
    sol = Solution(ok_small, routes)
    res = Result(sol, Statistics(), 0, 0.0)
    str_representation = str(res)

    # Test that the summary details are present in the string representation.
    assert_(res.summary() in str_representation)

    # And make sure that all routes are printed as well.
    for route in sol.routes():
        assert_(str(route) in str_representation)


@pytest.mark.parametrize("num_iterations", [0, 1, 10])
def test_result_can_be_pickled(ok_small, num_iterations: int):
    """
    Tests that a ``Result`` object can be pickled: it can be serialised and
    unserialised. This is useful for e.g. storing results to disk.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)
    pop = Population(broken_pairs_distance)
    stats = Statistics()

    for _ in range(num_iterations):
        pop.add(Solution.make_random(ok_small, rng), cost_evaluator)
        stats.collect_from(pop, cost_evaluator)

    best = Solution(ok_small, [[1, 2], [3], [4]])

    before_pickle = Result(best, stats, num_iterations, 1.2)
    bytes = pickle.dumps(before_pickle)
    after_pickle = pickle.loads(bytes)

    assert_equal(after_pickle.best, before_pickle.best)
    assert_equal(after_pickle.stats, before_pickle.stats)
    assert_equal(after_pickle, before_pickle)
