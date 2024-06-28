from numpy.testing import assert_, assert_equal

from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyParams
from pyvrp.Population import PopulationParams
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    Exchange10,
    NeighbourhoodParams,
    SwapStar,
    SwapTails,
)
from pyvrp.solve import SolveParams, solve
from pyvrp.stop import MaxIterations
from tests.helpers import DATA_DIR


def test_default_values():
    """
    Tests that the default values are set correctly.
    """
    params = SolveParams()

    assert_equal(params.genetic, GeneticAlgorithmParams())
    assert_equal(params.penalty, PenaltyParams())
    assert_equal(params.population, PopulationParams())
    assert_equal(params.neighbourhood, NeighbourhoodParams())
    assert_equal(params.node_ops, NODE_OPERATORS)
    assert_equal(params.route_ops, ROUTE_OPERATORS)


def test_solve_params_from_file():
    """
    Tests that the solver parameters are correctly loaded from a TOML file.
    """
    params = SolveParams.from_file(DATA_DIR / "test_config.toml")

    genetic = GeneticAlgorithmParams(0.1, 200)
    penalty = PenaltyParams(12, 100, 1.25, 0.85, 0.43)
    population = PopulationParams(10, 20, 3, 4, 0.0, 1.0)
    neighbourhood = NeighbourhoodParams(0, 0, 20, True, True)
    node_ops = [Exchange10, SwapTails]
    route_ops = [SwapStar]

    assert_equal(params.genetic, genetic)
    assert_equal(params.penalty, penalty)
    assert_equal(params.population, population)
    assert_equal(params.neighbourhood, neighbourhood)
    assert_equal(params.node_ops, node_ops)
    assert_equal(params.route_ops, route_ops)


def test_solve_params_from_file_defaults():
    """
    Tests that if the TOML file does not contain all solver parameters,
    it defaults to the constructor's default values.
    """
    params = SolveParams.from_file(DATA_DIR / "empty_config.toml")
    assert_equal(params, SolveParams())


def test_solve_same_seed(ok_small):
    """
    Smoke test that checks that that solving an instance with the same seed
    results in the same search trajectories.
    """
    res1 = solve(ok_small, stop=MaxIterations(10), seed=0)
    res2 = solve(ok_small, stop=MaxIterations(10), seed=0)

    assert_equal(res1.best, res2.best)
    assert_equal(res1.stats.feas_stats, res2.stats.feas_stats)
    assert_equal(res1.stats.infeas_stats, res2.stats.infeas_stats)


def test_solve_more_iterations_is_better(prize_collecting):
    """
    Smoke test that checks that running more iterations results in an improved
    solution.
    """
    cost10 = solve(prize_collecting, stop=MaxIterations(10)).cost()
    cost100 = solve(prize_collecting, stop=MaxIterations(100)).cost()
    assert_(cost100 < cost10)


def test_solve_custom_params(ok_small):
    """
    Tests that solving an instance with custom solver parameters works as
    expected by checking that the population size is respected.
    """
    # First solve using the default solver parameters.
    res = solve(ok_small, stop=MaxIterations(200), seed=0)

    pop_params = PopulationParams()
    max_pop_size = pop_params.min_pop_size + pop_params.generation_size

    # Neither subpopulation should exceed the maximum population size;
    # we use the statistics to check this.
    max_feas_size = max([datum.size for datum in res.stats.feas_stats])
    max_infeas_size = max([datum.size for datum in res.stats.infeas_stats])

    assert_(max_feas_size <= max_pop_size)
    assert_(max_infeas_size <= max_pop_size)

    # Let's now use custom parameters with a maximum population size of 15.
    pop_params = PopulationParams(min_pop_size=5, generation_size=10)
    params = SolveParams(population=pop_params)
    res = solve(ok_small, stop=MaxIterations(200), seed=0, params=params)

    max_pop_size = pop_params.min_pop_size + pop_params.generation_size
    max_feas_size = max([datum.size for datum in res.stats.feas_stats])
    max_infeas_size = max([datum.size for datum in res.stats.infeas_stats])

    assert_(max_feas_size <= max_pop_size)
    assert_(max_infeas_size <= max_pop_size)
