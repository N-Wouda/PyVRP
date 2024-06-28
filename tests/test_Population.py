import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal,
    assert_raises,
)
from pytest import mark

from pyvrp import (
    CostEvaluator,
    Population,
    PopulationParams,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.diversity import broken_pairs_distance as bpd


@mark.parametrize(
    (
        "min_pop_size",
        "generation_size",
        "nb_elite",
        "nb_close",
        "lb_diversity",
        "ub_diversity",
    ),
    [
        (1, 1, 1, 1, -1, 1.0),  # -1 lb_diversity
        (1, 1, 1, 1, 2, 1.0),  # 2 lb_diversity
        (1, 1, 1, 1, 0, -1.0),  # -1 ub_diversity
        (1, 1, 1, 1, 0, 2.0),  # 2 ub_diversity
        (1, 1, 1, 1, 1, 0.5),  # ub_diversity < lb_diversity
        (1, 1, 1, 1, 0.5, 0.5),  # ub_diversity == lb_diversity
    ],
)
def test_params_constructor_throws_when_arguments_invalid(
    min_pop_size: int,
    generation_size: int,
    nb_elite: int,
    nb_close: int,
    lb_diversity: float,
    ub_diversity: float,
):
    """
    Tests that invalid configurations are not accepted.
    """
    with assert_raises(ValueError):
        PopulationParams(
            min_pop_size,
            generation_size,
            nb_elite,
            nb_close,
            lb_diversity,
            ub_diversity,
        )


@mark.parametrize(
    (
        "min_pop_size",
        "generation_size",
        "nb_elite",
        "nb_close",
        "lb_diversity",
        "ub_diversity",
    ),
    [
        (1, 1, 1, 1, 0.0, 0.5),  # >0 min_pop_size
        (1, 0, 1, 1, 0.0, 0.5),  # 0 generation_size
        (1, 1, 0, 1, 0.0, 0.5),  # 0 nb_elite
        (1, 1, 1, 0, 0.0, 0.5),  # 0 nb_close
        (1, 1, 1, 1, 0.0, 0.5),  # 0 lb_diversity
        (1, 1, 1, 1, 0.0, 1.0),  # 1 ub_diversity
    ],
)
def test_params_constructor_does_not_raise_when_arguments_valid(
    min_pop_size: int,
    generation_size: int,
    nb_elite: int,
    nb_close: int,
    lb_diversity: float,
    ub_diversity: float,
):
    """
    Tests valid boundary cases.
    """
    params = PopulationParams(
        min_pop_size,
        generation_size,
        nb_elite,
        nb_close,
        lb_diversity,
        ub_diversity,
    )

    assert_equal(params.min_pop_size, min_pop_size)
    assert_equal(params.generation_size, generation_size)
    assert_equal(params.nb_elite, nb_elite)
    assert_equal(params.nb_close, nb_close)
    assert_allclose(params.lb_diversity, lb_diversity)
    assert_allclose(params.ub_diversity, ub_diversity)
    assert_equal(params.max_pop_size, min_pop_size + generation_size)


def test_add_triggers_purge(ok_small):
    """
    Tests that adding another solution to a population of maximum size triggers
    survivor selection, that is, a purge that reduces the relevant population
    back to minimum size.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    params = PopulationParams()
    pop = Population(bpd, params=params)
    for _ in range(params.min_pop_size):
        pop.add(Solution.make_random(ok_small, rng), cost_evaluator)

    # Population should initialise at least min_pop_size solutions
    assert_(len(pop) >= params.min_pop_size)
    assert_equal(len(pop), pop.num_feasible() + pop.num_infeasible())

    num_feas = pop.num_feasible()
    num_infeas = pop.num_infeasible()

    while True:  # keep adding feasible solutions until we are about to purge
        sol = Solution.make_random(ok_small, rng)

        if sol.is_feasible():
            pop.add(sol, cost_evaluator)
            num_feas += 1

            assert_equal(len(pop), num_feas + num_infeas)
            assert_equal(pop.num_feasible(), num_feas)

        if num_feas == params.max_pop_size:  # next add() triggers purge
            break

    # RNG is fixed, and this next solution is feasible. Since we now have a
    # feasible population that is of maximal size, adding this solution should
    # trigger survivor selection (purge). Survivor selection reduces the
    # feasible subpopulation to min_pop_size, so the overal population is then
    # just num_infeas + min_pop_size.
    sol = Solution.make_random(ok_small, rng)
    assert_(sol.is_feasible())

    pop.add(sol, cost_evaluator)
    assert_equal(pop.num_feasible(), params.min_pop_size)
    assert_equal(len(pop), num_infeas + params.min_pop_size)


def test_select_returns_same_parents_if_no_other_option(ok_small):
    """
    Tests that the ``select()`` method tries to return two different parents,
    but will return the same solution twice if there is no other option.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=2_147_483_647)

    params = PopulationParams(min_pop_size=0)
    pop = Population(bpd, params=params)

    assert_equal(len(pop), 0)

    pop.add(Solution(ok_small, [[3, 2], [1, 4]]), cost_evaluator)
    assert_equal(len(pop), 1)

    # We added a single solution, so we should now get the same parent twice.
    parents = pop.select(rng, cost_evaluator)
    assert_(parents[0] == parents[1])

    # Now we add another, different solution.
    pop.add(Solution(ok_small, [[3, 2], [1], [4]]), cost_evaluator)
    assert_equal(len(pop), 2)

    # We should now get two different solutions as parents, at least most of
    # the time. The actual probability of getting the same parents is very
    # small, but not zero. So let's do an experiment where we do 1000 selects,
    # and collect the number of times the parents are different.
    different_parents = 0
    for _ in range(1_000):
        parents = pop.select(rng, cost_evaluator)
        different_parents += parents[0] != parents[1]

    # The probability of selecting different parents is very close to 100%, so
    # we would expect to observe different parents much more than 90% of the
    # time. At the same time, it is very unlikely each one of the 1000 selects
    # returns a different parent pair.
    assert_(900 < different_parents < 1_000)


def test_pop_is_empty_with_zero_min_pop_size_and_generation_size(ok_small):
    """
    Tests that the population can never grow when it starts empty and the
    generation size is set to 0: adding a new solution then immediately
    triggers survivor selection.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=12)

    params = PopulationParams(min_pop_size=0, generation_size=0)
    pop = Population(bpd, params=params)

    assert_equal(len(pop), 0)

    for _ in range(10):
        # With zero min_pop_size and zero generation_size, every additional
        # solution triggers a purge. So the population size can never grow
        # beyond zero.
        pop.add(Solution.make_random(ok_small, rng), cost_evaluator)
        assert_equal(len(pop), 0)


@mark.parametrize("nb_elite", [5, 25])
def test_elite_solutions_are_not_purged(rc208, nb_elite: int):
    """
    Tests that elite solutions - those considered of such high quality that
    they should be given special treatment - are not purged during survivor
    selection.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    params = PopulationParams(nb_elite=nb_elite)
    rng = RandomNumberGenerator(seed=42)

    pop = Population(bpd, params=params)

    # Keep adding solutions until the infeasible subpopulation is of maximum
    # size.
    while pop.num_infeasible() != params.max_pop_size:
        pop.add(Solution.make_random(rc208, rng), cost_evaluator)

    assert_equal(pop.num_infeasible(), params.max_pop_size)

    # These are the elite best solutions in the current solution pool. These
    # should never be purged.
    curr_sols = [sol for sol in pop if not sol.is_feasible()]
    best_sols = sorted(curr_sols, key=cost_evaluator.penalised_cost)
    elite_sols = best_sols[:nb_elite]

    # Add a solution that is certainly not feasible, thus causing a purge.
    single_route = list(range(rc208.num_depots, rc208.num_locations))
    pop.add(Solution(rc208, [single_route]), cost_evaluator)

    # After the purge, there should remain min_pop_size infeasible solutions.
    assert_equal(pop.num_infeasible(), params.min_pop_size)

    # Elite solutions from before the purge should also still be present. We
    # test by id/memory location that these solutions are still present.
    new_sols = [id(sol) for sol in pop]
    for elite_sol in elite_sols:
        assert_(id(elite_sol) in new_sols)


@mark.parametrize("k", [2, 3])
def test_tournament_ranks_by_fitness(rc208, k: int):
    """
    Tests that the tournament-based parent selection on average returns
    solutions about as often as their relative fitness value would suggest.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)
    pop = Population(bpd)

    for _ in range(50):
        sol = Solution.make_random(rc208, rng)
        if not sol.is_feasible():
            pop.add(sol, cost_evaluator)

    assert_equal(pop.num_feasible(), 0)

    # Since this test requires the fitness values of the solutions, we have
    # to access the underlying infeasible subpopulation directly.
    infeas_pop = pop._infeas  # noqa: SLF001
    infeas_pop.update_fitness(cost_evaluator)

    items = [item for item in infeas_pop]
    by_fitness = sorted(items, key=lambda item: item.fitness)
    sol2idx = {item.solution: idx for idx, item in enumerate(by_fitness)}
    infeas_count = np.zeros(len(infeas_pop))

    for _ in range(10_000):
        sol = pop.tournament(rng, cost_evaluator, k=k)
        infeas_count[sol2idx[sol]] += 1

    # Now we compare the observed ranking from the tournament selection with
    # what we expect from the actual fitness ranking. We compute the percentage
    # of times we're incorrect, and test that that number is not too high.
    actual_rank = 1 + np.argsort(-infeas_count)  # higher is better
    expected_rank = 1 + np.arange(len(infeas_pop))
    pct_off = np.abs((actual_rank - expected_rank) / len(infeas_pop)).mean()
    assert_(pct_off < 0.05)

    # Previous test compared just rank. Now we compare expected frequency. An
    # item at rank i wins only when the other k - 1 items have a rank lower
    # than i. That happens with probability roughly proportional to
    #   (1 - i / #pop) ** (k - 1)
    actual_freq = infeas_count / infeas_count.sum()
    expected_freq = (1 - expected_rank / len(infeas_pop)) ** (k - 1)
    expected_freq /= expected_freq.sum()
    assert_allclose(actual_freq, expected_freq, atol=0.01)  # 1% tolerance


@mark.parametrize("k", [-100, -1, 0])  # k must be strictly positive
def test_tournament_raises_for_invalid_k(rc208, k: int):
    """
    Tests that k >= 0 is required for tournament-based selection.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    pop = Population(bpd)
    for _ in range(5):
        pop.add(Solution.make_random(rc208, rng), cost_evaluator)

    with assert_raises(ValueError):
        pop.tournament(rng, cost_evaluator, k=k)


def test_purge_removes_duplicates(rc208):
    """
    Tests that purging/survivor selection first removes duplicate solutions,
    before purging by (biased) fitness.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    params = PopulationParams(min_pop_size=5, generation_size=20)
    rng = RandomNumberGenerator(seed=42)

    pop = Population(bpd, params=params)
    for _ in range(params.min_pop_size - 1):
        pop.add(Solution.make_random(rc208, rng), cost_evaluator)

    assert_equal(len(pop), params.min_pop_size - 1)

    # This is the solution we are going to add a few times. That should make
    # sure the relevant subpopulation definitely contains duplicates.
    sol = Solution.make_random(rc208, rng)
    assert_(not sol.is_feasible())

    for _ in range(params.generation_size):
        pop.add(sol, cost_evaluator)

    # Keep adding the solution until we have had a purge, and returned to the
    # minimum population size. Note that the purge is done after adding the
    # solution, so we must add the same solution in order to have at most
    # min_pop_size - 1 other solutions than the duplicated solution.
    while pop.num_infeasible() != params.min_pop_size:
        pop.add(sol, cost_evaluator)

    # Since duplicates are purged first, there should now be only one of them
    # in the subpopulation.
    duplicates = sum(other == sol for other in pop)
    assert_equal(duplicates, 1)


def test_clear(rc208):
    """
    Tests that clearing the population reduces its size to zero.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    pop = Population(bpd)
    for _ in range(10):
        pop.add(Solution.make_random(rc208, rng), cost_evaluator)

    assert_equal(len(pop), 10)

    pop.clear()
    assert_equal(len(pop), 0)
