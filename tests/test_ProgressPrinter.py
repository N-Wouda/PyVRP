import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import (
    CostEvaluator,
    Population,
    RandomNumberGenerator,
    Result,
    Solution,
    Statistics,
)
from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.diversity import broken_pairs_distance as bpd


def test_start(ok_small, capsys):
    """
    Tests that the progress printer outputs some statistics about the given
    instance when calling start.
    """
    printer = ProgressPrinter(should_print=True)
    printer.start(ok_small)

    out = capsys.readouterr().out
    assert_(f"{ok_small.num_depots} depot" in out)
    assert_(f"{ok_small.num_clients} clients" in out)
    assert_(f"{ok_small.num_vehicles} vehicles" in out)


def test_start_multiple_depot_plural(ok_small_multi_depot, capsys):
    """
    Tests that "depots" is plural when there are multiple depots.
    """
    printer = ProgressPrinter(should_print=True)
    printer.start(ok_small_multi_depot)

    out = capsys.readouterr().out
    assert_(f"{ok_small_multi_depot.num_depots} depots" in out)


def test_end(ok_small, capsys):
    """
    Tests that the progress printer outputs some summary statistics about the
    best-found solution and solver run when calling end.
    """
    best = Solution(ok_small, [[1, 2], [3, 4]])
    res = Result(best, Statistics(), 25, 0.05)

    printer = ProgressPrinter(should_print=True)
    printer.end(res)

    out = capsys.readouterr().out
    assert_(str(round(res.cost())) in out)
    assert_(str(res.num_iterations) in out)
    assert_(str(round(res.runtime)) in out)
    assert_(res.summary() in out)


def test_restart(capsys):
    """
    Tests that calling restart outputs a line indicating such.
    """
    printer = ProgressPrinter(should_print=True)
    printer.restart()

    out = capsys.readouterr().out
    assert_("R" in out)
    assert_("restart" in out)


def test_iteration(ok_small, capsys):
    """
    Tests that calling iteration prints a line about the current feasible and
    infeasible populations, but only every 500 iterations.
    """
    pop = Population(bpd)
    rng = RandomNumberGenerator(seed=42)
    cost_eval = CostEvaluator(1, 1, 0)

    for _ in range(10):
        pop.add(Solution.make_random(ok_small, rng), cost_eval)

    assert_(pop.num_feasible() > 0)
    assert_(pop.num_infeasible() > 0)

    stats = Statistics()
    stats.collect_from(pop, cost_eval)

    printer = ProgressPrinter(should_print=True)

    # Output is printed only every once in a while. We do not print after a
    # single iteration.
    printer.iteration(stats)
    out = capsys.readouterr().out
    assert_(stats.num_iterations, 1)
    assert_equal(out, "")

    # But only every 500 iterations. Thus, we should output results now.
    stats.num_iterations = 500
    printer.iteration(stats)
    out = capsys.readouterr().out

    # Statistics about solver progress.
    assert_(str(stats.num_iterations) in out)
    assert_(str(round(sum(stats.runtimes))) in out)

    # Statistics about feasible population.
    feas = stats.feas_stats[-1]
    assert_(str(feas.size) in out)
    assert_(str(round(feas.avg_cost)) in out)
    assert_(str(round(feas.best_cost)) in out)

    # Statistics about infeasible population.
    infeas = stats.infeas_stats[-1]
    assert_(str(infeas.size) in out)
    assert_(str(round(infeas.avg_cost)) in out)
    assert_(str(round(infeas.best_cost)) in out)


def test_should_print_false_no_output(ok_small, capsys):
    """
    Tests that disabling printing works.
    """
    pop = Population(bpd)
    rng = RandomNumberGenerator(seed=42)
    cost_eval = CostEvaluator(1, 1, 0)

    for _ in range(10):
        pop.add(Solution.make_random(ok_small, rng), cost_eval)

    best = Solution(ok_small, [[1, 2], [3, 4]])  # BKS
    pop.add(best, cost_eval)

    # Checks that there are both feasible and infeasible solutions in this
    # population.
    assert_(pop.num_feasible() > 0)
    assert_(pop.num_infeasible() > 0)

    stats = Statistics()
    stats.collect_from(pop, cost_eval)

    # Set up the progress printer, call all its methods, and then check that
    # the output captured on stdout is empty.
    printer = ProgressPrinter(should_print=False)
    printer.start(ok_small)
    printer.iteration(stats=stats)
    printer.restart()
    printer.end(result=Result(best, stats, 25, 0.05))

    out = capsys.readouterr().out
    assert_equal(out, "")


@pytest.mark.parametrize(
    "routes",
    [
        [[1, 2], [3, 4]],  # feasible
        [[1, 2, 3, 4]],  # infeasible
    ],
)
def test_print_dash_when_subpopulation_is_empty(ok_small, routes, capsys):
    """
    Tests that a "-" is printed as cost if one of the subpopulations is empty.
    """
    pop = Population(bpd)
    cost_eval = CostEvaluator(1, 1, 0)

    solution = Solution(ok_small, routes)
    pop.add(solution, cost_eval)

    stats = Statistics()
    stats.collect_from(pop, cost_eval)
    stats.num_iterations = 500  # force printing

    printer = ProgressPrinter(should_print=True)
    printer.iteration(stats)

    # Population contains either one feasible or one infeasible solution, so
    # the costs of the empty subpopulation should be printed as "-".
    out = capsys.readouterr().out
    assert_("-" in out)
