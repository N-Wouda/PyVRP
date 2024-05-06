import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
from pytest import mark

from pyvrp import (
    Client,
    CostEvaluator,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
    VehicleType,
)
from pyvrp.search import (
    Exchange10,
    Exchange11,
    LocalSearch,
    NeighbourhoodParams,
    SwapStar,
    compute_neighbours,
)
from pyvrp.search._search import LocalSearch as cpp_LocalSearch


def test_local_search_returns_same_solution_with_empty_neighbourhood(ok_small):
    """
    Tests that calling the local search when it only has node operators and
    an empty neighbourhood is a no-op: since the node operators respect the
    neighbourhood definition, they cannot do anything with an empty
    neighbourhood.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    neighbours = [[] for _ in range(ok_small.num_locations)]
    ls = LocalSearch(ok_small, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small))
    ls.add_node_operator(Exchange11(ok_small))

    # The search is completed after one iteration due to the empty
    # neighbourhood. This also prevents moves involving empty routes,
    # which are not explicitly forbidden by the empty neighbourhood.
    sol = Solution.make_random(ok_small, rng)
    assert_equal(ls.search(sol, cost_evaluator), sol)


@mark.parametrize("size", [1, 2, 3, 4, 6, 7])  # num_clients + 1 == 5
def test_raises_when_neighbourhood_dimensions_do_not_match(ok_small, size):
    """
    Tests that the local search raises when the neighbourhood size does not
    correspond to the problem dimensions.
    """
    rng = RandomNumberGenerator(seed=42)

    # Each of the given sizes is either smaller than or bigger than desired.
    neighbours = [[] for _ in range(size)]

    with assert_raises(RuntimeError):
        LocalSearch(ok_small, rng, neighbours)

    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small))

    with assert_raises(RuntimeError):
        ls.set_neighbours(neighbours)


def test_raises_when_neighbourhood_contains_self_or_depot(ok_small):
    """
    Tests that the local search raises when the granular neighbourhood contains
    the depot (for any client) or the client is in its own neighbourhood.
    """
    rng = RandomNumberGenerator(seed=42)

    neighbours = [[], [2], [3], [4], [0]]  # 4 has depot as neighbour
    with assert_raises(RuntimeError):
        LocalSearch(ok_small, rng, neighbours)

    neighbours = [[], [1], [3], [4], [1]]  # 1 has itself as neighbour
    with assert_raises(RuntimeError):
        LocalSearch(ok_small, rng, neighbours)


@mark.parametrize(
    (
        "weight_wait_time",
        "weight_time_warp",
        "nb_granular",
        "symmetric_proximity",
        "symmetric_neighbours",
    ),
    [
        (20, 20, 10, True, False),
        (20, 20, 10, True, True),
        # From original c++ implementation
        # (18, 20, 34, False),
        (18, 20, 34, True, True),
    ],
)
def test_local_search_set_get_neighbours(
    rc208,
    weight_wait_time: int,
    weight_time_warp: int,
    nb_granular: int,
    symmetric_proximity: bool,
    symmetric_neighbours: bool,
):
    """
    Tests setting and getting neighbours on the local search instance.
    """
    rng = RandomNumberGenerator(seed=42)

    params = NeighbourhoodParams(nb_granular=1)
    prev_neighbours = compute_neighbours(rc208, params)
    ls = LocalSearch(rc208, rng, prev_neighbours)

    params = NeighbourhoodParams(
        weight_wait_time,
        weight_time_warp,
        nb_granular,
        symmetric_proximity,
        symmetric_neighbours,
    )
    neighbours = compute_neighbours(rc208, params)

    # Test that before we set neighbours we don't have same
    assert_(ls.neighbours() != neighbours)

    # Test after we set we have the same
    ls.set_neighbours(neighbours)
    ls_neighbours = ls.neighbours()
    assert_equal(ls_neighbours, neighbours)

    # Check that the bindings make a copy (in both directions)
    assert_(ls_neighbours is not neighbours)
    ls_neighbours[1] = []
    assert_(ls.neighbours() != ls_neighbours)
    assert_equal(ls.neighbours(), neighbours)
    neighbours[1] = []
    assert_(ls.neighbours() != neighbours)


def test_reoptimize_changed_objective_timewarp_OkSmall(ok_small):
    """
    This test reproduces a bug where loadSolution in LocalSearch.cpp would
    reset the timewarp for a route to 0 if the route was not changed. This
    would cause improving moves with a smaller timewarp not to be considered
    because the current cost doesn't count the current time warp.
    """
    rng = RandomNumberGenerator(seed=42)
    sol = Solution(ok_small, [[1, 2, 3, 4]])

    # We make neighbours only contain 1 -> 2, so the only feasible move
    # is changing [1, 2, 3, 4] into [2, 1, 3, 4] or moving one of the nodes
    # into its own route. Since those solutions have larger distance but
    # smaller time warp, they are considered improving moves with a
    # sufficiently large time warp penalty.
    neighbours = [[], [2], [], [], []]  # 1 -> 2 only
    ls = LocalSearch(ok_small, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small))

    # With 0 timewarp penalty, the solution should not change since
    # the solution [2, 1, 3, 4] has larger distance.
    improved_sol = ls.search(sol, CostEvaluator(0, 0, 0))
    assert_equal(sol, improved_sol)

    # Now doing it again with a large TW penalty, we must find the alternative
    # solution
    # (previously this was not the case since due to caching the current TW was
    # computed as being zero, causing the move to be evaluated as worse)
    cost_evaluator_tw = CostEvaluator(0, 1000, 0)
    improved_sol = ls.search(sol, cost_evaluator_tw)
    improved_cost = cost_evaluator_tw.penalised_cost(improved_sol)
    assert_(improved_cost < cost_evaluator_tw.penalised_cost(sol))


def test_prize_collecting(prize_collecting):
    """
    Tests that local search works on a small prize-collecting instance.
    """
    rng = RandomNumberGenerator(seed=42)
    cost_evaluator = CostEvaluator(1, 1, 0)

    sol = Solution.make_random(prize_collecting, rng)
    sol_cost = cost_evaluator.penalised_cost(sol)

    neighbours = compute_neighbours(prize_collecting)
    ls = LocalSearch(prize_collecting, rng, neighbours)
    ls.add_node_operator(Exchange10(prize_collecting))  # relocate
    ls.add_node_operator(Exchange11(prize_collecting))  # swap

    improved = ls.search(sol, cost_evaluator)
    improved_cost = cost_evaluator.penalised_cost(improved)

    assert_(improved.num_clients() < prize_collecting.num_clients)
    assert_(improved_cost < sol_cost)


def test_cpp_shuffle_results_in_different_solution(rc208):
    """
    Tests that calling shuffle changes the evaluation order, which can well
    result in different solutions generated from the same initial solution.
    """
    rng = RandomNumberGenerator(seed=42)

    ls = cpp_LocalSearch(rc208, compute_neighbours(rc208))
    ls.add_node_operator(Exchange10(rc208))
    ls.add_node_operator(Exchange11(rc208))

    cost_evaluator = CostEvaluator(1, 1, 0)
    sol = Solution.make_random(rc208, rng)

    # LocalSearch::search is deterministic, so two calls with the same base
    # solution should result in the same improved solution.
    improved1 = ls.search(sol, cost_evaluator)
    improved2 = ls.search(sol, cost_evaluator)
    assert_(improved1 == improved2)

    # But the shuffle method changes the order in which moves are evaluated,
    # which should result in a very different search trajectory.
    ls.shuffle(rng)
    improved3 = ls.search(sol, cost_evaluator)
    assert_(improved3 != improved1)


def test_vehicle_types_are_preserved_for_locally_optimal_solutions(rc208):
    """
    Tests that a solution that is already locally optimal returns the same
    solution, particularly w.r.t. the underlying vehicles. This exercises an
    issue where loading the solution in the local search did not preserve the
    vehicle types.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(rc208)

    ls = cpp_LocalSearch(rc208, neighbours)
    ls.add_node_operator(Exchange10(rc208))
    ls.add_node_operator(Exchange11(rc208))

    cost_evaluator = CostEvaluator(1, 1, 0)
    sol = Solution.make_random(rc208, rng)

    improved = ls.search(sol, cost_evaluator)

    # Now make the instance heterogeneous and update the local search.
    data = rc208.replace(
        vehicle_types=[
            VehicleType(25, capacity=10_000),
            VehicleType(25, capacity=10_000),
        ]
    )

    ls = cpp_LocalSearch(data, neighbours)
    ls.add_node_operator(Exchange10(data))
    ls.add_node_operator(Exchange11(data))

    # Update the improved (locally optimal) solution with vehicles of type 1.
    routes = [Route(data, r.visits(), 1) for r in improved.routes()]
    improved = Solution(data, routes)

    # Doing the search should not find any further improvements thus not change
    # the solution.
    further_improved = ls.search(improved, cost_evaluator)
    assert_equal(further_improved, improved)


def test_bugfix_vehicle_type_offsets(ok_small):
    """
    See https://github.com/PyVRP/PyVRP/pull/292 for details. This exercises a
    fix to a bug that would crash local search due to an incorrect internal
    mapping of vehicle types to route indices if the next vehicle type had
    more vehicles than the previous.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(1, capacity=10),
            VehicleType(2, capacity=10),
        ]
    )

    ls = cpp_LocalSearch(data, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    cost_evaluator = CostEvaluator(1, 1, 0)

    current = Solution(data, [Route(data, [1, 3], 1), Route(data, [2, 4], 1)])
    current_cost = cost_evaluator.penalised_cost(current)

    improved = ls.search(current, cost_evaluator)
    improved_cost = cost_evaluator.penalised_cost(improved)

    assert_(improved_cost <= current_cost)


def test_intensify_overlap_tolerance(rc208):
    """
    Tests that the local search's intensifying route operators respect the
    route overlap tolerance argument.
    """
    rng = RandomNumberGenerator(seed=42)

    neighbours = compute_neighbours(rc208)
    ls = LocalSearch(rc208, rng, neighbours)
    ls.add_route_operator(SwapStar(rc208))

    cost_eval = CostEvaluator(1, 1, 0)
    sol = Solution.make_random(rc208, rng)

    # Overlap tolerance is zero, so no routes should have overlap and thus
    # no intensification should take place.
    unchanged = ls.intensify(sol, cost_eval, overlap_tolerance=0)
    assert_equal(unchanged, sol)

    # But with full overlap tolerance, all routes should be checked. That
    # should lead to an improvement over the random solution.
    better = ls.intensify(sol, cost_eval, overlap_tolerance=1)
    assert_(better != sol)
    assert_(cost_eval.penalised_cost(better) < cost_eval.penalised_cost(sol))


@mark.parametrize("tol", [-1.0, -0.01, 1.01, 10.9, 1000])
def test_intensify_overlap_tolerance_raises_outside_unit_interval(rc208, tol):
    """
    Tests that calling ``intensify()`` raises when the overlap tolerance
    argument is not in [0, 1].
    """
    rng = RandomNumberGenerator(seed=42)

    neighbours = compute_neighbours(rc208)
    ls = LocalSearch(rc208, rng, neighbours)
    ls.add_route_operator(SwapStar(rc208))

    cost_eval = CostEvaluator(1, 1, 0)
    sol = Solution.make_random(rc208, rng)

    with assert_raises(RuntimeError):  # each tolerance value is outside [0, 1]
        ls.intensify(sol, cost_eval, overlap_tolerance=tol)


def test_no_op_results_in_same_solution(ok_small):
    """
    Tests that calling local search without first adding node or route
    operators is a no-op, and returns the same solution as the one that was
    given to it.
    """
    rng = RandomNumberGenerator(seed=42)

    cost_eval = CostEvaluator(1, 1, 0)
    sol = Solution.make_random(ok_small, rng)

    # Empty local search does not actually search anything, so it should return
    # the exact same solution as what was passed in.
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small))
    assert_equal(ls(sol, cost_eval), sol)
    assert_equal(ls.search(sol, cost_eval), sol)
    assert_equal(ls.intensify(sol, cost_eval), sol)


def test_intensify_can_improve_solution_further(rc208):
    """
    Tests that ``intensify()`` improves a solution further once ``search()`` is
    stuck.
    """
    rng = RandomNumberGenerator(seed=11)

    ls = LocalSearch(rc208, rng, compute_neighbours(rc208))
    ls.add_node_operator(Exchange11(rc208))
    ls.add_route_operator(SwapStar(rc208))

    cost_eval = CostEvaluator(1, 1, 0)

    # The following solution is locally optimal w.r.t. the node operators. This
    # solution cannot be improved further by repeated calls to ``search()``.
    search_opt = ls.search(Solution.make_random(rc208, rng), cost_eval)
    search_cost = cost_eval.penalised_cost(search_opt)

    # But it can be improved further using the intensifying route operators,
    # as the following solution shows.
    intensify_opt = ls.intensify(search_opt, cost_eval)
    intensify_cost = cost_eval.penalised_cost(intensify_opt)

    assert_(intensify_cost < search_cost)

    # Both solutions are locally optimal. ``search_opt`` w.r.t. to the node
    # operators, and ``intensify_opt`` w.r.t. to the route operators. Repeated
    # calls to ``search()`` and ``intensify`` do not result in further
    # improvements for such locally optimal solutions.
    for _ in range(10):
        assert_equal(ls.search(search_opt, cost_eval), search_opt)
        assert_equal(ls.intensify(intensify_opt, cost_eval), intensify_opt)


def test_local_search_completes_incomplete_solutions(ok_small_prizes):
    """
    Tests that the local search object improve solutions that are incomplete,
    and returns a completed solution. Passing an incomplete solution should
    return a completed solution after search.
    """
    rng = RandomNumberGenerator(seed=42)

    ls = LocalSearch(ok_small_prizes, rng, compute_neighbours(ok_small_prizes))
    ls.add_node_operator(Exchange10(ok_small_prizes))

    cost_eval = CostEvaluator(1, 1, 0)
    sol = Solution(ok_small_prizes, [[2], [3, 4]])
    assert_(not sol.is_complete())  # 1 is required but not visited

    new_sol = ls.search(sol, cost_eval)
    assert_(new_sol.is_complete())


def test_local_search_does_not_remove_required_clients():
    """
    Tests that the local search object does not remove required clients, even
    when that might result in a significant cost improvement.
    """
    rng = RandomNumberGenerator(seed=42)
    data = ProblemData(
        clients=[
            # This client cannot be removed, even though it causes significant
            # load violations.
            Client(x=1, y=1, delivery=100, required=True),
            # This client can be removed, and should be , because the prize is
            # not worth the detour.
            Client(x=2, y=2, prize=0, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, capacity=50)],
        distance_matrices=[np.where(np.eye(3), 0, 10)],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
    )

    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    sol = Solution(data, [[1, 2]])
    assert_(sol.is_complete())

    # Test that the improved solution contains the first client, but removes
    # the second. The first client is required, so could not be removed, but
    # the second could and that is an improving move.
    cost_eval = CostEvaluator(100, 100, 0)
    new_sol = ls.search(sol, cost_eval)
    assert_equal(new_sol.num_clients(), 1)
    assert_(new_sol.is_complete())

    sol_cost = cost_eval.penalised_cost(sol)
    new_cost = cost_eval.penalised_cost(new_sol)
    assert_(new_cost < sol_cost)


def test_mutually_exclusive_group(gtsp):
    """
    Smoke test that runs the local search on a medium-size TSP instance with
    fifty mutually exclusive client groups.
    """
    assert_equal(gtsp.num_groups, 50)

    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(gtsp)

    ls = LocalSearch(gtsp, rng, neighbours)
    ls.add_node_operator(Exchange10(gtsp))

    sol = Solution.make_random(gtsp, rng)
    cost_eval = CostEvaluator(20, 6, 0)
    improved = ls(sol, cost_eval)

    assert_(not sol.is_group_feasible())
    assert_(improved.is_group_feasible())

    sol_cost = cost_eval.penalised_cost(sol)
    improved_cost = cost_eval.penalised_cost(improved)
    assert_(improved_cost < sol_cost)


def test_mutually_exclusive_group_not_in_solution(
    ok_small_mutually_exclusive_groups,
):
    """
    Tests that the local search inserts a client from the mutually exclusive
    group if the entire group is missing from the solution.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(ok_small_mutually_exclusive_groups)

    ls = LocalSearch(ok_small_mutually_exclusive_groups, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small_mutually_exclusive_groups))

    sol = Solution(ok_small_mutually_exclusive_groups, [[4]])
    assert_(not sol.is_group_feasible())

    improved = ls(sol, CostEvaluator(20, 6, 0))
    assert_(improved.is_group_feasible())


def test_swap_if_improving_mutually_exclusive_group(
    ok_small_mutually_exclusive_groups,
):
    """
    Tests that we swap a client (1) in a mutually exclusive group when another
    client (3) in the group is better to have.
    """
    rng = RandomNumberGenerator(seed=42)
    neighbours = compute_neighbours(ok_small_mutually_exclusive_groups)

    ls = LocalSearch(ok_small_mutually_exclusive_groups, rng, neighbours)
    ls.add_node_operator(Exchange10(ok_small_mutually_exclusive_groups))

    cost_eval = CostEvaluator(20, 6, 0)
    sol = Solution(ok_small_mutually_exclusive_groups, [[1, 4]])
    improved = ls(sol, cost_eval)
    assert_(cost_eval.penalised_cost(improved) < cost_eval.penalised_cost(sol))

    routes = improved.routes()
    assert_equal(improved.num_routes(), 1)
    assert_equal(routes[0].visits(), [3, 4])
