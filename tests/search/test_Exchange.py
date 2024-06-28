import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import (
    Client,
    CostEvaluator,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Solution,
    VehicleType,
)
from pyvrp import Route as SolRoute
from pyvrp.search import (
    Exchange10,
    Exchange11,
    Exchange20,
    Exchange21,
    Exchange22,
    Exchange30,
    Exchange31,
    Exchange32,
    Exchange33,
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
)
from pyvrp.search._search import Node, Route


@pytest.mark.parametrize(
    "operator",
    [
        Exchange11,
        Exchange21,
        Exchange22,
        Exchange31,
        Exchange32,
        Exchange33,
    ],
)
def test_swap_single_route_stays_single_route(rc208, operator):
    """
    Swap operators ((N, M)-exchange operators with M > 0) on a single route can
    only move within the same route, so they can never find a solution that has
    more than one route.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=rc208.num_clients)
    ls = LocalSearch(rc208, rng, compute_neighbours(rc208, nb_params))
    ls.add_node_operator(operator(rc208))

    single_route = list(range(rc208.num_depots, rc208.num_locations))
    sol = Solution(rc208, [single_route])
    improved_sol = ls.search(sol, cost_evaluator)

    # The new solution should strictly improve on our original solution.
    assert_equal(improved_sol.num_routes(), 1)
    current_cost = cost_evaluator.penalised_cost(sol)
    improved_cost = cost_evaluator.penalised_cost(improved_sol)
    assert_(improved_cost < current_cost)


@pytest.mark.parametrize("operator", [Exchange10, Exchange20, Exchange30])
def test_relocate_uses_empty_routes(rc208, operator):
    """
    Unlike the swapping exchange operators, relocate should be able to relocate
    clients to empty routes if that is an improvement.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=rc208.num_clients)
    ls = LocalSearch(rc208, rng, compute_neighbours(rc208, nb_params))
    ls.add_node_operator(operator(rc208))

    single_route = list(range(rc208.num_depots, rc208.num_locations))
    sol = Solution(rc208, [single_route])
    improved_sol = ls.search(sol, cost_evaluator)

    # The new solution should strictly improve on our original solution, and
    # should use more routes.
    assert_(improved_sol.num_routes() > 1)
    current_cost = cost_evaluator.penalised_cost(sol)
    improved_cost = cost_evaluator.penalised_cost(improved_sol)
    assert_(improved_cost < current_cost)


@pytest.mark.parametrize(
    "operator",
    [
        Exchange22,
        Exchange30,
        Exchange31,
        Exchange32,
        Exchange33,
    ],
)
def test_cannot_exchange_when_parts_overlap_with_depot(ok_small, operator):
    """
    (N, M)-exchange works by exchanging N nodes starting at some node U with
    M nodes starting at some node V. But when there is no sequence of N or M
    nodes that does not contain the depot (because the routes are very short),
    then no exchange is possible.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=ok_small.num_clients)
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small, nb_params))
    ls.add_node_operator(operator(ok_small))

    sol = Solution(ok_small, [[1, 2], [3], [4]])
    new_sol = ls.search(sol, cost_evaluator)

    assert_equal(new_sol, sol)


@pytest.mark.parametrize("operator", [Exchange32, Exchange33])
def test_cannot_exchange_when_segments_overlap(ok_small, operator):
    """
    (3, 2)- and (3, 3)-exchange cannot exchange anything on a length-four
    single route solution: there's always overlap between the segments.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=ok_small.num_clients)
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small, nb_params))
    ls.add_node_operator(operator(ok_small))

    sol = Solution(ok_small, [[1, 2, 3, 4]])
    new_sol = ls.search(sol, cost_evaluator)

    assert_equal(new_sol, sol)


def test_cannot_swap_adjacent_segments(ok_small):
    """
    (2, 2)-exchange on a single route cannot swap adjacent segments, since
    that's already covered by (2, 0)-exchange.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=ok_small.num_clients)
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small, nb_params))
    ls.add_node_operator(Exchange22(ok_small))

    # An adjacent swap by (2, 2)-exchange could have created the single-route
    # solution [3, 4, 1, 2], which has a much lower cost. But that's not
    # allowed because adjacent swaps are not allowed.
    sol = Solution(ok_small, [[1, 2, 3, 4]])
    new_sol = ls.search(sol, cost_evaluator)

    assert_equal(new_sol, sol)


def test_swap_between_routes_OkSmall(ok_small):
    """
    On the OkSmall example, (2, 1)-exchange should be able to swap parts of a
    two route solution, resulting in something better.
    """
    cost_evaluator = CostEvaluator(20, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    nb_params = NeighbourhoodParams(nb_granular=ok_small.num_clients)
    ls = LocalSearch(ok_small, rng, compute_neighbours(ok_small, nb_params))
    ls.add_node_operator(Exchange21(ok_small))

    sol = Solution(ok_small, [[1, 2], [3, 4]])
    improved_sol = ls.search(sol, cost_evaluator)
    expected = Solution(ok_small, [[3, 4, 2], [1]])
    assert_equal(improved_sol, expected)

    current_cost = cost_evaluator.penalised_cost(sol)
    improved_cost = cost_evaluator.penalised_cost(improved_sol)
    assert_(improved_cost < current_cost)


def test_relocate_after_depot_should_work(ok_small):
    """
    This test exercises the bug identified in issue #142, involving a relocate
    action that should insert directly after the depot.
    """
    op = Exchange10(ok_small)

    # We create two routes: one with clients [1, 2, 3], and the other empty.
    # It is an improving move to insert client 3 into the empty route.
    route1 = Route(ok_small, idx=0, vehicle_type=0)
    route2 = Route(ok_small, idx=1, vehicle_type=0)

    nodes = [Node(loc=client) for client in [1, 2, 3]]
    for node in nodes:
        route1.append(node)
    route1.update()

    # This solution can be improved by moving 3 into its own route, that is,
    # inserting it after the depot of an empty route. Before the bug was fixed,
    # (1, 0)-exchange never performed this move.
    cost_evaluator = CostEvaluator(20, 6, 0)
    assert_(route1[3] is nodes[-1])
    assert_(op.evaluate(nodes[-1], route2[0], cost_evaluator) < 0)

    assert_(nodes[-1].route is route1)
    assert_equal(len(route1), 3)
    assert_equal(len(route2), 0)

    # Apply the move and check that the routes and nodes are appropriately
    # updated.
    op.apply(nodes[-1], route2[0])
    assert_(nodes[-1].route is route2)
    assert_equal(len(route1), 2)
    assert_equal(len(route2), 1)


def test_relocate_only_happens_when_distance_and_duration_allow_it():
    """
    Tests that (1, 0)-exchange checks the duration matrix for time-window
    feasibility before applying a move that improves the travelled distance.
    """
    # Distance-wise, the best route is 0 -> 1 -> 2 -> 0. Duration-wise,
    # however, the best route is 0 -> 2 -> 1 -> 0.
    data = ProblemData(
        clients=[
            Client(x=1, y=0, tw_early=0, tw_late=5),
            Client(x=2, y=0, tw_early=0, tw_late=5),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(1, tw_early=0, tw_late=10)],
        distance_matrices=[
            np.asarray(
                [
                    [0, 1, 5],
                    [5, 0, 1],
                    [1, 5, 0],
                ]
            ),
        ],
        duration_matrices=[
            np.asarray(
                [
                    [0, 100, 2],
                    [1, 0, 100],
                    [100, 2, 0],
                ]
            ),
        ],
    )

    # We consider two solutions. The first is duration optimal, and overall the
    # only feasible solution. This solution can thus not be improved further.
    # We also consider a distance-optimal solution that is not feasible. Since
    # we have non-zero time warp penalty, this solution should be improved into
    # the duration optimal solution.
    duration_optimal = Solution(data, [[2, 1]])
    distance_optimal = Solution(data, [[1, 2]])

    assert_(distance_optimal.distance() < duration_optimal.distance())
    assert_(duration_optimal.time_warp() < distance_optimal.time_warp())

    cost_evaluator = CostEvaluator(1, 1, 0)
    rng = RandomNumberGenerator(seed=42)
    ls = LocalSearch(data, rng, compute_neighbours(data))
    ls.add_node_operator(Exchange10(data))

    assert_equal(ls.search(duration_optimal, cost_evaluator), duration_optimal)
    assert_equal(ls.search(distance_optimal, cost_evaluator), duration_optimal)


def test_relocate_to_heterogeneous_empty_route(ok_small):
    """
    This test asserts that a customer will be relocated to a non-empty route
    with a different capacity even if there is another empty route in between.
    """
    vehicle_types = [VehicleType(1, capacity=cap) for cap in [12, 5, 1, 3]]
    data = ok_small.replace(vehicle_types=vehicle_types)

    # Use a huge cost for load penalties to make other aspects irrelevant
    cost_evaluator = CostEvaluator(100_000, 6, 0)
    rng = RandomNumberGenerator(seed=42)

    # This is a non-empty neighbourhood (so LS does not complain), but the only
    # client moves allowed by it will not improve the initial solution created
    # below. So the only improvements (1, 0)-exchange can make must come from
    # moving clients behind the depot of a route.
    neighbours = [[] for _ in range(data.num_locations)]
    neighbours[2].append(1)

    ls = LocalSearch(data, rng, neighbours)
    ls.add_node_operator(Exchange10(data))

    # The initial solution has routes with loads [13, 5, 0, 0]
    # with excess [1, 0, 0, 0]. Moving node 3 to route 4 will resolve all
    # load penalties, but other moves would increase load penalties.
    # Therefore, this requires moving to an empty route which is not the first.
    sol = Solution(
        data, [SolRoute(data, [1, 2, 3], 0), SolRoute(data, [4], 1)]
    )
    expected = Solution(
        data,
        [
            SolRoute(data, [1, 2], 0),
            SolRoute(data, [4], 1),
            SolRoute(data, [3], 3),
        ],
    )
    assert_equal(ls.search(sol, cost_evaluator), expected)


@pytest.mark.parametrize(
    ("op", "base_cost", "fixed_cost"),
    [
        (Exchange10, 256, 0),  # inexact; this move shortcuts
        (Exchange10, 256, 100),  # inexact; this move shortcuts
        (Exchange20, 1_417, 0),
        (Exchange20, 1_417, 9),
        (Exchange30, 135, 53),
        (Exchange30, 135, 997),
    ],
)
def test_relocate_fixed_vehicle_cost(ok_small, op, base_cost, fixed_cost):
    """
    Tests that relocate operators - (N, M)-exchange where M == 0 - also take
    into account fixed vehicle costs changes if one of the routes is empty. In
    particular, we fix the base cost of evaluating the route changes (that's
    not changed), and vary the fixed vehicle cost. The total delta cost should
    also vary as a result.
    """
    vehicle_type = VehicleType(2, capacity=10, fixed_cost=fixed_cost)
    data = ok_small.replace(vehicle_types=[vehicle_type])
    op = op(data)

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [2, 4, 1, 3]:
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=0)
    route2.update()

    # First route is not empty, second route is. The operator evaluates moving
    # some nodes to the second route, which would use both of them. That should
    # add to the fixed vehicle cost.
    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(
        op.evaluate(route1[1], route2[0], cost_eval), base_cost + fixed_cost
    )


@pytest.mark.parametrize(
    ("op", "max_dur", "cost"),
    [
        (Exchange20, 0, -4_044),
        (Exchange20, 5_000, 956),
        (Exchange21, 0, -693),
        (Exchange21, 5_000, -596),
    ],
)
def test_exchange_with_max_duration_constraint(ok_small, op, max_dur, cost):
    """
    Tests that the exchange operators correctly evaluate time warp due to
    maximum duration violations.
    """
    vehicle_type = VehicleType(2, capacity=10, max_duration=max_dur)
    data = ok_small.replace(vehicle_types=[vehicle_type])
    op = op(data)

    # Two routes: first route 0 -> 2 -> 4 -> 0, second route 0 -> 1 -> 3 -> 0.
    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [2, 4]:
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=0)
    for loc in [1, 3]:
        route2.append(Node(loc=loc))
    route2.update()

    # Without maximum duration, route1 has a duration of 5_229 and no time warp
    # while route2 has a duration of 5_814 and timewarp 2_087, for a net
    # duration of 5_814 - 2_087 = 3_727 so no violation.
    # Consolidation into a single route may or may not be improving as the
    # total distance decreases but the maximum duration violation increases.
    # Moving from the first to the second route reduces the maximum duration
    # violation in the first route and is typically improving.
    assert_equal(route1.duration(), 5_229)
    assert_equal(route2.duration(), 5_814)

    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(op.evaluate(route1[1], route2[1], cost_eval), cost)


@pytest.mark.parametrize("operator", [Exchange10, Exchange11])
def test_within_route_simultaneous_pickup_and_delivery(operator):
    """
    Tests that the Exchange operators correctly evaluate load violations within
    the same route.
    """
    data = ProblemData(
        clients=[
            Client(x=1, y=0, pickup=5),
            Client(x=2, y=0),
            Client(x=2, y=0, delivery=5),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(capacity=5)],
        distance_matrices=[np.where(np.eye(4), 0, 1)],
        duration_matrices=[np.zeros((4, 4), dtype=int)],
    )

    op = operator(data)

    route = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 2, 3]:
        route.append(Node(loc=loc))
    route.update()

    # Route is 1 -> 2 -> 3, and stores 1's pickup amount (5) before dropping
    # off 3's delivery amount (5). So total load is 10, and the excess load 5.
    assert_(not route.is_feasible())
    assert_equal(route.load(), 10)
    assert_equal(route.excess_load(), 5)

    # For (1, 0)-exchange, we evaluate inserting 1 after 3. That'd resolve the
    # excess load. For (1, 1)-exchange, we evaluate swapping 1 and 3, which
    # would also resolve the excess load: the important bit is that we visit 3
    # before 1.
    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(op.evaluate(route[1], route[3], cost_eval), -5)


@pytest.mark.parametrize(
    ("max_distance", "expected"),
    [
        (5_000, -3_332),  # move reduces max_distance violation
        (2_500, -6_542),  # which becomes more important here
        (0, 18_458),  # but with only violations the move is not improving
    ],
)
def test_relocate_max_distance(ok_small, max_distance: int, expected: int):
    """
    Tests that a relocate move correctly evaluates maximum distance constraint
    violations, and can identify improving moves that increase overall distance
    but reduce the maximum distance violation.
    """
    vehicle_type = VehicleType(2, capacity=10, max_distance=max_distance)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 2]:
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=0)
    route2.update()

    assert_equal(route1.distance(), 5_501)
    assert_equal(route1.excess_distance(), max(5_501 - max_distance, 0))

    cost_eval = CostEvaluator(0, 0, 10)
    op = Exchange10(data)

    # Moving client #2 from route1 to route2 does not improve the overall
    # distance, but can be helpful in reducing maximum distance violations.
    assert_equal(op.evaluate(route1[2], route2[0], cost_eval), expected)
    op.apply(route1[2], route2[0])

    route1.update()
    assert_equal(route1.distance(), 3_270)
    assert_equal(route1.excess_distance(), max(3_270 - max_distance, 0))

    route2.update()
    assert_equal(route2.distance(), 3_909)
    assert_equal(route2.excess_distance(), max(3_909 - max_distance, 0))

    delta_dist = 3_270 + 3_909 - 5_501  # compare manual delta cost
    delta_excess = sum(
        [
            max(3_270 - max_distance, 0),
            max(3_909 - max_distance, 0),
            -max(5_501 - max_distance, 0),
        ]
    )
    assert_equal(delta_dist + 10 * delta_excess, expected)


@pytest.mark.parametrize(
    ("max_distance", "expected"),
    [
        (5_000, -5_222),
        (2_500, -6_072),  # both routes now violate max dist constraint
        (0, -6_072),  # so tighter constraints do not improve anything
    ],
)
def test_swap_max_distance(ok_small, max_distance: int, expected: int):
    """
    Tests that a swap move correctly evaluates maximum distance constraint
    violations, and can identify improving moves that increase overall distance
    but reduce the maximum distance violation.
    """
    vehicle_type = VehicleType(2, capacity=10, max_distance=max_distance)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 2]:
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=0)
    route2.append(Node(loc=3))
    route2.update()

    assert_equal(route1.distance(), 5_501)
    assert_equal(route1.excess_distance(), max(5_501 - max_distance, 0))

    assert_equal(route2.distance(), 3_994)
    assert_equal(route2.excess_distance(), max(3_994 - max_distance, 0))

    cost_eval = CostEvaluator(0, 0, 10)
    op = Exchange11(data)

    # Swapping client #2 in route1 and client #3 in route2 improves the overall
    # distance and reduces the excess distance violations.
    assert_equal(op.evaluate(route1[2], route2[1], cost_eval), expected)
    op.apply(route1[2], route2[1])

    route1.update()
    assert_equal(route1.distance(), 5_034)
    assert_equal(route1.excess_distance(), max(5_034 - max_distance, 0))

    route2.update()
    assert_equal(route2.distance(), 3_909)
    assert_equal(route2.excess_distance(), max(3_909 - max_distance, 0))

    delta_dist = 5_034 + 3_909 - 5_501 - 3_994  # compare manual delta cost
    delta_excess = sum(
        [
            max(5_034 - max_distance, 0),
            max(3_909 - max_distance, 0),
            -max(5_501 - max_distance, 0),
            -max(3_994 - max_distance, 0),
        ]
    )
    assert_equal(delta_dist + 10 * delta_excess, expected)


def test_swap_with_different_profiles(ok_small_two_profiles):
    """
    Tests that swap correctly evaluates moves between routes with different
    profiles.
    """
    data = ok_small_two_profiles

    route1 = Route(data, idx=0, vehicle_type=0)
    route1.append(Node(loc=3))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=4))
    route2.update()

    op = Exchange11(data)
    cost_eval = CostEvaluator(0, 0, 0)  # all zero so no costs from penalties

    # This move evaluates swapping loc 3 and 4 between routes. The cost delta
    # is as follows, taking into account the different profiles' distances.
    dist1, dist2 = data.distance_matrices()
    delta = dist1[0, 4] + dist1[4, 0] + dist2[0, 3] + dist2[3, 0]
    delta -= route1.distance() + route2.distance()
    assert_equal(op.evaluate(route1[1], route2[1], cost_eval), delta)
