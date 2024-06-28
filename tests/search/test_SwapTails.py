import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import mark

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
from pyvrp.search import LocalSearch, SwapTails
from pyvrp.search._search import Node, Route


@mark.parametrize(
    "vehicle_types",
    [
        [VehicleType(capacity=10), VehicleType(capacity=10)],
        [VehicleType(capacity=8), VehicleType(capacity=10)],
        [VehicleType(capacity=10), VehicleType(capacity=8)],
        [VehicleType(capacity=9), VehicleType(capacity=9)],
        [VehicleType(capacity=8), VehicleType(capacity=8)],
    ],
)
def test_OkSmall_multiple_vehicle_types(
    ok_small, vehicle_types: list[VehicleType]
):
    """
    This test evaluates a move that is improving for some of the vehicle types,
    and not for others. Because the granular neighbourhood is very constrained,
    only a single move can be applied. When it's better to do so, it should
    have been applied, and when it's not better, it should not.
    """
    data = ok_small.replace(vehicle_types=vehicle_types)

    cost_evaluator = CostEvaluator(10_000, 6, 0)  # large load penalty
    rng = RandomNumberGenerator(seed=42)

    neighbours: list[list[int]] = [[], [2], [], [], []]  # only 1 -> 2
    ls = LocalSearch(data, rng, neighbours)
    ls.add_node_operator(SwapTails(data))

    routes1 = [SolRoute(data, [1, 3], 0), SolRoute(data, [2, 4], 1)]
    sol1 = Solution(data, routes1)

    routes2 = [SolRoute(data, [1, 4], 0), SolRoute(data, [2, 3], 1)]
    sol2 = Solution(data, routes2)

    cost1 = cost_evaluator.penalised_cost(sol1)
    cost2 = cost_evaluator.penalised_cost(sol2)
    assert_(not np.allclose(cost1, cost2))

    # Using the local search, the result should not get worse.
    improved_sol1 = ls.search(sol1, cost_evaluator)
    improved_sol2 = ls.search(sol2, cost_evaluator)

    expected_sol = sol1 if cost1 < cost2 else sol2
    assert_equal(improved_sol1, expected_sol)
    assert_equal(improved_sol2, expected_sol)


def test_move_involving_empty_routes():
    """
    This test checks that a SwapTails move that turns an empty route into a
    non-empty route (or vice-versa) is correctly evaluated.
    """
    data = ProblemData(
        clients=[Client(x=1, y=1), Client(x=1, y=0)],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[
            VehicleType(fixed_cost=10),
            VehicleType(fixed_cost=100),
        ],
        distance_matrices=[np.zeros((3, 3), dtype=int)],
        duration_matrices=[np.zeros((3, 3), dtype=int)],
    )

    route1 = Route(data, idx=0, vehicle_type=0)
    route2 = Route(data, idx=1, vehicle_type=1)

    for loc in [1, 2]:
        route1.append(Node(loc=loc))

    route1.update()  # depot -> 1 -> 2 -> depot
    route2.update()  # depot -> depot

    op = SwapTails(data)
    cost_eval = CostEvaluator(0, 0, 0)

    # This move does not change the route structure, so the delta cost is 0.
    assert_equal(op.evaluate(route1[2], route2[0], cost_eval), 0)

    # This move creates routes (depot -> 1 -> depot) and (depot -> 2 -> depot),
    # making route 2 non-empty and thus incurring its fixed cost of 100.
    assert_equal(op.evaluate(route1[1], route2[0], cost_eval), 100)

    # This move creates routes (depot -> depot) and (depot -> 1 -> 2 -> depot),
    # making route 1 empty, while making route 2 non-empty. The total fixed
    # cost incurred is thus -10 + 100 = 90.
    assert_equal(op.evaluate(route1[0], route2[0], cost_eval), 90)

    # Now we reverse the visits of route 1 and 2, so that we can hit the cases
    # where route 1 is empty.
    route1.clear()

    for loc in [1, 2]:
        route2.append(Node(loc=loc))

    route1.update()  # depot -> depot
    route2.update()  # depot -> 1 -> 2 -> depot

    # This move does not change the route structure, so the delta cost is 0.
    assert_equal(op.evaluate(route1[0], route2[2], cost_eval), 0)

    # This move creates routes (depot -> 2 -> depot) and (depot -> 1 -> depot),
    # making route 1 non-empty and thus incurring its fixed cost of 10.
    assert_equal(op.evaluate(route1[0], route2[1], cost_eval), 10)

    # This move creates routes (depot -> 1 -> 2 -> depot) and (depot -> depot),
    # making route 1 non-empty, while making route 2 empty. The total fixed
    # cost incurred is thus 10 - 100 = -90.
    assert_equal(op.evaluate(route1[0], route2[0], cost_eval), -90)


def test_move_involving_multiple_depots():
    """
    Tests that SwapTails correctly evaluates distance changes due to different
    start and end depots of different vehicle types.
    """
    # Locations with indices 0 and 1 are depots, with 2 and 3 are clients.
    data = ProblemData(
        clients=[Client(x=1, y=1), Client(x=4, y=4)],
        depots=[Depot(x=0, y=0), Depot(x=5, y=5)],
        vehicle_types=[
            VehicleType(start_depot=0, end_depot=0),
            VehicleType(start_depot=1, end_depot=1),
        ],
        distance_matrices=[
            [
                [0, 10, 2, 8],
                [10, 0, 8, 2],
                [2, 8, 0, 6],
                [8, 2, 6, 0],
            ]
        ],
        duration_matrices=[np.zeros((4, 4), dtype=int)],
    )

    # First route is 0 -> 3 -> 0.
    route1 = Route(data, idx=0, vehicle_type=0)
    route1.append(Node(loc=3))
    route1.update()

    # Second route is 1 -> 2 -> 1.
    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=2))
    route2.update()

    assert_equal(route1.distance(), 16)
    assert_equal(route2.distance(), 16)

    op = SwapTails(data)
    cost_eval = CostEvaluator(1, 1, 0)

    assert_equal(op.evaluate(route1[1], route2[1], cost_eval), 0)  # no-op

    # First would be 0 -> 3 -> 2 -> 0, second 1 -> 1. Distance on route2 would
    # be zero, and on route1 16. Thus delta cost is -16.
    assert_equal(op.evaluate(route1[1], route2[0], cost_eval), -16)

    # First would be 0 -> 0, second 1 -> 2 -> 3 -> 1. Distance on route1 would
    # be zero, and on route2 16. Thus delta cost is -16.
    assert_equal(op.evaluate(route1[0], route2[1], cost_eval), -16)


def test_move_with_different_profiles(ok_small_two_profiles):
    """
    Tests that SwapTails correctly evaluates moves between routes with
    different profiles.
    """
    data = ok_small_two_profiles
    dist1, dist2 = data.distance_matrices()

    route1 = Route(data, idx=0, vehicle_type=0)
    route1.append(Node(loc=3))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=2))
    route2.update()

    op = SwapTails(data)
    cost_eval = CostEvaluator(0, 0, 0)  # all zero so no costs from penalties

    # First route has profile 0, and its distance is thus computed using the
    # first distance matrix.
    assert_equal(route1.profile(), 0)
    assert_equal(route1.distance(), dist1[0, 3] + dist1[3, 0])

    # Second route has profile 1, and its distance is thus computed using the
    # second distance matrix.
    assert_equal(route2.profile(), 1)
    assert_equal(route2.distance(), dist2[0, 2] + dist2[2, 0])

    # This move evaluates the setting where the second route would be empty,
    # and the first becomes 0 -> 3 -> 2 -> 0.
    delta = dist1[3, 2] + dist1[2, 0] - dist1[3, 0] - route2.distance()
    assert_equal(op.evaluate(route1[1], route2[0], cost_eval), delta)

    # This move evaluates the setting where the first route would be empty, and
    # the second becomes 0 -> 2 -> 3 -> 0.
    delta = dist2[2, 3] + dist2[3, 0] - dist2[2, 0] - route1.distance()
    assert_equal(op.evaluate(route1[0], route2[1], cost_eval), delta)
