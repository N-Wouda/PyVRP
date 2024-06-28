import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import Client, CostEvaluator, Depot, ProblemData, VehicleType
from pyvrp.search import SwapRoutes
from pyvrp.search._search import Node, Route


@pytest.mark.parametrize(
    ("visits1", "visits2"),
    [
        ([], []),  # both empty
        ([1], []),  # first non-empty, second empty
        ([], [1]),  # first empty, second non-empty
        ([1], [2, 3]),  # both non-empty but unequal length
        ([2, 3], [1]),  # both non-empty but unequal length (flipped)
        ([2, 3], [1, 4]),  # both non-empty equal length
    ],
)
def test_apply(ok_small, visits1: list[int], visits2: list[int]):
    """
    Tests that applying SwapRoutes to two different routes indeed exchanges
    the visits.
    """
    route1 = Route(ok_small, idx=0, vehicle_type=0)
    for loc in visits1:
        route1.append(Node(loc=loc))

    route2 = Route(ok_small, idx=1, vehicle_type=0)
    for loc in visits2:
        route2.append(Node(loc=loc))

    route1.update()
    route2.update()

    # Before calling apply, route1 visits the clients in visits1, and route2
    # visits the clients in visits2.
    assert_equal(visits1, [node.client for node in route1])
    assert_equal(visits2, [node.client for node in route2])

    op = SwapRoutes(ok_small)
    op.apply(route1, route2)

    # But after apply, the visits are now swapped.
    assert_equal(visits2, [node.client for node in route1])
    assert_equal(visits1, [node.client for node in route2])


def test_evaluate_same_vehicle_type(ok_small):
    """
    Tests that evaluate() returns 0 in case the same vehicle types are used,
    since in that case swapping cannot result in cost savings.
    """
    route1 = Route(ok_small, idx=0, vehicle_type=0)
    route2 = Route(ok_small, idx=1, vehicle_type=0)
    assert_equal(route1.vehicle_type, route2.vehicle_type)

    route1.append(Node(loc=1))
    route2.append(Node(loc=2))

    route1.update()
    route2.update()

    op = SwapRoutes(ok_small)
    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(op.evaluate(route1, route2, cost_eval), 0)


def test_same_route(ok_small):
    """
    Tests that evaluate() returns 0 in case the same routes are passed in,
    since then swapping has no effect.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    route.append(Node(loc=1))
    route.update()

    op = SwapRoutes(ok_small)
    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(op.evaluate(route, route, cost_eval), 0)


def test_evaluate_empty_routes(ok_small):
    """
    Tests that evaluate() returns 0 when one or both of the routes are empty.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(3, capacity=10),
            VehicleType(3, capacity=10),
        ]
    )

    route1 = Route(data, idx=0, vehicle_type=0)
    route2 = Route(data, idx=1, vehicle_type=1)
    route3 = Route(data, idx=2, vehicle_type=0)

    route1.append(Node(loc=1))

    route1.update()
    route2.update()

    op = SwapRoutes(data)
    cost_eval = CostEvaluator(1, 1, 0)

    # Vehicle types are no longer the same, but one of the routes is empty.
    # That situation is not currently handled.
    assert_(route1.vehicle_type != route2.vehicle_type)
    assert_equal(op.evaluate(route1, route2, cost_eval), 0)
    assert_equal(op.evaluate(route2, route1, cost_eval), 0)

    # Both routes are empty, but of different vehicle type as well.
    assert_equal(len(route2), len(route3))
    assert_equal(op.evaluate(route3, route2, cost_eval), 0)


def test_evaluate_capacity_differences(ok_small):
    """
    Tests that changes in vehicle capacity violations are evaluated correctly.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(capacity=10), VehicleType(capacity=20)]
    )

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 2, 4]:
        route1.append(Node(loc=loc))

    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=3))

    route1.update()
    route2.update()

    # route1 has vehicle type 0, which has capacity 10. So there is excess load
    # since its client delivery demand sums to 15.
    assert_(route1.has_excess_load())
    assert_equal(route1.load(), 15)

    # route2, on the other hand, has capacity 20 and a load of only 3.
    assert_(not route2.has_excess_load())
    assert_equal(route2.load(), 3)

    op = SwapRoutes(data)
    cost_eval = CostEvaluator(40, 1, 0)

    # Swapping the route plans should alleviate the excess load, since the load
    # of 15 on route1 is below route2's capacity, and similarly for route2's
    # load and route1's capacity. Since we price unit load violations at 40,
    # this should result in a delta cost of -200.
    assert_equal(op.evaluate(route1, route2, cost_eval), -200)

    # Apply the move, update the routes, and then check if they're now both
    # feasible.
    op.apply(route1, route2)
    route1.update()
    route2.update()

    assert_equal(len(route1), 1)
    assert_(route1.is_feasible())

    assert_equal(len(route2), 3)
    assert_(route2.is_feasible())


def test_evaluate_shift_time_window_differences(ok_small):
    """
    Tests that SwapRoutes correctly evaluates changes in time warp due to
    different shift time windows.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(capacity=10, tw_early=10_000, tw_late=15_000),
            VehicleType(capacity=10, tw_early=15_000, tw_late=20_000),
        ]
    )

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 4]:  # depot -> 1 -> 4 -> depot
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=1)
    for loc in [3, 2]:  # depot -> 3 -> 2 -> depot
        route2.append(Node(loc=loc))
    route2.update()

    # Without shift time windows, both routes are feasible, and there is slack
    # on either route: the first route can start between [14'056, 16'003], and
    # the second between [9'002, 13'369]. Neither route can complete its visits
    # within the shift time window of its assigned vehicle type. However, the
    # other type has a shift duration that is much better aligned with its
    # route. Thus, we should have that swapping the vehicle types results in
    # a lower cost, due to decreased time warp on the routes.
    op = SwapRoutes(data)
    cost_eval = CostEvaluator(1, 1, 0)
    assert_(op.evaluate(route1, route2, cost_eval) < 0)


def test_evaluate_max_duration_constraints(ok_small):
    """
    Tests that SwapRoutes correctly evaluates changes in time warp due to
    different maximum duration constraints.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(capacity=10, max_duration=3_000),
            VehicleType(capacity=10),
        ]
    )

    route1 = Route(data, idx=0, vehicle_type=0)
    for loc in [1, 4]:  # depot -> 1 -> 4 -> depot
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(data, idx=1, vehicle_type=1)
    for loc in [3, 2]:  # depot -> 3 -> 2 -> depot
        route2.append(Node(loc=loc))
    route2.update()

    # First route takes 5'332, which is 2'332 more than its maximum duration
    # allows. There is no other source of time warp, so the total route time
    # warp must be 2'332.
    assert_equal(route1.duration(), 5_332)
    assert_equal(route1.time_warp(), 2_332)

    # Second route takes 5'323, and has no maximum duration constraint. There
    # is no other source of time warp, so total route time warp must be zero.
    assert_equal(route2.duration(), 5_323)
    assert_equal(route2.time_warp(), 0)

    # Swapping the routes results in a reduction of 5'332 - 5'323 = 9 units of
    # time warp.
    op = SwapRoutes(data)
    cost_eval = CostEvaluator(1, 1, 0)
    assert_equal(op.evaluate(route1, route2, cost_eval), -9)


def test_evaluate_with_different_depots():
    """
    Tests that SwapRoutes correctly evaluates distance changes due to different
    start and end depots of different vehicle types.
    """
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

    # First route is first depot -> second client -> first depot.
    route1 = Route(data, idx=0, vehicle_type=0)
    route1.append(Node(loc=3))
    route1.update()

    # Second route is second depot -> first client -> second depot.
    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=2))
    route2.update()

    op = SwapRoutes(data)
    cost_eval = CostEvaluator(1, 1, 0)

    # The routes each cost 16 distance which is not as efficient as swapping
    # them, as that would reduce each route's cost to 4, for an improvement
    # of 2 * 12 = 24.
    assert_equal(route1.distance(), 16)
    assert_equal(route2.distance(), 16)
    assert_equal(op.evaluate(route1, route2, cost_eval), -24)


def test_different_objectives(ok_small_multi_depot):
    """
    Tests that swapping between routes with vehicles having different objective
    coefficients correctly evaluates the resulting cost delta.
    """
    vehicle_types = [
        VehicleType(unit_duration_cost=0),
        VehicleType(
            start_depot=1,
            end_depot=1,
            unit_distance_cost=0,
            unit_duration_cost=1,
        ),
    ]

    data = ok_small_multi_depot.replace(vehicle_types=vehicle_types)
    op = SwapRoutes(data)
    cost_eval = CostEvaluator(1, 1, 0)

    # First route is first depot -> second client -> first depot.
    route1 = Route(data, idx=0, vehicle_type=0)
    route1.append(Node(loc=3))
    route1.update()

    assert_equal(route1.distance_cost(), 3_994)
    assert_equal(route1.duration_cost(), 0)

    # Second route is second depot -> first client -> second depot.
    route2 = Route(data, idx=1, vehicle_type=1)
    route2.append(Node(loc=2))
    route2.update()

    assert_equal(route2.distance_cost(), 0)
    assert_equal(route2.duration_cost(), 4_327)

    # The proposed new distance (on route1) is  dist(0, 2) + dist(2, 0) =
    # 3_909, so its cost contribution is also 3_909. The proposed new duration
    # (on route2) is duration(1, 3) + service(3) + duration(3, 1) = 3_280, with
    # again a cost contribution of 3_280.
    delta_cost = op.evaluate(route1, route2, cost_eval)
    assert_equal(delta_cost, 3_909 + 3_280 - 3_994 - 4_327)
