import pickle
from copy import copy, deepcopy

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

from pyvrp import (
    Client,
    ClientGroup,
    Depot,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
    VehicleType,
)
from tests.helpers import read


@pytest.mark.parametrize(
    "routes",
    [
        [[3, 4], [1, 2], []],
        [[3, 4], [], [1, 2]],
    ],
)
def test_route_constructor_raises_for_empty_routes(ok_small, routes):
    """
    Tests that constructing a ``Solution`` with empty routes fails.
    """
    with assert_raises(RuntimeError):
        Solution(ok_small, routes)


def test_route_constructor_with_different_vehicle_types(ok_small):
    """
    Tests that Solution's route constructor respects vehicle types.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(capacity=10), VehicleType(2, capacity=20)]
    )

    sol = Solution(data, [Route(data, [3, 4], 0), Route(data, [1, 2], 1)])

    # We expect Solution to return routes with the correct vehicle types.
    routes = sol.routes()
    assert_equal(len(routes), 2)

    assert_equal(routes[0].visits(), [3, 4])
    assert_equal(routes[0].vehicle_type(), 0)
    assert_equal(routes[0], Route(data, [3, 4], 0))

    assert_equal(routes[1].visits(), [1, 2])
    assert_equal(routes[1].vehicle_type(), 1)
    assert_equal(routes[1], Route(data, [1, 2], 1))


def test_route_depot_accessors(ok_small_multi_depot):
    """
    Tests that Route's start_depot() and end_depot() members return the correct
    depot location indices.
    """
    routes = [
        Route(ok_small_multi_depot, [2], 0),
        Route(ok_small_multi_depot, [3, 4], 1),
    ]

    assert_equal(routes[0].start_depot(), 0)
    assert_equal(routes[0].end_depot(), 0)
    assert_equal(routes[1].start_depot(), 1)
    assert_equal(routes[1].end_depot(), 1)


def test_route_eq(ok_small):
    """
    Tests ``Route``'s equality operator.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(capacity=10), VehicleType(2, capacity=20)]
    )

    route1 = Route(data, [1, 2], 0)
    assert_(route1 == route1)  # should equal self

    route2 = Route(data, [1, 2], 0)
    assert_equal(route1, route2)  # same route/vehicle type; different object

    route3 = Route(data, [1, 2], 1)
    assert_(route2 != route3)  # different vehicle type

    route4 = Route(data, [2, 1], 0)
    assert_(route2 != route4)  # same vehicle type, different visits

    assert_(route1 != "test")
    assert_(route1 != 0)
    assert_(route1 != -1.0)


def test_random_constructor_cycles_over_routes(ok_small):
    """
    Tests that a randomly constructed solution fills all available vehicles
    in turn.
    """
    # The OkSmall instance has four clients and three vehicles. Since 1 client
    # per vehicle would not work (insufficient vehicles), each route is given
    # two clients (and the last route should be empty).
    rng = RandomNumberGenerator(seed=42)

    sol = Solution.make_random(ok_small, rng)
    routes = sol.routes()

    assert_equal(sol.num_routes(), 2)
    assert_equal(len(routes), 2)

    for idx, size in enumerate([2, 2]):
        assert_equal(len(routes[idx]), size)


@pytest.mark.parametrize("num_vehicles", (4, 5, 1_000))
def test_random_constructor_uses_all_routes(ok_small, num_vehicles):
    """
    Tests that the randomly constructed solution has exactly as many routes as
    the number of clients when there are sufficient vehicles available.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(num_vehicles, capacity=10)]
    )
    assert_equal(data.num_clients, 4)

    rng = RandomNumberGenerator(seed=42)
    sol = Solution.make_random(data, rng)
    routes = sol.routes()

    for route in routes:
        assert_equal(len(route), 1)

    assert_equal(sol.num_routes(), data.num_clients)
    assert_equal(len(routes), data.num_clients)


def test_random_constructor_uses_all_vehicle_types(ok_small):
    """
    Tests that vehicles of each available vehicle type are used when there are
    more vehicles than clients.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(ok_small.num_clients, capacity=10),
            VehicleType(ok_small.num_clients, capacity=10),
        ]
    )

    rng = RandomNumberGenerator(seed=42)
    sol = Solution.make_random(data, rng)

    used_vehicle_types = {route.vehicle_type() for route in sol.routes()}
    assert_equal(used_vehicle_types, {0, 1})


def test_random_constructor_randomly_selects_optional_clients(
    ok_small_prizes,
):
    """
    Tests that the randomly constructed solution selects optional clients
    uniformly at random.
    """
    rng = RandomNumberGenerator(seed=42)

    sols = [Solution.make_random(ok_small_prizes, rng) for _ in range(100)]
    avg_num_clients = np.mean([sol.num_clients() for sol in sols])

    # There are 4 clients, three of which are optional, so the average number
    # of clients should be close to 2.5.
    assert_allclose(avg_num_clients, 2.5, atol=0.1)


def test_route_constructor_raises_too_many_vehicles(ok_small):
    """
    Tests that constructing a solution with more routes than available in the
    data instance raises.
    """
    assert_equal(ok_small.num_vehicles, 3)

    # Only two routes should not raise.
    sol = Solution(ok_small, [[1, 2], [4, 3]])
    assert_equal(len(sol.routes()), 2)

    # Three routes should not raise.
    Solution(ok_small, [[1, 2], [4], [3]])

    # More than three routes should raise, since we only have three vehicles.
    with assert_raises(RuntimeError):
        Solution(ok_small, [[1], [2], [3], [4]])

    # Now test the case with multiple vehicle types.
    data = ok_small.replace(
        vehicle_types=[VehicleType(2, capacity=10), VehicleType(capacity=20)]
    )

    # Only two routes (of type 0) should not raise.
    sol = Solution(data, [[1, 2], [4, 3]])
    assert_equal(len(sol.routes()), 2)

    # One route of both vehicle types should not raise.
    sol = Solution(data, [Route(data, [1, 2], 0), Route(data, [4, 3], 1)])
    assert_equal(len(sol.routes()), 2)

    # Two routes of type 1 and one of type 2 should not raise as we have those.
    sol = Solution(
        data,
        [Route(data, [1], 0), Route(data, [2], 0), Route(data, [4, 3], 1)],
    )
    assert_equal(len(sol.routes()), 3)

    # Two routes of vehicle type 1 should raise since we have only one.
    with assert_raises(RuntimeError):
        sol = Solution(data, [Route(data, [1, 2], 1), Route(data, [4, 3], 1)])

    # Three routes should raise since they are considered to be type 0.
    with assert_raises(RuntimeError):
        Solution(data, [[1, 2], [4], [3]])


def test_route_constructor_raises_for_multiple_visits(ok_small):
    """
    Tests that visiting the same client more than once raises.
    """
    with assert_raises(RuntimeError):
        Solution(ok_small, [[1, 2], [1, 3, 4]])  # client 1 is visited twice

    with assert_raises(RuntimeError):
        Solution(
            ok_small, [[1, 2], [1, 3, 4], [1]]
        )  # client 1 is visited thrice


def test_route_constructor_allows_incomplete_solutions(ok_small_prizes):
    """
    Tests that not visiting a client at all is allowed, but turns the solution
    infeasible (unless the client is not required). Allowing this is useful
    for both prize collecting, and in LNS settings where an incomplete solution
    is subsequently repaired.
    """
    # Client 1 is required but not visited.
    sol = Solution(ok_small_prizes, [[2], [3, 4]])
    assert_(not sol.is_complete())
    assert_(not sol.is_feasible())

    # All required clients are visited, but the solution is not feasible.
    sol = Solution(ok_small_prizes, [[1], [2, 3, 4]])
    assert_(not sol.is_feasible())
    assert_(sol.is_complete())

    # All required clients are visited and the solution is feasible.
    sol = Solution(ok_small_prizes, [[1]])
    assert_(sol.is_feasible())
    assert_(sol.is_complete())


def test_neighbours(ok_small):
    """
    Tests that the neighbour structure of (pred, succ) pairs for each client in
    the solution works correctly.
    """
    assert_equal(ok_small.num_clients, 4)

    sol = Solution(ok_small, [[3], [1, 2]])  # client 4 is not in the solution
    assert_(not sol.is_complete())

    neighbours = sol.neighbours()
    expected = [
        None,  # 0: is depot
        (0, 2),  # 1: between depot (0) and 2
        (1, 0),  # 2: between 1 and depot (0)
        (0, 0),  # 3: between depot (0) and depot (0)
        None,  # 4: unassigned
    ]

    for loc in range(ok_small.num_locations):
        assert_equal(neighbours[loc], expected[loc])


def test_neighbours_multi_depot(ok_small):
    """
    Tests that the neighbour structure of (pred, succ) pairs for each client in
    the solution works correctly when there are multiple depots.
    """
    # Make a two-depot instance by changing the first client in ok_small into
    # a depot, and adding a vehicle type that operates out of that depot.
    locations = ok_small.depots() + ok_small.clients()
    locations[1] = Depot(locations[1].x, locations[1].y)

    data = ok_small.replace(
        depots=locations[:2],
        clients=locations[2:],
        vehicle_types=[
            VehicleType(start_depot=0, end_depot=0),
            VehicleType(start_depot=1, end_depot=1),
        ],
    )

    sol = Solution(data, [Route(data, [4], 0), Route(data, [2, 3], 1)])
    assert_(sol.is_complete())

    neighbours = sol.neighbours()
    expected = [
        None,  # 0: is depot
        None,  # 1: is depot
        (1, 3),  # 2: between depot (1) and 3
        (2, 1),  # 3: between 2 and depot (1)
        (0, 0),  # 4: between depot (0) and depot (0)
    ]

    for loc in range(ok_small.num_locations):
        assert_equal(neighbours[loc], expected[loc])


def test_feasibility(ok_small):
    """
    Tests that solutions are infeasible when they have load or time window
    violations.
    """
    # This solution is infeasible due to both load and time window violations.
    sol = Solution(ok_small, [[1, 2, 3, 4]])
    assert_(not sol.is_feasible())

    # First route has total load 18, but vehicle capacity is only 10.
    assert_(sol.has_excess_load())

    # Client 4 has TW [8400, 15300], but client 2 cannot be visited before
    # 15600, so there must be time warp on the single-route solution.
    assert_(sol.has_time_warp())

    # Let's try another solution that's actually feasible.
    sol = Solution(ok_small, [[1, 2], [3], [4]])
    assert_(sol.is_feasible())
    assert_(not sol.has_excess_load())
    assert_(not sol.has_time_warp())


def test_feasibility_release_times():
    """
    Tests solutions can be infeasible due to release time violations, which
    adds time warp.
    """
    data = read("data/OkSmallReleaseTimes.txt")

    # Client 1 is released at 20'000, but client 2 time window ends at 19'500,
    # so this solution must be infeasible due to time-warping. We arrive at
    # client 1 at time 20'000 + 1'544 = 21'544 before the TW closes (22'500).
    # We arrive at client 2 at 21'544 + 360 + 1'992 = 23'896, so we incur a
    # time warp of 23'896 - 19'500 = 4'396.
    sol = Solution(data, [[1, 2], [3], [4]])
    assert_(not sol.is_feasible())
    assert_equal(sol.time_warp(), 4396)

    # Visiting clients 2 and 3 together is feasible: both clients are released
    # at time 5'000. We arrive at client 2 at 5'000 + 1'944 and wait till the
    # TW opens (12'000). We arrive at client 3 at 12'000 + 360 + 621 = 12'981,
    # which is before the TW closes (15'300).
    sol = Solution(data, [[1], [2, 3], [4]])
    assert_(sol.is_feasible())


def test_feasibility_max_duration(ok_small):
    """
    Tests that the maximum duration constraint can affect the feasibility of
    particular solutions.
    """
    # First check that these two routes are feasible when there is no maximum
    # duration constraint.
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    assert_(sol.is_feasible())

    # Modify the data to impose a maximum route duration constraint of 3'000,
    # and check that the previously feasible solution is now not feasible.
    vehicle_type = VehicleType(4, capacity=10, max_duration=3_000)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    sol = Solution(data, [[1, 2], [3, 4]])
    routes = sol.routes()

    # First route has duration 6'221, and the second route duration 5'004.
    # Since the maximum duration is 3'000, these routes incur time warp of
    # 3'221 + 2'004 = 5'225, and the solution is thus no longer feasible.
    assert_equal(routes[0].duration(), 6_221)
    assert_equal(routes[1].duration(), 5_004)
    assert_equal(sol.time_warp(), 5_225)

    assert_(not routes[0].is_feasible())
    assert_(not routes[1].is_feasible())
    assert_(not sol.is_feasible())


def test_feasibility_max_distance(ok_small):
    """
    Tests that the maximum distance constraint affects solution and route
    feasibility when it is violated.
    """
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    assert_(sol.is_feasible())

    vehicle_type = VehicleType(4, capacity=10, max_distance=5_000)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    sol = Solution(data, [[1, 2], [3, 4]])
    routes = sol.routes()

    assert_equal(routes[0].distance(), 5501)
    assert_equal(routes[0].excess_distance(), 501)
    assert_(not routes[0].has_time_warp())
    assert_(not routes[0].is_feasible())

    assert_equal(routes[1].distance(), 4224)
    assert_equal(routes[1].excess_distance(), 0)
    assert_(routes[1].is_feasible())

    assert_equal(sol.excess_distance(), 501)
    assert_(sol.has_excess_distance())
    assert_(not sol.is_feasible())


def test_distance_calculation(ok_small):
    """
    Tests that route distance calculations are correct, and that the overall
    Solution's distance is the sum of the route distances.
    """
    sol = Solution(ok_small, [[1, 2], [3], [4]])
    routes = sol.routes()

    # Solution is feasible, so all its routes should also be feasible.
    assert_(sol.is_feasible())
    assert_(all(route.is_feasible() for route in routes))

    # Solution distance should be equal to all routes' distances. These we
    # check separately.
    assert_equal(sol.distance(), sum(route.distance() for route in routes))

    distances = ok_small.distance_matrix(profile=0)
    expected = distances[0, 1] + distances[1, 2] + distances[2, 0]
    assert_equal(routes[0].distance(), expected)

    expected = distances[0, 3] + distances[3, 0]
    assert_equal(routes[1].distance(), expected)

    expected = distances[0, 4] + distances[4, 0]
    assert_equal(routes[2].distance(), expected)


def test_excess_load_calculation(ok_small):
    """
    Tests the Solution's excess load calculation on a single-route case.
    """
    sol = Solution(ok_small, [[4, 3, 1, 2]])
    assert_(sol.has_excess_load())
    assert_(not sol.has_time_warp())

    # All clients are visited on the same route/by the same vehicle. The total
    # delivery demand is 18, but the vehicle capacity is only 10.
    assert_equal(sol.excess_load(), 18 - ok_small.vehicle_type(0).capacity)


def test_excess_load_calculation_with_multiple_vehicle_capacities(ok_small):
    """
    Tests that vehicles of different capacities result in different (excess)
    load calaculations.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(2, capacity=10), VehicleType(capacity=20)]
    )

    # This instance has capacities 10 and 20 for vehicle type 0 and 1. The
    # total delivery demand is 18 so if all demand is put in vehicle type 0 the
    # excess_load is 18 - 10 = 8.
    sol = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    assert_(sol.has_excess_load())
    assert_equal(sol.excess_load(), 8)

    # With vehicle type 1, the capacity 20 is larger than 18.
    sol = Solution(data, [Route(data, [1, 2, 3, 4], 1)])
    assert_(not sol.has_excess_load())
    assert_equal(sol.excess_load(), 0)


def test_route_access_methods(ok_small):
    """
    Tests that accessing route statistics returns the correct numbers.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.routes()

    # Test route access: getting the route plan should return a simple list, as
    # given to the solution above.
    assert_equal(routes[0].visits(), [1, 3])
    assert_equal(routes[1].visits(), [2, 4])

    # There's no excess load, so all excess load should be zero.
    assert_(not sol.has_excess_load())
    assert_equal(routes[0].excess_load(), 0)
    assert_equal(routes[1].excess_load(), 0)

    # Total route delivery demand (and pickups, which are all zero for this
    # instance).
    deliveries = [0] + [client.delivery for client in ok_small.clients()]
    assert_equal(routes[0].delivery(), deliveries[1] + deliveries[3])
    assert_equal(routes[1].delivery(), deliveries[2] + deliveries[4])

    assert_equal(routes[0].pickup(), 0)
    assert_equal(routes[1].pickup(), 0)

    # The first route is not feasible due to time warp, but the second one is.
    # See also the tests below.
    assert_(not routes[0].is_feasible())
    assert_(routes[1].is_feasible())

    # Total service duration.
    services = [0] + [client.service_duration for client in ok_small.clients()]
    assert_equal(routes[0].service_duration(), services[1] + services[3])
    assert_equal(routes[1].service_duration(), services[2] + services[4])


def test_route_time_warp_calculations(ok_small):
    """
    Tests route time warp calculations.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.routes()

    # There is time warp on the first route: duration(0, 1) = 1'544, so we
    # arrive at 1 before its opening window of 15'600. Service (360) thus
    # starts at 15'600, and completes at 15'600 + 360. Then we drive for
    # duration(1, 3) = 1'427, where we arrive after 15'300 (its closing time
    # window). This is where we incur time warp: we need to 'warp' to 15'300.
    assert_(sol.has_time_warp())
    assert_(routes[0].has_time_warp())
    assert_equal(routes[0].time_warp(), 15_600 + 360 + 1_427 - 15_300)

    # The second route has no time warp, so the overall solution time warp is
    # all incurred on the first route.
    assert_(not routes[1].has_time_warp())
    assert_equal(routes[1].time_warp(), 0)
    assert_equal(sol.time_warp(), routes[0].time_warp())


def test_route_wait_time_calculations():
    """
    Tests route wait time and slack calculations.
    """
    data = read("data/OkSmallWaitTime.txt")
    sol = Solution(data, [[1, 3], [2, 4]])
    routes = sol.routes()

    # In the second route, the time window of client 2 closes at 15'000. After
    # service and travel, we then arrive at client 4 before its time window is
    # open, so we have to wait. In particular, we have to wait:
    #   twEarly(4) - duration(2, 4) - serv(2) - twLate(2)
    #     = 18'000 - 1'090 - 360 - 15'000
    #     = 1'550.
    assert_equal(routes[1].wait_duration(), 1_550)

    # Since there is waiting time, there is no slack in the schedule. We should
    # thus start as late as possible, at:
    #   twLate(2) - duration(0, 2)
    #     = 15'000 - 1'944
    #     = 13'056.
    assert_equal(routes[1].slack(), 0)
    assert_equal(routes[1].start_time(), 13_056)

    # So far we have tested a route that had wait duration, but not time warp.
    # We now test a solution with a route that has both.
    sol = Solution(data, [[1, 2, 4], [3]])
    route, *_ = sol.routes()

    # This route has the same wait time as explained above. The time warp is
    # incurred earlier in the route, between 1 -> 2:
    #   twEarly(1) + serv(1) + duration(1, 2) - twLate(2)
    #     = 15'600 + 360 + 1'992 - 15'000
    #     = 2'952.
    assert_equal(route.time_warp(), 2_952)
    assert_equal(route.wait_duration(), 1_550)
    assert_equal(route.slack(), 0)

    # Finally, the overall route duration should be equal to the sum of the
    # travel, service, and waiting durations.
    assert_equal(
        route.duration(),
        route.travel_duration()
        + route.service_duration()
        + route.wait_duration(),
    )


def test_route_start_and_end_time_calculations(ok_small):
    """
    Tests route start time, slack, and end time calculations for cases with
    and without time warp or wait duration.
    """
    sol = Solution(ok_small, [[1, 3], [2, 4]])
    routes = sol.routes()

    # The first route has timewarp, so there is no slack in the schedule. We
    # should thus depart as soon as possible to arrive at the first client the
    # moment its time window opens.
    durations = ok_small.duration_matrix(profile=0)
    start_time = ok_small.location(1).tw_early - durations[0, 1]
    end_time = start_time + routes[0].duration() - routes[0].time_warp()

    assert_(routes[0].has_time_warp())
    assert_equal(routes[0].slack(), 0)
    assert_equal(routes[0].start_time(), start_time)
    assert_equal(routes[0].end_time(), end_time)

    # The second route has no time warp. The latest it can start is calculated
    # backwards from the closing of client 4's time window:
    #   twLate(4) - duration(2, 4) - serv(2) - duration(0, 2)
    #     = 19'500 - 1'090 - 360 - 1'944
    #     = 16'106.
    #
    # Because client 4 has a large time window, the earliest this route can
    # start is determined completely by client 2: we should not arrive before
    # its time window, because that would incur needless waiting. We should
    # thus not depart before:
    #   twEarly(2) - duration(0, 2)
    #     = 12'000 - 1'944
    #     = 10'056.
    assert_(not routes[1].has_time_warp())
    assert_equal(routes[1].wait_duration(), 0)
    assert_equal(routes[1].start_time(), 10_056)
    assert_equal(routes[1].slack(), 16_106 - 10_056)

    # The overall route duration is given by:
    #   duration(0, 2) + serv(2) + duration(2, 4) + serv(4) + duration(4, 0)
    #     = 1'944 + 360 + 1'090 + 360 + 1'475
    #     = 5'229.
    assert_equal(routes[1].duration(), 1_944 + 360 + 1_090 + 360 + 1_475)
    assert_equal(routes[1].end_time(), 10_056 + 5_229)


def test_route_release_time():
    """
    Tests that routes return the correct release times, and, when feasible,
    start after the release time.
    """
    data = read("data/OkSmallReleaseTimes.txt")
    sol = Solution(data, [[1, 3], [2, 4]])
    routes = sol.routes()

    # The client release times are 20'000, 5'000, 5'000 and 1'000. So the first
    # route has a release time of max(20'000, 5'000) = 20'000, and the second
    # has a release time of max(5'000, 1'000) = 5'000.
    assert_equal(routes[0].release_time(), 20_000)
    assert_equal(routes[1].release_time(), 5_000)

    # Second route is feasible, so should have start time not smaller than
    # release time.
    assert_(not routes[1].has_time_warp())
    assert_(routes[1].start_time() > routes[1].release_time())


@pytest.mark.parametrize(
    "dist_mat",
    [
        np.where(np.eye(3), 0, 100),
        np.where(np.eye(3), 0, 1),
        np.where(np.eye(3), 0, 1000),
    ],
)
def test_time_warp_for_a_very_constrained_problem(dist_mat):
    """
    This tests an artificial instance where the second client cannot be reached
    directly from the depot in a feasible solution, but only after the first
    client.
    """
    dur_mat = [
        [0, 1, 10],  # cannot get to 2 from depot within 2's time window
        [1, 0, 1],
        [1, 1, 0],
    ]

    data = ProblemData(
        clients=[
            Client(x=1, y=0, tw_late=5),
            Client(x=2, y=0, tw_late=5),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(2, tw_late=10)],
        distance_matrices=[dist_mat],
        duration_matrices=[dur_mat],
    )

    # This solution directly visits the second client from the depot, which is
    # not time window feasible.
    infeasible = Solution(data, [[1], [2]])
    assert_(infeasible.has_time_warp())
    assert_(not infeasible.has_excess_load())
    assert_(not infeasible.is_feasible())

    # But visiting the second client after the first is feasible.
    feasible = Solution(data, [[1, 2]])
    assert_(not feasible.has_time_warp())
    assert_(not feasible.has_excess_load())
    assert_(feasible.is_feasible())

    assert_equal(
        feasible.distance(),
        dist_mat[0, 1] + dist_mat[1, 2] + dist_mat[2, 0],
    )


def test_time_warp_return_to_depot():
    """
    This tests wether the calculated total duration and time warp includes the
    travel back to the depot.
    """
    data = ProblemData(
        clients=[Client(x=1, y=0)],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(tw_late=1)],
        distance_matrices=[np.asarray([[0, 0], [0, 0]])],
        duration_matrices=[np.asarray([[0, 1], [1, 0]])],
    )

    sol = Solution(data, [[1]])
    route, *_ = sol.routes()

    # Travel from depot to client and back gives duration 1 + 1 = 2. This is 1
    # more than the vehicle time window 1, giving a time warp of 1.
    assert_equal(route.duration(), 2)
    assert_equal(data.vehicle_type(0).tw_late, 1)
    assert_equal(sol.time_warp(), 1)


def tests_that_not_specifying_the_vehicle_type_assumes_a_default(ok_small):
    """
    Not specifying the vehicle type when providing a list of visits uses the
    first vehicle type to complete the routes. That could result in a solution
    using too many vehicles of the first type.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(2, capacity=10), VehicleType(capacity=20)]
    )

    sol = Solution(data, [[1, 2, 3, 4]])
    assert_equal(sol.num_routes(), 1)

    sol = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    assert_equal(sol.num_routes(), 1)

    sol = Solution(data, [[1, 2], [3, 4]])
    assert_equal(sol.num_routes(), 2)

    with assert_raises(RuntimeError):
        # This raises since we don't specify route types, which means we create
        # 3 routes of type 0 whereas we only have 2 available.
        sol = Solution(data, [[1], [2], [3, 4]])

    # It works if we specify the correct vehicle types.
    sol = Solution(
        data,
        [Route(data, [1], 0), Route(data, [2], 0), Route(data, [3, 4], 1)],
    )
    assert_equal(sol.num_routes(), 3)

    # But not if we violate the number of available vehicles of that type.
    with assert_raises(RuntimeError):
        sol = Solution(
            data,
            [Route(data, [1], 0), Route(data, [2], 1), Route(data, [3, 4], 1)],
        )


def test_copy(ok_small):
    """
    Tests that copied solutions are equal to the original solution, but not
    the exact same object.
    """
    sol = Solution(ok_small, [[1, 2, 3, 4]])
    copy_sol = copy(sol)
    deepcopy_sol = deepcopy(sol)

    # Copied solutions are equal to the original solution.
    assert_(sol == copy_sol)
    assert_(sol == deepcopy_sol)

    # But they are not the same object.
    assert_(sol is not copy_sol)
    assert_(sol is not deepcopy_sol)


def test_eq(ok_small):
    """
    Tests the solution's equality operator.
    """
    sol1 = Solution(ok_small, [[1, 2, 3, 4]])
    sol2 = Solution(ok_small, [[1, 2], [3], [4]])
    sol3 = Solution(ok_small, [[1, 2, 3, 4]])

    assert_(sol1 == sol1)  # Solutions should be equal to themselves
    assert_(sol2 == sol2)
    assert_(sol1 != sol2)  # different routes, so should not be equal
    assert_(sol1 == sol3)  # same routes, different solution

    sol4 = Solution(ok_small, [[1, 2, 3], [4]])
    sol5 = Solution(ok_small, [[4], [1, 2, 3]])

    assert_(sol4 == sol5)  # routes are the same, but in different order

    # And a few tests against things that are not solutions, just to be sure
    # there's also a type check in there somewhere.
    assert_(sol4 != 1)
    assert_(sol4 != "abc")
    assert_(sol5 != 5)
    assert_(sol5 != "cd")


def test_eq_with_multiple_vehicle_types(ok_small):
    """
    Tests that two solutions are not considered equal if they have the same
    routes (orders of clients) but served by different vehicle types.
    """
    # Make sure capacities are different but large enough (>18) to have no
    # violations so have the same attributes, such that we actually test if the
    # assignments are used for the equality comparison.
    data = ok_small.replace(
        vehicle_types=[VehicleType(2, capacity=20), VehicleType(capacity=30)]
    )

    # These two should be the same
    sol1 = Solution(data, [[1, 2, 3, 4]])
    sol2 = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    # Create solution with different vehicle type
    sol3 = Solution(data, [Route(data, [1, 2, 3, 4], 1)])

    # First two solution have one route with the same vehicle type
    assert_(sol1 == sol2)
    # Solution 3 is different since the route has a different vehicle type
    assert_(sol1 != sol3)

    # Order should not matter so these should be the same
    sol1 = Solution(data, [Route(data, [1, 2], 0), Route(data, [3, 4], 1)])
    sol2 = Solution(data, [Route(data, [3, 4], 1), Route(data, [1, 2], 0)])
    assert_(sol1 == sol2)

    # But changing the vehicle types should be different
    sol3 = Solution(data, [Route(data, [1, 2], 1), Route(data, [3, 4], 0)])
    assert_(sol1 != sol3)


def test_eq_unassigned():
    """
    Tests the equality operator for solutions with unassigned clients.
    """
    dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    data = ProblemData(
        clients=[
            Client(x=0, y=1, required=False),
            Client(x=1, y=0, required=False),
        ],
        depots=[Depot(x=0, y=0)],
        vehicle_types=[VehicleType(2, capacity=1)],
        distance_matrices=[dist],
        duration_matrices=[dist],
    )

    sol1 = Solution(data, [[1]])
    sol2 = Solution(data, [[1]])
    sol3 = Solution(data, [[2]])

    assert_(sol1 == sol2)
    assert_(sol1 != sol3)


def test_duplicate_vehicle_types(ok_small):
    """
    Tests that it is allowed to have duplicate vehicle types. These will be
    considered completely different during optimisation.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(capacity=10), VehicleType(capacity=10)]
    )

    sol1 = Solution(data, [Route(data, [1, 2, 3, 4], 0)])
    sol2 = Solution(data, [Route(data, [1, 2, 3, 4], 1)])

    assert_(sol1 != sol2)


@pytest.mark.parametrize(
    "vehicle_types",
    [
        [VehicleType(3, capacity=10)],
        [VehicleType(2, capacity=10), VehicleType(capacity=20)],
    ],
)
def test_str_contains_routes(ok_small, vehicle_types):
    """
    Tests that the Solution's string representation contains each route.
    """
    data = ok_small.replace(vehicle_types=vehicle_types)
    rng = RandomNumberGenerator(seed=2)

    for _ in range(5):  # let's do this a few times to really make sure
        sol = Solution.make_random(data, rng)
        str_representation = str(sol).splitlines()
        routes = sol.routes()

        # There should be no more than len(routes) lines (each detailing a
        # single route).
        assert_equal(len(str_representation), len(routes))

        # Each line should contain a route, where each route should contain
        # every client that is in the route as returned by routes().
        for route, str_route in zip(routes, str_representation):
            for client in route:
                assert_(str(client) in str_route)


def test_hash(ok_small):
    """
    Tests that solutions that compare the same have the same hash.
    """
    rng = RandomNumberGenerator(seed=2)

    sol1 = Solution.make_random(ok_small, rng)
    sol2 = Solution.make_random(ok_small, rng)

    # Two random solutions. They're not the same, so the hashes should not be
    # the same either (unless there's a collision, which is not the case here).
    assert_(sol1 != sol2)
    assert_(hash(sol1) != hash(sol2))

    sol3 = deepcopy(sol2)  # is a direct copy

    # These two are the same solution, so their hashes should be the same too.
    assert_equal(sol2, sol3)
    assert_equal(hash(sol2), hash(sol3))


def test_route_centroid(ok_small):
    """
    Tests that each route's center point is the center point of all clients
    visited by that route.
    """
    x = np.array([ok_small.location(idx).x for idx in range(5)])
    y = np.array([ok_small.location(idx).y for idx in range(5)])

    routes = [
        Route(ok_small, [1, 2], 0),
        Route(ok_small, [3], 0),
        Route(ok_small, [4], 0),
    ]

    for route in routes:
        x_center, y_center = route.centroid()
        assert_allclose(x_center, x[route].mean())
        assert_allclose(y_center, y[route].mean())


def test_solution_can_be_pickled(ok_small):
    """
    Tests that a solution can be serialised and unserialised.
    """
    rng = RandomNumberGenerator(seed=2)

    before_pickle = Solution.make_random(ok_small, rng)
    bytes = pickle.dumps(before_pickle)
    after_pickle = pickle.loads(bytes)

    assert_equal(after_pickle, before_pickle)


def test_route_can_be_pickled(rc208):
    """
    Tests that individual routes can be serialised and unserialised.
    """
    rng = RandomNumberGenerator(seed=2)
    sol = Solution.make_random(rc208, rng)

    for before_pickle in sol.routes():
        bytes = pickle.dumps(before_pickle)
        after_pickle = pickle.loads(bytes)

        assert_equal(after_pickle, before_pickle)


@pytest.mark.parametrize(
    ("assignment", "expected"), [((0, 0), 0), ((0, 1), 10), ((1, 1), 20)]
)
def test_fixed_vehicle_cost(
    ok_small, assignment: tuple[int, int], expected: int
):
    """
    Tests that the solution tracks the total fixed vehicle costs of the
    vehicles used for its routes.
    """
    # First vehicle type is free, second costs 10 per vehicle. The solution
    # should be able to track this.
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(2, capacity=10, fixed_cost=0),
            VehicleType(2, capacity=10, fixed_cost=10),
        ]
    )

    routes = [
        Route(data, [1, 2], assignment[0]),
        Route(data, [3, 4], assignment[1]),
    ]

    sol = Solution(data, routes)
    assert_equal(sol.fixed_vehicle_cost(), expected)


@pytest.mark.parametrize(
    ("tw_early", "tw_late", "expected"),
    [
        (0, 0, 20_277),  # cannot be back at the depot before 20'277
        (0, 20_000, 277),  # larger shift window decreases time warp
        (0, 20_277, 0),  # and in this case there is no more time warp
        (15_000, 20_000, 1_221),  # minimum route duration is 6'221
        (10_000, 20_000, 277),  # before earliest possible return
    ],
)
def test_route_shift_duration(
    ok_small, tw_early: int, tw_late: int, expected: int
):
    """
    Tests that Route computes time warp due to shift durations correctly on a
    simple, two-client route.
    """
    data = ok_small.replace(
        vehicle_types=[
            VehicleType(2, capacity=10, tw_early=tw_early, tw_late=tw_late)
        ]
    )

    # Overall route duration is, at the bare minimum, dist(0, 1) + dist(1, 2)
    # + dist(2, 0) + serv(1) + serv(2). That's 1'544 + 1'992 + 1'965 + 360
    # + 360 = 6'221. We cannot service client 1 before 15'600, and it takes
    # 1'544 to get there from the depot, so we leave at 14'056. Thus, the
    # earliest completion time is 14'056 + 6'221 = 20'277.
    route = Route(data, [1, 2], vehicle_type=0)
    assert_equal(route.time_warp(), expected)


@pytest.mark.parametrize(
    ("routes", "feasible"),
    [
        ([[1], [3, 4]], True),  # only one - OK
        ([[2], [3, 4]], True),  # only one - OK
        ([[1, 2], [3, 4]], False),  # both are in the solution - not OK
        ([[3, 4]], False),  # none - not OK
    ],
)
def test_solution_feasibility_with_mutually_exclusive_groups(
    ok_small, routes: list[list[int]], feasible: bool
):
    """
    Tests that the Solution class correctly accounts for feasibility regarding
    any mutually exclusive groups in the data.
    """
    # Clients 1 and 2 are part of a mutually exclusive group. Of these clients,
    # exactly one must be part of a feasible solution.
    clients = ok_small.clients()
    clients[0] = Client(1, 1, required=False, group=0)
    clients[1] = Client(2, 2, required=False, group=0)

    group = ClientGroup([1, 2], required=True)
    assert_(group.required)
    assert_(group.mutually_exclusive)

    data = ok_small.replace(clients=clients, groups=[group])
    sol = Solution(data, routes)
    assert_equal(sol.is_feasible(), feasible)
    assert_equal(sol.is_group_feasible(), feasible)


def test_optional_mutually_exclusive_group(ok_small):
    """
    Tests that mutually exclusive client groups can be skipped if they are
    not required. In that case at most one client from the group needs to be
    in the solution, but zero is also OK.
    """
    # Clients 1 and 2 are part of a mutually exclusive group. Of these clients,
    # at most one must be part of a feasible solution.
    clients = ok_small.clients()
    clients[0] = Client(1, 1, required=False, group=0)
    clients[1] = Client(2, 2, required=False, group=0)

    group = ClientGroup([1, 2], required=False)
    assert_(not group.required)
    assert_(group.mutually_exclusive)

    data = ok_small.replace(clients=clients, groups=[group])
    sol = Solution(data, [[3, 4]])
    assert_(sol.is_feasible())
    assert_(sol.is_group_feasible())


def test_distance_duration_cost_calculations(ok_small):
    """
    Tests route- and solution-level distance and duration cost calculations.
    """
    vehicle_types = [
        VehicleType(capacity=10, unit_distance_cost=5, unit_duration_cost=1),
        VehicleType(capacity=10, unit_distance_cost=1, unit_duration_cost=5),
    ]
    data = ok_small.replace(vehicle_types=vehicle_types)

    routes = [Route(data, [1, 2], 0), Route(data, [3, 4], 1)]
    assert_equal(routes[0].distance_cost(), 5 * routes[0].distance())
    assert_equal(routes[0].duration_cost(), 1 * routes[0].duration())
    assert_equal(routes[1].distance_cost(), 1 * routes[1].distance())
    assert_equal(routes[1].duration_cost(), 5 * routes[1].duration())

    sol = Solution(data, routes)
    assert_equal(sol.distance(), sum(r.distance() for r in routes))
    assert_equal(sol.distance_cost(), sum(r.distance_cost() for r in routes))
    assert_equal(sol.duration(), sum(r.duration() for r in routes))
    assert_equal(sol.duration_cost(), sum(r.duration_cost() for r in routes))


def test_start_end_depot_not_same_on_empty_route(ok_small_multi_depot):
    """
    Tests that empty routes correctly evaluate distance and duration travelled
    between depots, even though there are no actual clients on the route.
    """
    vehicle_type = VehicleType(3, 10, start_depot=0, end_depot=1)
    data = ok_small_multi_depot.replace(vehicle_types=[vehicle_type])

    route = Route(data, [], vehicle_type=0)

    assert_equal(route.start_depot(), 0)
    assert_equal(route.end_depot(), 1)

    dist_mat = data.distance_matrix(0)
    assert_equal(route.distance(), dist_mat[0, 1])

    dur_mat = data.duration_matrix(0)
    assert_equal(route.duration(), dur_mat[0, 1])
