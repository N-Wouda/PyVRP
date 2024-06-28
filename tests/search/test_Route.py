import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal

from pyvrp import ProblemData, VehicleType
from pyvrp.search._search import Node, Route


@pytest.mark.parametrize("loc", [0, 1, 10])
def test_node_init(loc: int):
    """
    Tests that after initialisation, a Node has index 0 and is not in a route.
    """
    node = Node(loc=loc)
    assert_equal(node.client, loc)
    assert_equal(node.idx, 0)
    assert_(node.route is None)


@pytest.mark.parametrize(("idx", "vehicle_type"), [(0, 0), (1, 0), (1, 1)])
def test_route_init(ok_small, idx: int, vehicle_type: int):
    """
    Tests that after initialisation, a Route has the given index and vehicle
    type.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(1, capacity=1), VehicleType(2, capacity=2)]
    )

    route = Route(data, idx=idx, vehicle_type=vehicle_type)
    assert_equal(route.idx, idx)
    assert_equal(route.vehicle_type, vehicle_type)


@pytest.mark.parametrize("loc", [0, 1, 10])
def test_new_nodes_are_not_depots(loc: int):
    """
    Tests that new nodes cannot be depots: to be a depot in this context
    requires route membership, and new nodes are not in routes yet.
    """
    node = Node(loc=loc)
    assert_(not node.is_depot())


def test_insert_and_remove_update_node_idx_and_route_properties(ok_small):
    """
    Tests that after a node is inserted into a route, its index and route
    properties are updated to reflect its new position in the route. After
    the node is removed from the route, these properties revert to their
    defaults.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    # After construction, the node is not in a route yet.
    node = Node(loc=1)
    assert_equal(node.idx, 0)
    assert_(node.route is None)

    # Add to the route, test the route and idx properties are updated.
    route.append(node)
    assert_equal(node.idx, 1)
    assert_(node.route is route)

    # Remove and test the node reverts to its initial state.
    del route[1]
    assert_equal(node.idx, 0)
    assert_(node.route is None)


def test_route_depots_are_depots(ok_small):
    """
    Tests that the start and end depot nodes in a route known they are, in
    fact, depots.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    assert_equal(route.start_depot(), ok_small.vehicle_type(0).start_depot)
    assert_equal(route.end_depot(), ok_small.vehicle_type(0).end_depot)

    for loc in range(1, 3):
        # The depots flank the clients at indices {1, ..., len(route)}. Thus,
        # depots are at indices 0 and len(route) + 1.
        route.append(Node(loc=loc))
        assert_(route[0].is_depot())
        assert_(route[len(route) + 1].is_depot())


def test_route_append_increases_route_len(ok_small):
    """
    Tests that appending nodes to a route increases the route's length.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    assert_equal(len(route), 0)

    node = Node(loc=1)
    route.append(node)
    assert_equal(len(route), 1)
    assert_(route[1] is node)  # pointers, so must be same object

    node = Node(loc=2)
    route.append(node)
    assert_equal(len(route), 2)
    assert_(route[2] is node)  # pointers, so must be same object


def test_route_insert(ok_small):
    """
    Tests that inserting and appending nodes works as expected: appending adds
    to the end, inserting places at the given index.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    assert_equal(len(route), 0)

    # Insert a few nodes so we have an actual route.
    route.append(Node(loc=1))
    route.append(Node(loc=2))
    assert_equal(len(route), 2)
    assert_equal(route[1].client, 1)
    assert_equal(route[2].client, 2)

    # # Now insert a new nodes at index 1.
    route.insert(1, Node(loc=3))
    assert_equal(len(route), 3)
    assert_equal(route[1].client, 3)
    assert_equal(route[2].client, 1)
    assert_equal(route[3].client, 2)


def test_route_iter_returns_all_clients(ok_small):
    """
    Tests that iterating over a route returns all clients in the route, but
    not the depots.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    for loc in [1, 2, 3]:
        route.append(Node(loc=loc))

    nodes = [node for node in route]
    assert_equal(len(nodes), len(route))

    # Iterating the Route object returns all clients, not the depots at index
    # ``0`` and index ``len(route) + 1`` in the Route object.
    assert_equal(nodes[0], route[1])
    assert_equal(nodes[1], route[2])
    assert_equal(nodes[2], route[3])


def test_route_add_and_delete_client_leaves_route_empty(ok_small):
    """
    Tests that adding and removing a client leaves a route empty.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    route.append(Node(loc=1))
    assert_equal(len(route), 1)

    del route[1]
    assert_equal(len(route), 0)


def test_route_delete_reduces_size_by_one(ok_small):
    """
    Deleting an item at an index removes only the indicated index, not more.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    route.append(Node(loc=1))
    route.append(Node(loc=2))
    assert_equal(len(route), 2)

    del route[1]
    assert_equal(len(route), 1)
    assert_equal(route[1].client, 2)


@pytest.mark.parametrize("num_nodes", range(4))
def test_route_clear_empties_entire_route(ok_small, num_nodes: int):
    """
    The clear() method should clear the entire route, not just remove part of
    it.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    for loc in range(1, num_nodes + 1):
        route.append(Node(loc=loc))

    assert_equal(len(route), num_nodes)

    route.clear()
    assert_equal(len(route), 0)


def test_excess_load(ok_small):
    """
    Tests that the route calculations excess load correctly.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for loc in [1, 2, 3, 4]:
        route.append(Node(loc=loc))
    route.update()

    # The instance has four clients, which have a total delivery demand of 18.
    # The only vehicle type in the instance has a capacity of 10, so this route
    # has excess load.
    assert_(route.has_excess_load())
    assert_equal(route.excess_load(), 8)
    assert_equal(route.load(), 18)
    assert_equal(route.capacity(), 10)


@pytest.mark.parametrize("fixed_cost", [0, 9])
def test_fixed_vehicle_cost(ok_small, fixed_cost: int):
    """
    Tests that the fixed vehicle cost method returns the assigned vehicle
    type's fixed cost value.
    """
    data = ok_small.replace(
        vehicle_types=[VehicleType(2, capacity=10, fixed_cost=fixed_cost)]
    )
    route = Route(data, idx=0, vehicle_type=0)
    assert_equal(route.fixed_vehicle_cost(), fixed_cost)


@pytest.mark.parametrize("client", [1, 2, 3, 4])
def test_dist_and_load_for_single_client_routes(ok_small, client: int):
    """
    Tests that the route calculates distance and load correctly for a
    single-client route.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    route.append(Node(loc=client))
    route.update()

    # Only the client has any delivery demand, so the total route load should
    # be equal to it.
    assert_equal(route.load(), ok_small.location(client).delivery)
    assert_equal(
        route.load_between(0, 2).load(), ok_small.location(client).delivery
    )

    # The load_between() function is inclusive.
    assert_equal(route.load_between(0, 0).load(), 0)
    assert_equal(
        route.load_between(1, 1).load(), ok_small.location(client).delivery
    )

    # Distances on various segments of the route.
    distances = ok_small.distance_matrix(profile=0)
    assert_equal(route.dist_between(0, 1).distance(), distances[0, client])
    assert_equal(route.dist_between(1, 2).distance(), distances[client, 0])
    assert_equal(
        route.dist_between(0, 2).distance(),
        distances[0, client] + distances[client, 0],
    )

    # This should always be zero because distance is a property of the edges,
    # not the nodes.
    assert_equal(route.dist_at(0).distance(), 0)
    assert_equal(route.dist_at(1).distance(), 0)


def test_route_overlaps_with_self_no_matter_the_tolerance_value(ok_small):
    """
    Tests that a route always overlaps with itself.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    route.append(Node(loc=1))
    route.append(Node(loc=2))

    route.update()

    assert_(route.overlaps_with(route, 0))
    assert_(route.overlaps_with(route, 0.5))
    assert_(route.overlaps_with(route, 1))


def test_all_routes_overlap_with_maximum_tolerance_value(ok_small):
    """
    Tests that any route overlaps with any other route with the maximum
    tolerance value.
    """
    route1 = Route(ok_small, idx=0, vehicle_type=0)
    for loc in [1, 2]:
        route1.append(Node(loc=loc))
    route1.update()

    route2 = Route(ok_small, idx=0, vehicle_type=0)
    for loc in [3, 4]:
        route2.append(Node(loc=loc))
    route2.update()

    # The routes are clearly not the same, and don't overlap with zero
    # tolerance.
    assert_(not route1.overlaps_with(route2, 0))
    assert_(not route2.overlaps_with(route1, 0))

    # But with maximum tolerance, they do.
    assert_(route1.overlaps_with(route2, 1))
    assert_(route2.overlaps_with(route1, 1))


@pytest.mark.parametrize("locs", [(1, 2, 3), (3, 4), (1,)])
def test_str_contains_route(ok_small, locs: list[int]):
    """
    Test that each client in the route is also printed in the route's __str__.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)

    for loc in locs:
        route.append(Node(loc=loc))

    for loc in locs:
        assert_(str(loc) in str(route))


def test_route_duration_access(ok_small):
    """
    Tests access to a client's or depot's duration segment, as tracked by
    the route.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for client in range(ok_small.num_depots, ok_small.num_locations):
        route.append(Node(loc=client))
    route.update()

    for idx in range(len(route) + 2):
        is_depot = idx % (len(route) + 1) == 0
        loc = ok_small.location(idx % (len(route) + 1))
        ds = route.duration_at(idx)

        assert_equal(ds.time_warp(), 0)

        if is_depot:
            vehicle_type = ok_small.vehicle_type(route.vehicle_type)
            assert_equal(ds.tw_early(), vehicle_type.tw_early)
            assert_equal(ds.tw_late(), vehicle_type.tw_late)
            assert_equal(ds.duration(), 0)
        else:
            assert_equal(ds.tw_early(), loc.tw_early)
            assert_equal(ds.tw_late(), loc.tw_late)
            assert_equal(ds.duration(), loc.service_duration)


@pytest.mark.parametrize("loc", [1, 2, 3, 4])
def test_duration_between_client_returns_node_duration(ok_small, loc: int):
    """
    Tests that calling the ``duration_between()`` with the same start and end
    arguments returns a node's duration segment data.
    """
    client = ok_small.location(loc)

    route = Route(ok_small, idx=0, vehicle_type=0)
    route.append(Node(loc=loc))
    route.update()

    # Duration of the depot node DS's is zero, and for the client it is equal
    # to the service duration.
    assert_equal(route.duration_between(0, 0).duration(), 0)
    assert_equal(
        route.duration_between(1, 1).duration(), client.service_duration
    )
    assert_equal(route.duration_between(2, 2).duration(), 0)

    # Single route solutions are all feasible for this instance.
    assert_equal(route.time_warp(), 0)


def test_duration_between_equal_to_before_after_when_one_is_depot(ok_small):
    """
    Tests that ``duration_between()`` returns the same value as
    ``duration_before()`` or ``duration_after()`` when one side is the depot.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for client in range(ok_small.num_depots, ok_small.num_locations):
        route.append(Node(loc=client))

    route.update()

    for idx in [1, 2, 3, 4]:
        before = route.duration_before(idx)
        between_before = route.duration_between(0, idx)
        assert_equal(before.duration(), between_before.duration())
        assert_equal(before.time_warp(), between_before.time_warp())

        after = route.duration_after(idx)
        between_after = route.duration_between(idx, len(route) + 1)
        assert_equal(after.duration(), between_after.duration())
        assert_equal(after.time_warp(), between_after.time_warp())


def test_duration_between_single_route_has_correct_time_warp(ok_small):
    """
    Tests duration segment access on a single-route solution where we know
    exactly where in the route time warp occurs.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for client in range(ok_small.num_depots, ok_small.num_locations):
        route.append(Node(loc=client))

    assert_equal(len(route), ok_small.num_clients)

    route.update()
    assert_(route.has_time_warp())
    assert_equal(route.duration_between(0, 5).time_warp(), route.time_warp())

    # Client #1 (at idx 1) causes the time warp in combination with client #3:
    # #1 can only be visited after #3's window has already closed.
    assert_equal(route.time_warp(), 3_633)
    assert_equal(route.duration_between(1, 4).time_warp(), 3_633)
    assert_equal(route.duration_between(0, 4).time_warp(), 3_633)
    assert_equal(route.duration_between(1, 5).time_warp(), 3_633)
    assert_equal(route.duration_between(1, 3).time_warp(), 3_633)

    # But excluding client #1, other subtours are (time-)feasible:
    for start, end in [(2, 4), (3, 5), (2, 3), (4, 5), (5, 5), (0, 1), (0, 2)]:
        assert_equal(route.duration_between(start, end).time_warp(), 0)


def test_distance_is_equal_to_dist_between_over_whole_route(ok_small):
    """
    Tests that calling distance() on the route object is the same as calling
    dist_between() with the start and end depot indices as arguments.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for client in range(ok_small.num_depots, ok_small.num_locations):
        route.append(Node(loc=client))
    route.update()

    assert_equal(
        route.distance(), route.dist_between(0, len(route) + 1).distance()
    )


@pytest.mark.parametrize("clients", [(1, 2, 3, 4), (1, 2), (3, 4)])
def test_route_centroid(ok_small, clients):
    """
    Tests that Route computes the center point of client locations correctly.
    """
    route = Route(ok_small, 0, 0)
    for client in clients:
        route.append(Node(loc=client))

    route.update()

    x = [ok_small.location(client).x for client in clients]
    y = [ok_small.location(client).y for client in clients]
    assert_allclose(route.centroid(), (np.mean(x), np.mean(y)))


@pytest.mark.parametrize(
    ("max_duration", "expected"),
    [
        (100_000, 3_633),  # large enough; same time warp as other tests
        (5_000, 3_633),  # no effect of max_duration due to existing time warp
        (4_000, 3_950),  # now max_duration constraint applies
        (3_000, 4_950),  # the max_duration constraint scales linearly
        (0, 7_950),  # max_duration = 0, so time warp equals route duration
    ],
)
def test_max_duration(ok_small: ProblemData, max_duration: int, expected: int):
    """
    Tests that the maximum duration attribute of vehicle types is reflected
    in the route's time warp calculations.
    """
    vehicle_type = VehicleType(3, capacity=10, max_duration=max_duration)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    route = Route(data, 0, 0)
    for client in range(data.num_depots, data.num_locations):
        route.append(Node(loc=client))

    route.update()
    assert_equal(route.duration(), 7_950)
    assert_(route.has_time_warp())
    assert_equal(route.time_warp(), expected)


@pytest.mark.parametrize(
    ("max_distance", "expected"),
    [
        (100_000, 0),  # large enough; there is now no excess distance
        (5_000, 1_450),  # now max_distance constraint applies
        (0, 6_450),  # max_distance scales linearly: this is the full distance
    ],
)
def test_max_distance(ok_small: ProblemData, max_distance: int, expected: int):
    """
    Tests that the maximum distance attribute of vehicle types is reflected
    in the route's excess distance calculations.
    """
    vehicle_type = VehicleType(3, capacity=10, max_distance=max_distance)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    route = Route(data, 0, 0)
    for client in range(data.num_depots, data.num_locations):
        route.append(Node(loc=client))

    route.update()
    assert_equal(route.distance(), 6_450)
    assert_equal(route.has_excess_distance(), expected > 0)
    assert_equal(route.excess_distance(), expected)


@pytest.mark.parametrize(
    ("visits", "load_feas", "time_feas", "dist_feas"),
    [
        ([1, 2, 3], False, False, False),
        ([1, 3], True, False, True),
        ([1, 2], True, True, True),
    ],
)
def test_is_feasible(
    ok_small,
    visits: list[int],
    load_feas: bool,
    time_feas: bool,
    dist_feas: bool,
):
    """
    Tests that various constraint violations are taken into account when
    determining overall route feasibility.
    """
    vehicle_type = VehicleType(3, capacity=10, max_distance=6_000)
    data = ok_small.replace(vehicle_types=[vehicle_type])

    route = Route(data, 0, 0)
    for client in visits:
        route.append(Node(loc=client))
    route.update()

    assert_equal(route.is_feasible(), load_feas and time_feas and dist_feas)
    assert_equal(not route.has_excess_distance(), dist_feas)
    assert_equal(not route.has_excess_load(), load_feas)
    assert_equal(not route.has_time_warp(), time_feas)


def test_dist_between_equal_to_before_after_when_one_is_depot(ok_small):
    """
    Tests that ``dist_between()`` returns the same value as
    ``dist_before()`` or ``distance_after()`` when one side is the depot.
    """
    route = Route(ok_small, idx=0, vehicle_type=0)
    for client in range(ok_small.num_depots, ok_small.num_locations):
        route.append(Node(loc=client))

    route.update()

    for idx in [1, 2, 3, 4]:
        before = route.dist_before(idx)
        between_before = route.dist_between(0, idx)
        assert_equal(before.distance(), between_before.distance())

        after = route.dist_after(idx)
        between_after = route.dist_between(idx, len(route) + 1)
        assert_equal(after.distance(), between_after.distance())


def test_load_between_equal_to_before_after_when_one_is_depot(small_spd):
    """
    Tests that ``load_between()`` returns the same value as
    ``load_before()`` or ``load_after()`` when one side is the depot.
    """
    route = Route(small_spd, idx=0, vehicle_type=0)
    for client in range(small_spd.num_depots, small_spd.num_locations):
        route.append(Node(loc=client))

    route.update()

    for idx in [1, 2, 3, 4]:
        before = route.load_before(idx)
        between_before = route.load_between(0, idx)
        assert_equal(before.load(), between_before.load())
        assert_equal(before.pickup(), between_before.pickup())
        assert_equal(before.delivery(), between_before.delivery())

        after = route.load_after(idx)
        between_after = route.load_between(idx, len(route) + 1)
        assert_equal(after.load(), between_after.load())
        assert_equal(after.pickup(), between_after.pickup())
        assert_equal(after.delivery(), between_after.delivery())


def test_distance_different_profiles(ok_small_two_profiles):
    """
    Tests that accessing the distance concatenation scheme for different route
    segments takes into account the profile argument.
    """
    data = ok_small_two_profiles
    route = Route(data, idx=0, vehicle_type=0)
    for client in range(data.num_depots, data.num_locations):
        route.append(Node(loc=client))
    route.update()

    assert_equal(route.distance(), 6_450)
    assert_equal(route.profile(), 0)

    # Let's test with a different profile. The distance on the route should be
    # double using the second profile.
    depot_to_depot = route.dist_between(0, len(route) + 1, profile=1)
    assert_equal(depot_to_depot.distance(), 2 * route.distance())

    after_start = route.dist_after(0, profile=1)
    assert_equal(after_start.distance(), depot_to_depot.distance())

    before_end = route.dist_before(len(route) + 1, profile=1)
    assert_equal(before_end.distance(), depot_to_depot.distance())


def test_duration_different_profiles(ok_small_two_profiles):
    """
    Tests that accessing the duration concatenation scheme for different route
    segments takes into account the profile argument.
    """
    data = ok_small_two_profiles
    route = Route(data, idx=0, vehicle_type=0)
    for client in range(data.num_depots, data.num_locations):
        route.append(Node(loc=client))
    route.update()

    assert_equal(route.duration(), 7_950)
    assert_equal(route.profile(), 0)

    # Let's test with a different profile. The travel duration on the route
    # doubles using the second profile, but that does not mean the actual
    # *duration* doubles, since e.g. service duration remains the same. There
    # is no wait time, so the new duration is twice the original duration,
    # adjusted for the service duration.
    depot_to_depot = route.duration_between(0, len(route) + 1, profile=1)
    service = sum(c.service_duration for c in data.clients())
    assert_equal(depot_to_depot.duration(), 2 * route.duration() - service)

    after_start = route.duration_after(0, profile=1)
    assert_equal(after_start.duration(), depot_to_depot.duration())

    before_end = route.duration_before(len(route) + 1, profile=1)
    assert_equal(before_end.duration(), depot_to_depot.duration())


def test_start_end_depot_not_same_on_empty_route(ok_small_multi_depot):
    """
    Tests that empty routes correctly evaluate distance and duration travelled
    between depots, even though there are no actual clients on the route.
    """
    vehicle_type = VehicleType(3, 10, start_depot=0, end_depot=1)
    data = ok_small_multi_depot.replace(vehicle_types=[vehicle_type])

    route = Route(data, idx=0, vehicle_type=0)
    route.update()

    assert_equal(route.start_depot(), 0)
    assert_equal(route.end_depot(), 1)

    dist_mat = data.distance_matrix(0)
    assert_equal(route.distance(), dist_mat[0, 1])

    dur_mat = data.duration_matrix(0)
    assert_equal(route.duration(), dur_mat[0, 1])
