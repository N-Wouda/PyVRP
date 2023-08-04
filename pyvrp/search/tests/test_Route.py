import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import VehicleType
from pyvrp.search._search import Node, Route
from pyvrp.tests.helpers import make_heterogeneous, read


@pytest.mark.parametrize("loc", [0, 1, 10])
def test_node_init(loc: int):
    node = Node(loc=loc)
    assert_equal(node.client, loc)
    assert_equal(node.idx, 0)
    assert_(node.route is None)


@pytest.mark.parametrize(("idx", "vehicle_type"), [(0, 0), (1, 0), (1, 1)])
def test_route_init(idx: int, vehicle_type: int):
    data = read("data/OkSmall.txt")
    data = make_heterogeneous(data, [VehicleType(1, 1), VehicleType(2, 2)])

    route = Route(data, idx=idx, vehicle_type=vehicle_type)
    assert_equal(route.idx, idx)
    assert_equal(route.vehicle_type, vehicle_type)


@pytest.mark.parametrize("loc", [0, 1, 10])
def test_new_nodes_are_not_depots(loc: int):
    node = Node(loc=loc)
    assert_(not node.is_depot())


def test_route_depots_are_depots():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)

    for loc in range(2):
        # The depots flank the clients at indices {1, ..., len(route)}. Thus,
        # depots are at indices 0 and len(route) + 1.
        route.append(Node(loc=loc))
        assert_(route[0].is_depot())
        assert_(route[len(route) + 1].is_depot())


def test_route_append_increases_route_len():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)
    assert_equal(len(route), 0)

    node = Node(loc=1)
    route.append(node)
    assert_equal(len(route), 1)
    assert_(route[1] is node)  # pointers, so must be same object

    node = Node(loc=2)
    route.append(node)
    assert_equal(len(route), 2)
    assert_(route[2] is node)  # pointers, so must be same object


def test_route_insert():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)
    assert_equal(len(route), 0)

    # Insert a few customers so we have an actual route.
    route.append(Node(loc=1))
    route.append(Node(loc=2))
    assert_equal(len(route), 2)
    assert_equal(route[1].client, 1)
    assert_equal(route[2].client, 2)

    # # Now insert a new customer at index 1.
    route.insert(1, Node(loc=3))
    assert_equal(len(route), 3)
    assert_equal(route[1].client, 3)
    assert_equal(route[2].client, 1)
    assert_equal(route[3].client, 2)


def test_route_iter():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)

    for loc in [1, 2, 3]:
        route.append(Node(loc=loc))

    nodes = [node for node in route]
    assert_equal(len(nodes), len(route))

    # Iterating the Route object returns all clients, not the depots at index
    # ``0`` and index ``len(route) + 1`` in the Route object.
    assert_equal(nodes[0], route[1])
    assert_equal(nodes[1], route[2])
    assert_equal(nodes[2], route[3])


def test_route_add_and_remove_client_leaves_route_empty():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)

    route.append(Node(loc=1))
    assert_equal(len(route), 1)

    route.remove(1)
    assert_equal(len(route), 0)


def test_route_remove_reduces_size_by_one():
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)

    route.append(Node(loc=1))
    route.append(Node(loc=2))
    assert_equal(len(route), 2)

    route.remove(1)
    assert_equal(len(route), 1)


@pytest.mark.parametrize("num_nodes", range(4))
def test_route_clear(num_nodes: int):
    data = read("data/OkSmall.txt")
    route = Route(data, idx=0, vehicle_type=0)

    for loc in range(num_nodes):
        route.append(Node(loc=loc))
        assert_equal(len(route), loc + 1)

    route.clear()
    assert_equal(len(route), 0)


def test_load_feasibility():
    pass


def test_time_warp_feasibility():
    pass


def test_feasible_flags():
    pass


def test_route_overlap():
    pass


def test_tw_between():
    pass


def test_load_between():
    pass


def test_dist_between():
    pass
