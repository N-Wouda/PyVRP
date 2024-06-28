import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_equal,
    assert_raises,
    assert_warns,
)

from pyvrp import Client, ClientGroup, Depot, Model, Profile, VehicleType
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning
from pyvrp.stop import MaxIterations


def test_model_data():
    """
    Tests that calling the ``data()`` member on the model returns a data
    instance representing the added data.
    """
    model = Model()

    # Let's add some data: a single client, and edges from/to the depot.
    depot = model.add_depot(0, 0)
    client = model.add_client(0, 1, delivery=1)
    model.add_edge(depot, client, 1, 1)
    model.add_edge(client, depot, 1, 1)
    model.add_vehicle_type(capacity=1, num_available=1)

    data = model.data()
    assert_equal(data.num_clients, 1)
    assert_equal(data.num_vehicle_types, 1)
    assert_equal(data.num_vehicles, 1)


def test_add_edge_raises_negative_distance_or_duration():
    """
    Negative distances or durations are not understood. Attempting to add
    such edges should raise an error.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    client = model.add_client(0, 1)

    model.add_edge(depot, client, distance=0, duration=0)  # zero should be OK

    with assert_raises(ValueError):  # negative distance should not be OK
        model.add_edge(client, depot, distance=-1, duration=0)

    with assert_raises(ValueError):  # negative duration should also not be OK
        model.add_edge(client, depot, distance=0, duration=-1)


def test_add_edge_raises_self_connection_nonzero_distance_or_duration():
    """
    Edges connecting a node to itself must have 0 distance and duration.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    client = model.add_client(0, 1)

    model.add_edge(depot, client, distance=1, duration=1)  # should be OK
    model.add_edge(depot, depot, distance=0, duration=0)  # should be OK
    model.add_edge(client, client, distance=0, duration=0)  # should be OK

    with assert_raises(ValueError):  # self loop with nonzero distance not OK
        model.add_edge(depot, depot, distance=1, duration=0)

    with assert_raises(ValueError):  # self loop with nonzero duration not OK
        model.add_edge(depot, depot, distance=0, duration=1)

    with assert_raises(ValueError):  # self loop with nonzero distance not OK
        model.add_edge(client, client, distance=1, duration=0)

    with assert_raises(ValueError):  # self loop with nonzero duration not OK
        model.add_edge(client, client, distance=0, duration=1)


def test_add_client_attributes():
    """
    Smoke test that checks, for a single client, that the model adds a client
    whose attributes are the same as what was passed in.
    """
    model = Model()
    client = model.add_client(
        x=1,
        y=2,
        delivery=3,
        pickup=9,
        service_duration=4,
        tw_early=5,
        tw_late=6,
        release_time=0,
        prize=8,
        required=False,
    )

    assert_equal(client.x, 1)
    assert_equal(client.y, 2)
    assert_equal(client.delivery, 3)
    assert_equal(client.pickup, 9)
    assert_equal(client.service_duration, 4)
    assert_equal(client.tw_early, 5)
    assert_equal(client.tw_late, 6)
    assert_equal(client.release_time, 0)
    assert_equal(client.prize, 8)
    assert_(not client.required)


def test_add_depot_attributes():
    """
    Smoke test that checks the depot attributes are the same as what was passed
    in.
    """
    model = Model()
    depot = model.add_depot(x=1, y=0)
    assert_equal(depot.x, 1)
    assert_equal(depot.y, 0)


def test_add_edge():
    """
    Smoke test that checks edge attributes are the same as what was passed in.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    client = model.add_client(0, 1)
    edge = model.add_edge(depot, client, distance=15, duration=49)

    assert_(edge.frm is depot)
    assert_(edge.to is client)
    assert_equal(edge.distance, 15)
    assert_equal(edge.duration, 49)


def test_add_vehicle_type():
    """
    Smoke test that checks vehicle type attributes are the same as what was
    passed in.
    """
    model = Model()
    vehicle_type = model.add_vehicle_type(
        num_available=10,
        capacity=998,
        fixed_cost=1_001,
        tw_early=17,
        tw_late=19,
        max_duration=93,
        max_distance=97,
    )

    assert_equal(vehicle_type.num_available, 10)
    assert_equal(vehicle_type.capacity, 998)
    assert_equal(vehicle_type.fixed_cost, 1_001)
    assert_equal(vehicle_type.tw_early, 17)
    assert_equal(vehicle_type.tw_late, 19)
    assert_equal(vehicle_type.max_duration, 93)
    assert_equal(vehicle_type.max_distance, 97)


def test_add_vehicle_type_default_depots():
    """
    Tests that ``Model.add_vehicle_type`` correctly sets the (default) depot
    attributes on the vehicle type.
    """
    m = Model()
    depot1 = m.add_depot(x=0, y=0)
    depot2 = m.add_depot(x=1, y=1)

    # No depot specified: should default to the first (location index 0).
    vehicle_type1 = m.add_vehicle_type()
    assert_equal(vehicle_type1.start_depot, 0)
    assert_equal(vehicle_type1.end_depot, 0)

    # First depot specified, should set first (location index 0).
    vehicle_type2 = m.add_vehicle_type(start_depot=depot1, end_depot=depot1)
    assert_equal(vehicle_type2.start_depot, 0)
    assert_equal(vehicle_type2.end_depot, 0)

    # Second depot specified, should set second (location index 1).
    vehicle_type3 = m.add_vehicle_type(start_depot=depot2, end_depot=depot2)
    assert_equal(vehicle_type3.start_depot, 1)
    assert_equal(vehicle_type3.start_depot, 1)

    # A mix is also OK.
    vehicle_type3 = m.add_vehicle_type(start_depot=depot1, end_depot=depot2)
    assert_equal(vehicle_type3.start_depot, 0)
    assert_equal(vehicle_type3.end_depot, 1)


def test_add_vehicle_type_raises_for_unknown_depot():
    """
    Tests that adding a vehicle type with a depot that's not known to the model
    raises a ValueError.
    """
    m = Model()
    depot = Depot(x=0, y=0)

    with assert_raises(ValueError):
        m.add_vehicle_type(start_depot=depot)

    with assert_raises(ValueError):
        m.add_vehicle_type(end_depot=depot)


def test_get_locations():
    """
    Checks that the ``locations`` property returns the depot and all clients.
    """
    model = Model()
    client1 = model.add_client(0, 1)
    depot = model.add_depot(0, 0)
    client2 = model.add_client(0, 2)

    # Test that depot is always first and that we can get the clients by index.
    assert_equal(model.locations[0], depot)
    assert_equal(model.locations[1], client1)
    assert_equal(model.locations[2], client2)


def test_get_vehicle_types():
    """
    Tests the ``vehicle_types`` property.
    """
    model = Model()
    vehicle_type1 = model.add_vehicle_type(1, capacity=2)
    vehicle_type2 = model.add_vehicle_type(1, capacity=3)

    # Test that we can get the vehicle types by index, or as a list.
    assert_equal(model.vehicle_types[0], vehicle_type1)
    assert_equal(model.vehicle_types[1], vehicle_type2)
    assert_equal(model.vehicle_types, [vehicle_type1, vehicle_type2])


def test_from_data(small_cvrp):
    """
    Tests that initialising the model from a data instance results in a valid
    model representation of that data instance.
    """
    m = Model.from_data(small_cvrp)
    m_data = m.data()

    # We can first check if the overall problem dimension numbers agree.
    assert_equal(m_data.num_clients, small_cvrp.num_clients)
    assert_equal(m_data.num_vehicles, small_cvrp.num_vehicles)
    assert_equal(
        m_data.vehicle_type(0).capacity,
        small_cvrp.vehicle_type(0).capacity,
    )

    assert_equal(m_data.distance_matrices(), small_cvrp.distance_matrices())
    assert_equal(m_data.duration_matrices(), small_cvrp.duration_matrices())


def test_from_data_and_solve(small_cvrp, ok_small):
    """
    Tests that solving a model initialised from a data instance finds the
    correct (known) solutions.
    """
    model = Model.from_data(small_cvrp)
    res = model.solve(stop=MaxIterations(100), seed=0)
    assert_equal(res.cost(), 3_743)
    assert_(res.is_feasible())

    model = Model.from_data(ok_small)
    res = model.solve(stop=MaxIterations(100), seed=0)
    assert_equal(res.cost(), 9_155)
    assert_(res.is_feasible())


def test_model_and_solve(ok_small):
    """
    Tests that solving a model initialised using the modelling interface
    finds the correct (known) solutions.
    """
    model = Model.from_data(ok_small)
    res = model.solve(stop=MaxIterations(100), seed=0)
    assert_equal(res.cost(), 9_155)
    assert_(res.is_feasible())

    # Now do the same thing, but model the instance using the modelling API.
    # This should of course result in the same solution.
    model = Model()
    model.add_vehicle_type(
        num_available=3,
        capacity=10,
        tw_early=0,
        tw_late=45000,
    )

    depot = model.add_depot(x=2334, y=726)
    clients = [
        model.add_client(226, 1297, 5, 0, 360, 15600, 22500),
        model.add_client(590, 530, 5, 0, 360, 12000, 19500),
        model.add_client(435, 718, 3, 0, 420, 8400, 15300),
        model.add_client(1191, 639, 5, 0, 360, 12000, 19500),
    ]

    edge_weights = [
        [0, 1544, 1944, 1931, 1476],
        [1726, 0, 1992, 1427, 1593],
        [1965, 1975, 0, 621, 1090],
        [2063, 1433, 647, 0, 818],
        [1475, 1594, 1090, 828, 0],
    ]

    for idx, client in enumerate(clients, 1):
        from_depot = edge_weights[0][idx]
        to_depot = edge_weights[idx][0]

        model.add_edge(depot, client, from_depot, from_depot)
        model.add_edge(client, depot, to_depot, to_depot)

        for other_idx, other in enumerate(clients, 1):
            from_client = edge_weights[idx][other_idx]
            to_client = edge_weights[other_idx][idx]

            model.add_edge(client, other, from_client, from_client)
            model.add_edge(other, client, to_client, to_client)

    res = model.solve(stop=MaxIterations(100), seed=0)

    assert_(res.is_feasible())
    assert_equal(res.cost(), 9_155)


def test_model_solve_display_argument(ok_small, capsys):
    """
    Tests that solving a model displays solver progress when the ``display``
    argument is ``True``.
    """
    model = Model.from_data(ok_small)

    # First solve with display turned off. We should not see any output in this
    # case.
    model.solve(stop=MaxIterations(10), seed=0, display=False)
    printed = capsys.readouterr().out
    assert_equal(printed, "")

    # Now solve with display turned on. We should see output now.
    res = model.solve(stop=MaxIterations(10), seed=0, display=True)
    printed = capsys.readouterr().out

    # Check that some of the header data is in the output.
    assert_("PyVRP" in printed)
    assert_("Time" in printed)
    assert_("Iters" in printed)
    assert_("Feasible" in printed)
    assert_("Infeasible" in printed)

    # Check that we include the cost and total runtime in the output somewhere.
    assert_(str(round(res.cost())) in printed)
    assert_(str(round(res.runtime)) in printed)


def test_partial_distance_duration_matrix():
    """
    Tests that adding a full distance or duration matrix is not required. Any
    "missing" arcs are given large default values, ensuring they are unused.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    clients = [model.add_client(0, 1), model.add_client(1, 0)]

    # Only a subset of all possible edges exist. The model should be able to
    # handle that.
    model.add_edge(depot, clients[0], distance=1)
    model.add_edge(clients[0], clients[1], distance=2)
    model.add_edge(clients[1], depot, distance=1)

    model.add_vehicle_type(capacity=0, num_available=1)

    # These edges were not set, so their distance values should default to the
    # maximum value we use for such edges.
    data = model.data()
    distances = data.distance_matrix(profile=0)
    assert_equal(distances[0, 2], MAX_VALUE)
    assert_equal(distances[1, 0], MAX_VALUE)

    res = model.solve(stop=MaxIterations(100), seed=4)
    assert_equal(res.best.num_routes(), 1)
    assert_equal(res.cost(), 4)  # depot -> client 1 -> client 2 -> depot
    assert_(res.is_feasible())


def test_data_warns_about_scaling_issues(recwarn):
    """
    Tests that the modelling interface warns when an edge is added whose
    distance or duration values are too large. This situation is likely caused
    by scaling issues, so a warning is appropriate.
    """
    model = Model()
    model.add_vehicle_type(capacity=0, num_available=1)
    depot = model.add_depot(0, 0)
    client = model.add_client(1, 1)

    # Normal distance sizes; should all be OK.
    for distance in [1, 10, 100, 1_000, 10_000, 100_000, MAX_VALUE]:
        model.add_edge(client, depot, distance=distance)
        assert_equal(len(recwarn), 0)

    # But a value exceeding the maximum value is not OK. This should warn (both
    # for distance and/or duration).
    with assert_warns(ScalingWarning):
        model.add_edge(depot, client, distance=MAX_VALUE + 1)

    with assert_warns(ScalingWarning):
        model.add_edge(depot, client, distance=0, duration=MAX_VALUE + 1)


def test_model_solves_instance_with_zero_or_one_clients():
    """
    This test exercises the bug identified in issue #272, where the model
    could not solve an instance with zero clients or just one client.
    """
    m = Model()
    m.add_vehicle_type(capacity=15, num_available=1)
    depot = m.add_depot(x=0, y=0)

    # Solve an instance with no clients.
    res = m.solve(stop=MaxIterations(1))
    solution = [r.visits() for r in res.best.routes()]
    assert_equal(solution, [])

    # Solve an instance with one client.
    clients = [m.add_client(x=0, y=0)]
    m.add_edge(depot, clients[0], distance=0)
    m.add_edge(clients[0], depot, distance=0)

    res = m.solve(stop=MaxIterations(1))
    solution = [r.visits() for r in res.best.routes()]
    assert_equal(solution, [[1]])


def test_model_solves_small_instance_with_fixed_costs():
    """
    High-level test that creates and solves a small instance with vehicle fixed
    costs, to see if the model (and thus the underlying solution algorithm)
    can handle that. This test exercises the bug identified in issue #380.
    Before fixing this bug, the solver would hang on this instance.
    """
    m = Model()

    for idx in range(2):
        m.add_vehicle_type(
            capacity=0,
            num_available=5,
            fixed_cost=10,
            tw_early=0,
            tw_late=40,
        )

    m.add_depot(x=0, y=0)

    for idx in range(5):
        m.add_client(x=idx, y=idx, service_duration=1, tw_early=0, tw_late=20)

    for frm in m.locations:
        for to in m.locations:
            if frm != to:
                m.add_edge(frm, to, distance=0, duration=5)

    res = m.solve(stop=MaxIterations(100))
    assert_(res.is_feasible())


def test_model_solves_small_instance_with_shift_durations():
    """
    High-level test that creates and solves a small instance with shift
    durations, to see if the model (and thus the underlying solution algorithm)
    can handle that.
    """
    m = Model()

    # Two vehicle types with different shift time windows: the first has a
    # shift time window of [0, 15], the second of [5, 25]. There are four
    # vehicles in total, two for each vehicle type.
    for tw_early, tw_late in [(0, 15), (5, 25)]:
        m.add_vehicle_type(
            capacity=0,
            num_available=2,
            tw_early=tw_early,
            tw_late=tw_late,
        )

    m.add_depot(x=0, y=0)

    for idx in range(5):
        # Vehicles of the first type can visit two clients before having to
        # return to the depot. The second vehicle type can be used to visit
        # a single client before having to return to the depot. So we need
        # at least three routes.
        m.add_client(
            x=idx,
            y=idx,
            service_duration=1,
            tw_early=0,
            tw_late=20,
        )

    for frm in m.locations:
        for to in m.locations:
            if frm != to:
                m.add_edge(frm, to, distance=0, duration=5)

    res = m.solve(stop=MaxIterations(100))
    assert_(res.is_feasible())
    assert_(res.best.num_routes() > 2)


def test_model_solves_line_instance_with_multiple_depots():
    """
    High-level test that creates and solves a small instance on a line, where
    there are two depots. This test checks that the model and underlying
    algorithm correctly handle multiple depots.
    """
    m = Model()

    depot1 = m.add_depot(x=0, y=0)  # location 0
    depot2 = m.add_depot(x=5, y=0)  # location 1

    m.add_vehicle_type(1, start_depot=depot1, end_depot=depot1)
    m.add_vehicle_type(1, start_depot=depot2, end_depot=depot2)

    for idx in range(1, 5):  # locations 2, 3, 4, and 5
        m.add_client(x=idx, y=0)

    # All locations are on a horizontal line, with the depots on each end. The
    # line is organised as follows:
    #     D C C C C D   (depot or client)
    #     0 2 3 4 5 1   (location index)
    # Thus, clients 2 and 3 are closest to depot 0, and clients 4 and 5 are
    # closest to depot 1.
    for frm in m.locations:
        for to in m.locations:
            m.add_edge(frm, to, distance=abs(frm.x - to.x))

    res = m.solve(stop=MaxIterations(100), seed=3)
    assert_(res.is_feasible())

    # Test that there are two routes, with the clients closest to depot 0
    # assigned to the first route, and clients closest to depot 1 assigned to
    # the second route. Route membership is compared using sets because the
    # optimal visit order is not unique.
    routes = res.best.routes()
    assert_equal(set(routes[0].visits()), {2, 3})
    assert_equal(set(routes[1].visits()), {4, 5})


def test_client_depot_and_vehicle_type_name_fields():
    """
    Tests that name fields are properly passed to client, depot, and vehicle
    types.
    """
    m = Model()

    depot = m.add_depot(1, 1, name="depot1")
    assert_equal(depot.name, "depot1")
    assert_equal(str(depot), "depot1")

    veh_type = m.add_vehicle_type(name="veh_type1")
    assert_equal(veh_type.name, "veh_type1")
    assert_equal(str(veh_type), "veh_type1")

    client = m.add_client(1, 2, name="client1")
    assert_equal(client.name, "client1")
    assert_equal(str(client), "client1")


@pytest.mark.parametrize(
    ("pickups", "deliveries", "expected_excess_load"),
    [
        # The route should have 1 excess load (since the total pickup amount
        # sums to 11, and the vehicle capacity is 10). Same with similar
        # deliveries.
        ([1, 2, 3, 5], [0, 0, 0, 0], 1),
        ([0, 0, 0, 0], [1, 2, 3, 5], 1),
        ([1, 2, 3, 5], [1, 2, 3, 5], 1),
        # The following pickup and delivery schedule is tight, but should be
        # fine: the vehicle leaves full, and returns full, but there is a
        # configuration whereby it never exceeds its capacity along the way.
        ([1, 2, 3, 4], [4, 3, 2, 1], 0),
        # And no delivery or pickup amounts should of course also be OK!
        ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ],
)
def test_model_solves_instances_with_pickups_and_deliveries(
    pickups: list[int],
    deliveries: list[int],
    expected_excess_load: int,
):
    """
    High-level test that creates and solves a single-route instance where
    clients have pickups, deliveries, and sometimes both at the same time.
    """
    m = Model()
    m.add_depot(0, 0)
    m.add_vehicle_type(capacity=10)

    m.add_client(x=1, y=1, delivery=deliveries[0], pickup=pickups[0])
    m.add_client(x=2, y=2, delivery=deliveries[1], pickup=pickups[1])
    m.add_client(x=3, y=3, delivery=deliveries[2], pickup=pickups[2])
    m.add_client(x=4, y=4, delivery=deliveries[3], pickup=pickups[3])

    for frm in m.locations:
        for to in m.locations:
            manhattan = abs(frm.x - to.x) + abs(frm.y - to.y)
            m.add_edge(frm, to, distance=manhattan)

    res = m.solve(stop=MaxIterations(100))
    route = res.best.routes()[0]

    assert_equal(route.excess_load(), expected_excess_load)
    assert_equal(route.has_excess_load(), expected_excess_load > 0)


def test_add_client_raises_unknown_group():
    """
    Tests that the model's ``add_client`` method raises when it's given a group
    argument that is not known to the model.
    """
    m = Model()
    group = ClientGroup()

    with assert_raises(ValueError):
        m.add_client(1, 1, group=group)


def test_from_data_client_group(ok_small):
    """
    Test that creating a model from a given data instance with client groups
    correctly sets up the client groups in the model.
    """
    clients = ok_small.clients()
    clients[0] = Client(1, 1, required=False, group=0)
    clients[1] = Client(1, 1, required=False, group=0)

    group = ClientGroup([1, 2])

    data = ok_small.replace(clients=clients, groups=[group])
    model = Model.from_data(data)

    # Test that the clients have been correctly registered, and that there is
    # a client group in the model.
    assert_equal(model.locations[1].group, 0)
    assert_equal(model.locations[2].group, 0)
    assert_equal(len(model.groups), 1)

    # Test that that group actually contains the clients.
    group = model.groups[0]
    assert_(group.required)
    assert_equal(len(group), 2)
    assert_equal(group.clients, [1, 2])


def test_to_data_client_group():
    """
    Tests that creating a small model with a client group results in a correct
    ProblemData instance that has the appropriate group memberships set up.
    """
    m = Model()
    m.add_depot(1, 1)

    group = m.add_client_group()
    m.add_client(1, 1, required=False, group=group)
    m.add_client(2, 2, required=False, group=group)

    # Generate the data instance. There should be a single client group, and
    # the first two clients should be members of that group.
    data = m.data()
    assert_equal(data.num_groups, 1)

    group = data.group(0)
    assert_equal(group.clients, [1, 2])


def test_raises_mutually_exclusive_client_group_required_client():
    """
    Tests that adding a required client to a mutually exclusive client group
    raises, because that does not make any sense.
    """
    m = Model()

    group = m.add_client_group()
    assert_(group.mutually_exclusive)

    with assert_raises(ValueError):
        m.add_client(1, 1, required=True, group=group)


def test_client_group_membership_works_with_intermediate_changes():
    """
    Tests that repeatedly calling data() does not add clients more than once
    to each client group they are in, and everything continues to work when
    the model changes between calls to data().
    """
    m = Model()
    m.add_depot(1, 1)
    m.add_vehicle_type()

    # Add three clients to the model, with (for now) indices 1, 2, 3.
    group = m.add_client_group()
    m.add_client(1, 1, required=False, group=group)
    m.add_client(1, 1, required=False, group=group)
    m.add_client(1, 1, required=False, group=group)

    m.data()
    assert_equal(len(group), 3)
    assert_equal(group.clients, [1, 2, 3])

    # Add another depot and another client. The clients now have indices 2, 3,
    # 4, and 5.
    m.add_depot(1, 2)
    m.add_client(1, 1, required=False, group=group)

    m.data()
    assert_equal(len(group), 4)
    assert_equal(group.clients, [2, 3, 4, 5])


def test_tsp_instance_with_mutually_exclusive_groups(gtsp):
    """
    Smoke test that tests if the model can solve a generalised TSP instance
    where all clients are spread over fifty mutually exclusive groups.
    """
    m = Model.from_data(gtsp)
    res = m.solve(stop=MaxIterations(5))

    assert_(res.best.is_feasible())
    assert_(res.best.is_group_feasible())
    assert_equal(res.best.num_clients(), gtsp.num_groups)


def test_minimise_distance_or_duration(ok_small):
    """
    Small test that checks the model knows how to solve instances with
    different objective values.
    """
    orig_model = Model.from_data(ok_small)

    vehicle_types = [
        VehicleType(capacity=10, unit_distance_cost=1, unit_duration_cost=0),
        VehicleType(capacity=10, unit_distance_cost=0, unit_duration_cost=1),
    ]
    data = ok_small.replace(vehicle_types=vehicle_types)
    new_model = Model.from_data(data)

    orig_res = orig_model.solve(stop=MaxIterations(20), seed=82)
    new_res = new_model.solve(stop=MaxIterations(20), seed=82)

    assert_equal(orig_res.cost(), 9_155)
    assert_equal(new_res.cost(), 9_875)

    # The given instance has the same distance and duration matrix. There is
    # thus no difference in actual travel time or distance. But the duration
    # objective should also count service duration along the route, and that
    # is something we can check.
    service = sum(data.location(loc).service_duration for loc in [1, 4])
    assert_equal(new_res.cost(), orig_res.cost() + service)


def test_adding_vehicle_type_with_unknown_profile_raises():
    """
    Tests that adding a vehicle type with a routing profile that is not in the
    model raises.
    """
    m = Model()

    profile = Profile()
    assert_(profile not in m.profiles)

    with assert_raises(ValueError):
        m.add_vehicle_type(profile=profile)


def test_adding_multiple_routing_profiles():
    """
    Tests that adding multiple routing profiles to the model works, and the
    solver finds the optimal solution.
    """
    m = Model()

    profile1 = m.add_profile()
    assert_(m.profiles[0] is profile1)

    profile2 = m.add_profile()
    assert_(m.profiles[1] is profile2)

    veh_type1 = m.add_vehicle_type(profile=profile1)
    assert_equal(veh_type1.profile, 0)

    veh_type2 = m.add_vehicle_type(profile=profile2)
    assert_equal(veh_type2.profile, 1)

    m.add_depot(x=1, y=1)
    m.add_client(x=2, y=2)

    for frm in m.locations:
        for to in m.locations:
            if frm != to:
                m.add_edge(frm, to, distance=10, duration=5, profile=profile1)
                m.add_edge(frm, to, distance=5, duration=10, profile=profile2)

    data = m.data()
    assert_equal(data.num_profiles, 2)

    # Check that the distance and duration matrices of both profiles are
    # defined correctly.
    assert_equal(data.distance_matrix(profile=0), [[0, 10], [10, 0]])
    assert_equal(data.duration_matrix(profile=0), [[0, 5], [5, 0]])
    assert_equal(data.distance_matrix(profile=1), [[0, 5], [5, 0]])
    assert_equal(data.duration_matrix(profile=1), [[0, 10], [10, 0]])

    res = m.solve(stop=MaxIterations(10))
    assert_(res.is_feasible())
    assert_equal(res.cost(), 10)


def test_profiles_build_on_base_edges():
    """
    Tests that distance and duration matrices belonging to different profiles
    are all constructed from the same base matrices, with profile-specific
    changes applied on top.
    """
    m = Model()

    depot = m.add_depot(x=1, y=1)
    client = m.add_client(x=2, y=2)

    # Add base edges. These edges are used to construct base matrices that the
    # profiles build on. Essentially, if an edge is not specifically provided
    # in the profile, it will be inherited from the base edges.
    for frm in m.locations:
        for to in m.locations:
            dist = abs(frm.x - to.x) + abs(frm.y - to.y)
            m.add_edge(frm, to, dist)

    prof1 = m.add_profile()
    prof2 = m.add_profile()

    # We have not yet added profile-specific edges. This means the profile
    # matrices should all be the same as the base matrices.
    data = m.data()
    assert_equal(data.distance_matrix(0), [[0, 2], [2, 0]])
    assert_equal(data.distance_matrix(1), [[0, 2], [2, 0]])
    assert_equal(data.duration_matrix(0), np.zeros((2, 2)))
    assert_equal(data.duration_matrix(1), np.zeros((2, 2)))

    # Let's now add a few profile-specific edges and test that these overwrite
    # the base data in the new data instance.
    m.add_edge(depot, client, distance=5, duration=10, profile=prof1)
    m.add_edge(depot, client, distance=10, duration=5, profile=prof2)

    data = m.data()
    assert_equal(data.distance_matrix(0), [[0, 5], [2, 0]])
    assert_equal(data.distance_matrix(1), [[0, 10], [2, 0]])
    assert_equal(data.duration_matrix(0), [[0, 10], [0, 0]])
    assert_equal(data.duration_matrix(1), [[0, 5], [0, 0]])


def test_model_solves_instances_with_multiple_profiles():
    """
    Smoke test to check that the model knows how to solve an instance with
    multiple profiles.
    """
    m = Model()
    m.add_depot(x=1, y=1)
    m.add_client(x=1, y=2)
    m.add_client(x=2, y=1)

    prof1 = m.add_profile()  # this profile only cares about distance on x axis
    prof2 = m.add_profile()  # this one only about distance on y axis

    for frm in m.locations:
        for to in m.locations:
            m.add_edge(frm, to, abs(frm.x - to.x), profile=prof1)
            m.add_edge(frm, to, abs(frm.y - to.y), profile=prof2)

    m.add_vehicle_type(1, profile=prof1)
    m.add_vehicle_type(1, profile=prof2)

    # The best we can do is have the first vehicle visit the first client (no
    # distance), and the second vehicle the second client (also no distance).
    # The resulting cost is thus zero.
    res = m.solve(stop=MaxIterations(10))
    assert_equal(res.cost(), 0)

    route1, route2 = res.best.routes()
    assert_equal(route1.visits(), [1])
    assert_equal(route2.visits(), [2])
