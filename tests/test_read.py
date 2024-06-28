from math import sqrt

import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal,
    assert_raises,
    assert_warns,
)

from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning
from tests.helpers import read


@pytest.mark.parametrize(
    ("where", "exception"),
    [
        ("data/UnknownEdgeWeightFmt.txt", ValueError),
        ("data/UnknownEdgeWeightType.txt", ValueError),
        ("somewhere that does not exist", FileNotFoundError),
        ("data/DepotNotOne.txt", ValueError),
        ("data/DepotsNotLowerIndices.txt", ValueError),
        ("data/TimeWindowOpenLargerThanClose.txt", ValueError),
        ("data/EdgeWeightsNoExplicit.txt", ValueError),
        ("data/EdgeWeightsNotFullMatrix.txt", ValueError),
        ("data/MissingVehicleData.txt", ValueError),
    ],
)
def test_raises_invalid_file(where: str, exception: Exception):
    """
    Tests that ``read()`` raises when there are issues with the given file.
    """
    with assert_raises(exception):
        read(where)


def test_raises_unknown_round_func():
    """
    Tests that ``read()`` raises when the rounding function is not known.
    """
    with assert_raises(TypeError):
        # Unknown round_func, so should raise.
        read("data/OkSmall.txt", round_func="asdbsadfas")

    # Is the default round_func, so should not raise.
    read("data/OkSmall.txt", round_func="none")


def test_reading_OkSmall_instance():
    """
    Tests that the parsed data from the "OkSmall" instance is correct.
    """
    data = read("data/OkSmall.txt")

    # From the DIMENSION, VEHICLES, and CAPACITY fields in the file.
    assert_equal(data.num_clients, 4)
    assert_equal(data.num_vehicles, 3)
    assert_equal(data.num_vehicle_types, 1)
    assert_equal(data.vehicle_type(0).capacity, 10)

    # From the NODE_COORD_SECTION in the file
    expected = [
        (2334, 726),
        (226, 1297),
        (590, 530),
        (435, 718),
        (1191, 639),
    ]

    for loc in range(data.num_locations):
        assert_equal(data.location(loc).x, expected[loc][0])
        assert_equal(data.location(loc).y, expected[loc][1])

    # From the EDGE_WEIGHT_SECTION in the file
    expected = [
        [0, 1544, 1944, 1931, 1476],
        [1726, 0, 1992, 1427, 1593],
        [1965, 1975, 0, 621, 1090],
        [2063, 1433, 647, 0, 818],
        [1475, 1594, 1090, 828, 0],
    ]

    # For instances read through VRPLIB/read(), distance is duration. So the
    # dist/durs should be the same as the expected edge weights above.
    assert_equal(data.num_profiles, 1)
    assert_equal(data.distance_matrix(profile=0), expected)
    assert_equal(data.duration_matrix(profile=0), expected)

    # From the DEMAND_SECTION in the file
    expected = [0, 5, 5, 3, 5]

    for loc in range(1, data.num_locations):  # excl. depot (has no delivery)
        assert_equal(data.location(loc).delivery, expected[loc])

    # From the TIME_WINDOW_SECTION in the file
    expected = [
        (0, 45000),
        (15600, 22500),
        (12000, 19500),
        (8400, 15300),
        (12000, 19500),
    ]

    for loc in range(data.num_depots, data.num_locations):
        assert_equal(data.location(loc).tw_early, expected[loc][0])
        assert_equal(data.location(loc).tw_late, expected[loc][1])

    # Vehicle time window is derived from the depot's time window in the
    # TIME_WINDOW_SECTION of the file.
    vehicle_type = data.vehicle_type(0)
    assert_equal(vehicle_type.tw_early, expected[0][0])
    assert_equal(vehicle_type.tw_late, expected[0][1])

    # From the SERVICE_TIME_SECTION in the file
    expected = [0, 360, 360, 420, 360]

    for loc in range(1, data.num_locations):  # excl. depot (has no service)
        assert_equal(data.location(loc).service_duration, expected[loc])


def test_reading_vrplib_instance():
    """
    Tests that a small VRPLIB-style instance is correctly parsed.
    """
    data = read("data/E-n22-k4.txt", round_func="dimacs")

    assert_equal(data.num_clients, 21)
    assert_equal(data.num_depots, 1)
    assert_equal(data.num_locations, 22)
    assert_equal(data.vehicle_type(0).capacity, 60_000)

    assert_equal(len(data.depots()), data.num_depots)
    assert_equal(len(data.clients()), data.num_clients)
    assert_equal(data.num_locations, data.num_depots + data.num_clients)

    # Coordinates are scaled by 10 to align with 1 decimal distance precision
    assert_equal(data.location(0).x, 1450)  # depot [x, y] location
    assert_equal(data.location(0).y, 2150)

    assert_equal(data.location(1).x, 1510)  # first customer [x, y] location
    assert_equal(data.location(1).y, 2640)

    # The data file specifies distances as 2D Euclidean. We take that and
    # should compute integer equivalents with up to one decimal precision.
    # For depot -> first customer:
    # For depot -> first customer:
    #      dX = 151 - 145 = 6
    #      dY = 264 - 215 = 49
    #      dist = sqrt(dX^2 + dY^2) = 49.37
    #      int(10 * dist) = 493
    assert_equal(data.num_profiles, 1)
    distances = data.distance_matrix(profile=0)
    assert_equal(distances[0, 1], 493)
    assert_equal(distances[1, 0], 493)

    # This is a CVRP instance, so all other fields should have default values.
    for loc in range(1, data.num_locations):
        assert_equal(data.location(loc).service_duration, 0)
        assert_equal(data.location(loc).tw_early, 0)
        assert_equal(data.location(loc).tw_late, np.iinfo(np.int64).max)
        assert_equal(data.location(loc).release_time, 0)
        assert_equal(data.location(loc).prize, 0)
        assert_equal(data.location(loc).required, True)


def test_warns_about_scaling_issues():
    """
    Tests that ``read()`` warns about scaling issues when a distance value is
    very large.
    """
    with assert_warns(ScalingWarning):
        # The arc from the depot to client 4 is really large (one billion), so
        # that should trigger a warning.
        read("data/ReallyLargeDistance.txt")


def test_round_func_round_nearest():
    """
    Tests rounding to the nearest integer works well for the RC208 instance,
    which has Euclidean distances computed from integer coordinates. Since the
    instance is large, we'll test one particular distance.
    """
    data = read("data/RC208.vrp", "round")

    # We're going to test dist(0, 1) and dist(1, 0), which should be the same
    # since the distances are symmetric/Euclidean.
    assert_equal(data.location(0).x, 40)
    assert_equal(data.location(0).y, 50)

    assert_equal(data.location(1).x, 25)
    assert_equal(data.location(1).y, 85)

    # Compute the distance, and assert that it is indeed correctly rounded.
    distances = data.distance_matrix(profile=0)
    expected_dist = round(sqrt((40 - 25) ** 2 + (85 - 50) ** 2))
    assert_equal(distances[0, 1], expected_dist)
    assert_equal(distances[1, 0], expected_dist)


def test_round_func_exact():
    """
    Tests rounding with the ``exact`` round function also works well for the
    RC208 instance. This test is similar to the one for ``round``, but all
    values are now multiplied by 1_000 before rounding.
    """
    data = read("data/RC208.vrp", "exact")

    # We're going to test dist(0, 1) and dist(1, 0), which should be the same
    # since the distances are symmetric/Euclidean.
    assert_equal(data.location(0).x, 40_000)
    assert_equal(data.location(0).y, 50_000)

    assert_equal(data.location(1).x, 25_000)
    assert_equal(data.location(1).y, 85_000)

    # Compute the distance, and assert that it is indeed correctly rounded.
    distances = data.distance_matrix(profile=0)
    expected_dist = round(sqrt((40 - 25) ** 2 + (85 - 50) ** 2) * 1_000)
    assert_equal(distances[0, 1], expected_dist)
    assert_equal(distances[1, 0], expected_dist)


def test_service_time_specification():
    """
    Tests that specifying the service time as a specification (key-value pair)
    results in a uniform service time for all clients.
    """
    data = read("data/ServiceTimeSpecification.txt")

    # Clients should all have the same service time.
    services = [client.service_duration for client in data.clients()]
    assert_allclose(services, 360)


def test_multiple_depots():
    """
    Tests parsing a slightly modified version of the OkSmall instance, which
    now has two depots rather than one.
    """
    data = read("data/OkSmallMultipleDepots.txt")

    # Still five locations, but now with two depots and three clients.
    assert_equal(data.num_locations, 5)
    assert_equal(data.num_depots, 2)
    assert_equal(data.num_clients, 3)
    assert_equal(data.num_vehicle_types, 2)  # two, each at a different depot
    assert_equal(data.num_vehicles, 3)

    # First vehicle type should have two vehicles at the first depot.
    veh_type1 = data.vehicle_type(0)
    assert_equal(veh_type1.profile, 0)
    assert_equal(veh_type1.start_depot, 0)
    assert_equal(veh_type1.end_depot, 0)
    assert_equal(veh_type1.num_available, 2)
    assert_equal(veh_type1.tw_early, 0)
    assert_equal(veh_type1.tw_late, 45_000)

    # Second vehicle type should have one vehicle at the second depot. The
    # vehicle should have a tighter time window than that associated with the
    # first vehicle type.
    veh_type2 = data.vehicle_type(1)
    assert_equal(veh_type2.profile, 0)
    assert_equal(veh_type2.start_depot, 1)
    assert_equal(veh_type2.end_depot, 1)
    assert_equal(veh_type2.num_available, 1)
    assert_equal(veh_type2.tw_early, 5_000)
    assert_equal(veh_type2.tw_late, 20_000)

    depot1, depot2 = data.depots()

    # Test that the depot coordinates have been parsed correctly.
    assert_equal(depot1.x, 2_334)
    assert_equal(depot1.y, 726)
    assert_equal(depot2.x, 226)
    assert_equal(depot2.y, 1_297)


def test_mdvrptw_instance():
    """
    Tests that reading an MDVRPTW instance happens correctly, particularly the
    maximum route duration and multiple depot aspects.
    """
    data = read("data/PR11A.vrp", round_func="trunc")

    assert_equal(data.num_locations, 364)
    assert_equal(data.num_depots, 4)
    assert_equal(data.num_clients, 360)

    assert_equal(data.num_vehicles, 40)
    assert_equal(data.num_vehicle_types, 4)  # one vehicle type per depot

    for idx, vehicle_type in enumerate(data.vehicle_types()):
        # There should be ten vehicles for each depot, with the following
        # capacities and maximum route durations.
        assert_equal(vehicle_type.num_available, 10)
        assert_equal(vehicle_type.start_depot, idx)
        assert_equal(vehicle_type.end_depot, idx)
        assert_equal(vehicle_type.capacity, 200)
        assert_equal(vehicle_type.max_duration, 450)

        # Essentially all vehicle indices for each depot, separated by a comma.
        # Each depot has ten vehicles, and they are nicely grouped (so the
        # first ten are assigned to the first depot, the second ten to the
        # second depot, etc.).
        expected_name = ",".join(str(10 * idx + veh) for veh in range(10))
        assert_equal(vehicle_type.name, expected_name)

    # We haven't seen many instances with negative coordinates, but this
    # MDVRPTW instance has those. That should be allowed.
    assert_(any(depot.x < 0) for depot in data.depots())
    assert_(any(depot.y < 0) for depot in data.depots())
    assert_(any(client.x < 0) for client in data.clients())
    assert_(any(client.y < 0) for client in data.clients())


def test_vrpspd_instance():
    """
    Tests that reading an VRPSPD instance happens correctly, particularly the
    linehaul and backhaul data.
    """
    data = read("data/SmallVRPSPD.vrp", round_func="round")

    assert_equal(data.num_locations, 5)
    assert_equal(data.num_depots, 1)
    assert_equal(data.num_clients, 4)

    assert_equal(data.num_vehicles, 4)
    assert_equal(data.num_vehicle_types, 1)

    vehicle_type = data.vehicle_type(0)
    assert_equal(vehicle_type.num_available, 4)
    assert_equal(vehicle_type.capacity, 200)

    # The first client is a linehaul client (only delivery, no pickup), and
    # the second client is a backhaul client (only pickup, no delivery). All
    # other clients have both delivery and pickup.
    deliveries = [1, 0, 16, 18]
    pickups = [0, 3, 10, 40]

    for idx, client in enumerate(data.clients()):
        assert_equal(client.delivery, deliveries[idx])
        assert_equal(client.pickup, pickups[idx])

    # Test that distance/duration are not set to a large value, as in VRPB.
    assert_equal(np.max(data.distance_matrix(profile=0)), 39)
    assert_equal(np.max(data.duration_matrix(profile=0)), 39)


def test_vrpb_instance():
    """
    Tests that reading an VRPB instance happens correctly, particularly the
    backhaul data and modified distances to ensure linehaul is served before
    backhaul.
    """
    data = read("data/X-n101-50-k13.vrp", round_func="round")

    assert_equal(data.num_locations, 101)
    assert_equal(data.num_depots, 1)
    assert_equal(data.num_clients, 100)

    assert_equal(data.num_vehicles, 100)
    assert_equal(data.num_vehicle_types, 1)

    vehicle_type = data.vehicle_type(0)
    assert_equal(vehicle_type.num_available, 100)
    assert_equal(vehicle_type.capacity, 206)

    # The first 50 clients are linehaul, the rest are backhaul.
    clients = data.clients()

    for client in clients[:50]:
        assert_equal(client.pickup, 0)
        assert_(client.delivery > 0)

    for client in clients[50:]:
        assert_(client.pickup > 0)
        assert_equal(client.delivery, 0)

    # Tests that distance/duration from depot to backhaul clients is set to
    # ``MAX_VALUE``, as well as for backhaul to linehaul clients.
    linehauls = set(range(1, 51))
    backhauls = set(range(51, 101))

    distances = data.distance_matrix(profile=0)
    durations = data.duration_matrix(profile=0)

    for frm in range(data.num_locations):
        for to in range(data.num_locations):
            depot2back = frm == 0 and to in backhauls
            back2line = frm in backhauls and to in linehauls

            if depot2back or back2line:
                assert_equal(distances[frm, to], MAX_VALUE)
                assert_equal(durations[frm, to], MAX_VALUE)
            else:
                assert_(distances[frm, to] < MAX_VALUE)
                assert_(durations[frm, to] < MAX_VALUE)


def test_max_distance_constraint():
    """
    Tests a small instance with a maximum distance constraint, to confirm those
    constraints are properly recognised and parsed.
    """
    data = read("data/OkSmallMaxDistance.txt")
    for vehicle_type in data.vehicle_types():
        assert_equal(vehicle_type.max_distance, 5_000)


def test_reading_mutually_exclusive_group():
    """
    Tests that read() correctly parses a small instance with mutually exclusive
    client groups.
    """
    data = read("data/OkSmallMutuallyExclusiveGroups.txt")
    assert_equal(data.num_groups, 1)

    group = data.group(0)
    assert_equal(len(group), 3)
    assert_equal(group.clients, [1, 2, 3])

    for client in data.group(0):
        client_data = data.location(client)  # type: ignore
        assert_equal(client_data.required, False)
        assert_equal(client_data.group, 0)


def test_reading_allowed_clients():
    """
    Tests that read() correctly parses a small instance with allowed clients
    for each vehicle.
    """
    data = read("data/OkSmallAllowedClients.txt")
    assert_equal(data.num_vehicle_types, 2)

    veh_type1 = data.vehicle_type(0)
    assert_equal(veh_type1.capacity, 10)
    assert_equal(veh_type1.num_available, 2)
    assert_equal(veh_type1.profile, 0)

    distance_matrix = data.distance_matrix(veh_type1.profile)
    duration_matrix = data.duration_matrix(veh_type1.profile)

    # First vehicle type has no vehicle-client restrictions.
    assert_(np.all(distance_matrix != MAX_VALUE))
    assert_(np.all(duration_matrix != MAX_VALUE))

    veh_type2 = data.vehicle_type(1)
    assert_equal(veh_type2.capacity, 10)
    assert_equal(veh_type2.num_available, 1)
    assert_equal(veh_type2.profile, 1)

    distance_matrix = data.distance_matrix(veh_type2.profile)
    duration_matrix = data.duration_matrix(veh_type2.profile)

    # Second vehicle type is not allowed to serve client idx 4.
    assert_equal(distance_matrix[:3, 4], MAX_VALUE)
    assert_equal(distance_matrix[4, :3], MAX_VALUE)
    assert_equal(duration_matrix[:3, 4], MAX_VALUE)
    assert_equal(duration_matrix[4, :3], MAX_VALUE)


def test_sdvrptw_instance():
    """
    Tests that reading an SDVRPTW instance happens correctly, particularly the
    heterogeneous vehicles data sections.
    """
    data = read("data/PR01.vrp")

    # One routing profile per unique client group.
    assert_equal(data.num_profiles, 4)
    assert_equal(len(data.distance_matrices()), 4)
    assert_equal(len(data.duration_matrices()), 4)

    # Each vehicle type has a different capacity. We only check the first two.
    veh_type1 = data.vehicle_type(0)
    assert_equal(veh_type1.num_available, 2)
    assert_equal(veh_type1.capacity, 100)
    assert_equal(veh_type1.max_duration, 500)
    assert_equal(veh_type1.profile, 0)

    veh_type2 = data.vehicle_type(1)
    assert_equal(veh_type2.num_available, 2)
    assert_equal(veh_type2.capacity, 150)
    assert_equal(veh_type2.max_duration, 500)
    assert_equal(veh_type2.profile, 1)

    # The first vehicle type cannot serve clients 38-48. Let's check that the
    # distance and duration matrices reflect this.
    distance_matrix = data.distance_matrix(veh_type1.profile)
    duration_matrix = data.duration_matrix(veh_type1.profile)

    for client in range(38, 48):
        # This avoids checking diagonals.
        idcs = [idx for idx in range(data.num_locations) if idx != client]
        assert_equal(distance_matrix[idcs, client], MAX_VALUE)
        assert_equal(distance_matrix[client, idcs], MAX_VALUE)
        assert_equal(duration_matrix[idcs, client], MAX_VALUE)
        assert_equal(duration_matrix[client, idcs], MAX_VALUE)
