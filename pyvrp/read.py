import pathlib
from collections import defaultdict
from numbers import Number
from typing import Callable, Union
from warnings import warn

import numpy as np
import vrplib

from pyvrp._pyvrp import Client, ClientGroup, Depot, ProblemData, VehicleType
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning

_Routes = list[list[int]]
_RoundingFunc = Callable[[np.ndarray], np.ndarray]

_INT_MAX = np.iinfo(np.int64).max


def round_nearest(vals: np.ndarray):
    return np.round(vals).astype(np.int64)


def truncate(vals: np.ndarray):
    return vals.astype(np.int64)


def dimacs(vals: np.ndarray):
    return (10 * vals).astype(np.int64)


def exact(vals: np.ndarray):
    return round_nearest(1_000 * vals)


def no_rounding(vals):
    return vals


ROUND_FUNCS: dict[str, _RoundingFunc] = {
    "round": round_nearest,
    "trunc": truncate,
    "dimacs": dimacs,
    "exact": exact,
    "none": no_rounding,
}


def read(
    where: Union[str, pathlib.Path],
    round_func: Union[str, _RoundingFunc] = "none",
) -> ProblemData:
    """
    Reads the ``VRPLIB`` file at the given location, and returns a
    :class:`~pyvrp._pyvrp.ProblemData` instance.

    .. note::

       See the
       :doc:`VRPLIB format explanation <../dev/supported_vrplib_fields>` page
       for more details.

    Parameters
    ----------
    where
        File location to read. Assumes the data on the given location is in
        ``VRPLIB`` format.
    round_func
        Optional rounding function that is applied to all data values in the
        instance. This can either be a function or a string:

            * ``'round'`` rounds the values to the nearest integer;
            * ``'trunc'`` truncates the values to an integer;
            * ``'dimacs'`` scales by 10 and truncates the values to an integer;
            * ``'exact'`` scales by 1000 and rounds to the nearest integer.
            * ``'none'`` does no rounding. This is the default.

    Raises
    ------
    TypeError
        When ``round_func`` does not name a rounding function, or is not
        callable.
    ValueError
        When the data file does not provide information on the problem size.

    Returns
    -------
    ProblemData
        Data instance constructed from the read data.
    """
    if (key := str(round_func)) in ROUND_FUNCS:
        round_func = ROUND_FUNCS[key]

    if not callable(round_func):
        raise TypeError(
            f"round_func = {round_func} is not understood. Can be a function,"
            f" or one of {ROUND_FUNCS.keys()}."
        )

    instance = vrplib.read_instance(where)

    # VRPLIB instances typically do not have a duration data field, so we
    # assume duration == distance.
    durations = distances = round_func(instance["edge_weight"])

    dimension: int = instance.get("dimension", durations.shape[0])
    depot_idcs: np.ndarray = instance.get("depot", np.array([0]))
    num_depots = len(depot_idcs)
    num_vehicles: int = instance.get("vehicles", dimension - 1)

    if isinstance(instance.get("capacity"), np.ndarray):
        capacities = round_func(instance["capacity"])
    elif "capacity" in instance:
        capacities = round_func(np.full(num_vehicles, instance["capacity"]))
    else:
        capacities = np.full(num_vehicles, _INT_MAX)

    if "vehicles_allowed_clients" in instance:
        vehicles_allowed_clients = [
            tuple(idx - 1 for idx in clients)
            for clients in instance["vehicles_allowed_clients"]
        ]
    else:
        vehicles_allowed_clients = [
            tuple(idx for idx in range(num_depots, dimension))
            for _ in range(num_vehicles)
        ]

    if "vehicles_depot" in instance:
        vehicles_depots = instance["vehicles_depot"] - 1  # zero-indexed
    else:
        vehicles_depots = np.full(num_vehicles, depot_idcs[0])

    if isinstance(instance.get("vehicles_max_distance"), np.ndarray):
        vehicles_max_distance = round_func(instance["vehicles_max_distance"])
    elif "vehicles_max_distance" in instance:
        vehicles_max_distance = round_func(
            np.full(num_vehicles, instance["vehicles_max_distance"])
        )
    else:
        vehicles_max_distance = np.full(num_vehicles, _INT_MAX)

    if isinstance(instance.get("vehicles_max_duration"), np.ndarray):
        vehicles_max_duration = round_func(instance["vehicles_max_duration"])
    elif "vehicles_max_duration" in instance:
        vehicles_max_duration = round_func(
            np.full(num_vehicles, instance["vehicles_max_duration"])
        )
    else:
        vehicles_max_duration = np.full(num_vehicles, _INT_MAX)

    if "backhaul" in instance:
        backhauls: np.ndarray = round_func(instance["backhaul"])
    else:
        backhauls = np.zeros(dimension, dtype=np.int64)

    if "demand" in instance or "linehaul" in instance:
        demands: np.ndarray = instance.get("demand", instance.get("linehaul"))
        demands = round_func(demands)
    else:
        demands = np.zeros(dimension, dtype=np.int64)

    if "node_coord" in instance:
        coords: np.ndarray = round_func(instance["node_coord"])
    else:
        coords = np.zeros((dimension, 2), dtype=np.int64)

    if "service_time" in instance:
        if isinstance(instance["service_time"], Number):
            # Some instances describe a uniform service time as a single value
            # that applies to all clients.
            service_times = np.full(dimension, instance["service_time"])
            service_times[0] = 0
            service_times = round_func(service_times)
        else:
            service_times = round_func(instance["service_time"])
    else:
        service_times = np.zeros(dimension, dtype=np.int64)

    if "time_window" in instance:
        time_windows: np.ndarray = round_func(instance["time_window"])
    else:
        # No time window data. So the time window component is not relevant.
        time_windows = np.empty((dimension, 2), dtype=np.int64)
        time_windows[:, 0] = 0
        time_windows[:, 1] = _INT_MAX

    if "release_time" in instance:
        release_times: np.ndarray = round_func(instance["release_time"])
    else:
        release_times = np.zeros(dimension, dtype=np.int64)

    default_prizes = np.zeros(dimension, dtype=np.int64)
    prizes = round_func(instance.get("prize", default_prizes))

    if "mutually_exclusive_group" in instance:
        group_data = instance["mutually_exclusive_group"]

        # This assumes groups are numeric, and are numbered {1, 2, ...}.
        raw_groups: list[list[int]] = [[] for _ in range(max(group_data))]
        for client, group in enumerate(group_data):
            raw_groups[group - 1].append(client)

        # Only keep groups if they have more than one member. Empty groups or
        # groups with one member are trivial to decide, so there is no point
        # in keeping them.
        groups = [ClientGroup(group) for group in raw_groups if len(group) > 1]
    else:
        groups = []

    if instance.get("type") == "VRPB":
        # In VRPB, linehauls must be served before backhauls. This can be
        # enforced by setting a high value for the distance/duration from depot
        # to backhaul (forcing linehaul to be served first) and a large value
        # from backhaul to linehaul (avoiding linehaul after backhaul clients).
        linehaul = np.flatnonzero(demands > 0)
        backhaul = np.flatnonzero(backhauls > 0)
        distances[0, backhaul] = MAX_VALUE
        distances[np.ix_(backhaul, linehaul)] = MAX_VALUE

    vehicles_data = (
        capacities,
        vehicles_allowed_clients,
        vehicles_depots,
        vehicles_max_distance,
        vehicles_max_duration,
    )
    if any(len(arr) != num_vehicles for arr in vehicles_data):
        msg = """
        The number of elements in the vehicles data attributes should be equal
        to the number of vehicles in the problem.
        """
        raise ValueError(msg)

    # Checks
    contiguous_lower_idcs = np.arange(num_depots)
    if num_depots == 0 or (depot_idcs != contiguous_lower_idcs).any():
        msg = """
        Source file should contain at least one depot in the contiguous lower
        indices, starting from 1.
        """
        raise ValueError(msg)

    if max(distances.max(), durations.max()) > MAX_VALUE:
        msg = """
        The maximum distance or duration value is very large. This might
        impact numerical stability. Consider rescaling your input data.
        """
        warn(msg, ScalingWarning)

    depots = [
        Depot(x=coords[idx][0], y=coords[idx][1]) for idx in range(num_depots)
    ]

    idx2group = [None for _ in range(dimension)]
    for group, members in enumerate(groups):
        for client in members:
            idx2group[client] = group

    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            delivery=demands[idx],
            pickup=backhauls[idx],
            service_duration=service_times[idx],
            tw_early=time_windows[idx][0],
            tw_late=time_windows[idx][1],
            release_time=release_times[idx],
            prize=prizes[idx],
            required=np.isclose(prizes[idx], 0) and idx2group[idx] is None,
            group=idx2group[idx],
        )
        for idx in range(num_depots, dimension)
    ]

    veh_type2idcs = defaultdict(list)
    for idx, veh_type in enumerate(zip(*vehicles_data)):
        veh_type2idcs[veh_type].append(idx)

    vehicle_types = []
    distance_matrices = []
    duration_matrices = []

    for type_idx, (attributes, vehicles) in enumerate(veh_type2idcs.items()):
        (
            capacity,
            allowed_clients,
            depot_idx,
            max_distance,
            max_duration,
        ) = attributes

        vehicle_type = VehicleType(
            num_available=len(vehicles),
            capacity=capacity,
            start_depot=depot_idx,
            end_depot=depot_idx,
            # The literature specifies depot time windows. We do not have depot
            # time windows but instead set those on the vehicles, generalising
            # the depot time windows.
            tw_early=time_windows[depot_idx][0],
            tw_late=time_windows[depot_idx][1],
            max_duration=max_duration,
            max_distance=max_distance,
            profile=type_idx,
            # A bit hacky, but this csv-like name is really useful to track the
            # actual vehicles that make up this vehicle type.
            name=",".join(map(str, vehicles)),
        )
        vehicle_types.append(vehicle_type)

        dist = distances.copy()
        dur = durations.copy()

        for idx in range(num_depots, dimension):
            if idx not in allowed_clients:
                # Set MAX_VALUE to and from disallowed clients, preventing
                # this vehicle type from serving them.
                dist[:, idx] = dist[idx, :] = MAX_VALUE
                dur[:, idx] = dur[idx, :] = MAX_VALUE

        np.fill_diagonal(dist, 0)
        np.fill_diagonal(dur, 0)

        distance_matrices.append(dist)
        duration_matrices.append(dur)

    return ProblemData(
        clients,
        depots,
        vehicle_types,
        distance_matrices,
        duration_matrices,
        groups,
    )


def read_solution(where: Union[str, pathlib.Path]) -> _Routes:
    """
    Reads a solution in ``VRPLIB`` format from the give file location, and
    returns the routes contained in it.

    Parameters
    ----------
    where
        File location to read. Assumes the solution in the file on the given
        location is in ``VRPLIB`` solution format.

    Returns
    -------
    list
        List of routes, where each route is a list of client numbers.
    """
    sol = vrplib.read_solution(str(where))
    return sol["routes"]  # type: ignore
