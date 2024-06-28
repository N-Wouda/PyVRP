import pathlib
from collections import defaultdict
from itertools import count
from numbers import Number
from typing import Callable, Optional, Union
from warnings import warn

import numpy as np
import vrplib

from pyvrp._pyvrp import Client, ClientGroup, Depot, ProblemData, VehicleType
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning

_Routes = list[list[int]]
_RoundingFunc = Callable[[np.ndarray], np.ndarray]

_INT_MAX = np.iinfo(np.int64).max


ROUND_FUNCS: dict[str, _RoundingFunc] = {
    "round": lambda vals: np.round(vals).astype(np.int64),
    "trunc": lambda vals: vals.astype(np.int64),
    "dimacs": lambda vals: (10 * vals).astype(np.int64),
    "exact": lambda vals: np.round(1_000 * vals).astype(np.int64),
    "none": lambda vals: vals,
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

    parser = _InstanceParser(vrplib.read_instance(where), round_func)
    builder = _ProblemDataBuilder(parser)
    return builder.data()


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


class _InstanceParser:
    """
    read() helper that parses VRPLIB data into meaningful parts for further
    processing.
    """

    def __init__(self, instance: dict, round_func: _RoundingFunc):
        self.instance = instance
        self.round_func = round_func

    @property
    def num_locations(self) -> int:
        return self.instance["dimension"]

    @property
    def num_depots(self) -> int:
        return self.instance.get("depot", np.array([0])).size

    @property
    def num_clients(self) -> int:
        return self.num_locations - self.num_depots

    @property
    def num_vehicles(self) -> int:
        return self.instance.get("vehicles", self.num_locations - 1)

    def type(self) -> str:
        return self.instance.get("type", "")

    def edge_weight(self) -> np.ndarray:
        return self.round_func(self.instance["edge_weight"])

    def depot_idcs(self) -> np.ndarray:
        return self.instance.get("depot", np.array([0]))

    def backhauls(self) -> np.ndarray:
        if "backhaul" not in self.instance:
            return np.zeros(self.num_locations, dtype=np.int64)

        return self.round_func(self.instance["backhaul"])

    def demands(self) -> np.ndarray:
        if "demand" not in self.instance and "linehaul" not in self.instance:
            return np.zeros(self.num_locations, dtype=np.int64)

        return self.round_func(
            self.instance.get("demand", self.instance.get("linehaul"))
        )

    def coords(self) -> np.ndarray:
        if "node_coord" not in self.instance:
            return np.zeros((self.num_locations, 2), dtype=np.int64)

        return self.round_func(self.instance["node_coord"])

    def service_times(self) -> np.ndarray:
        if "service_time" not in self.instance:
            return np.zeros(self.num_locations, dtype=np.int64)

        service_times = self.instance["service_time"]

        if isinstance(service_times, Number):
            # Some instances describe a uniform service time as a single value
            # that applies to all clients.
            service_times = np.full(self.num_locations, service_times)
            service_times[0] = 0

        return self.round_func(service_times)

    def time_windows(self) -> np.ndarray:
        if "time_window" not in self.instance:
            time_windows = np.empty((self.num_locations, 2), dtype=np.int64)
            time_windows[:, 0] = 0
            time_windows[:, 1] = _INT_MAX
            return time_windows

        return self.round_func(self.instance["time_window"])

    def release_times(self) -> np.ndarray:
        if "release_time" not in self.instance:
            return np.zeros(self.num_locations, dtype=np.int64)

        return self.round_func(self.instance["release_time"])

    def prizes(self) -> np.ndarray:
        if "prize" not in self.instance:
            return np.zeros(self.num_locations, dtype=np.int64)

        return self.round_func(self.instance["prize"])

    def capacities(self) -> np.ndarray:
        if "capacity" not in self.instance:
            return np.full(self.num_vehicles, _INT_MAX)

        capacities = self.instance["capacity"]

        if isinstance(capacities, Number):
            # Some instances describe a uniform capacity as a single value
            # that applies to all vehicles.
            capacities = np.full(self.num_vehicles, capacities)

        return self.round_func(capacities)

    def allowed_clients(self) -> list[tuple[int, ...]]:
        if "vehicles_allowed_clients" not in self.instance:
            client_idcs = tuple(range(self.num_depots, self.num_locations))
            return [client_idcs for _ in range(self.num_vehicles)]

        return [
            tuple(idx - 1 for idx in clients)
            for clients in self.instance["vehicles_allowed_clients"]
        ]

    def vehicles_depots(self) -> np.ndarray:
        if "vehicles_depot" not in self.instance:
            depot_idcs = self.depot_idcs()
            return np.full(self.num_vehicles, depot_idcs[0])

        return self.instance["vehicles_depot"] - 1  # zero-indexed

    def max_distances(self) -> np.ndarray:
        if "vehicles_max_distance" not in self.instance:
            return np.full(self.num_vehicles, _INT_MAX)

        max_distances = self.instance["vehicles_max_distance"]

        if isinstance(max_distances, Number):
            # Some instances describe a uniform max distance as a single
            # value that applies to all vehicles.
            max_distances = np.full(self.num_vehicles, max_distances)

        return self.round_func(max_distances)

    def max_durations(self) -> np.ndarray:
        if "vehicles_max_duration" not in self.instance:
            return np.full(self.num_vehicles, _INT_MAX)

        max_durations = self.instance["vehicles_max_duration"]

        if isinstance(max_durations, Number):
            # Some instances describe a uniform max duration as a single
            # value that applies to all vehicles.
            max_durations = np.full(self.num_vehicles, max_durations)

        return self.round_func(max_durations)

    def mutually_exclusive_groups(self) -> list[list[int]]:
        if "mutually_exclusive_group" not in self.instance:
            return []

        group_data = self.instance["mutually_exclusive_group"]

        # This assumes groups are numeric, and are numbered {1, 2, ...}.
        raw_groups: list[list[int]] = [[] for _ in range(max(group_data))]
        for client, group in enumerate(group_data):
            raw_groups[group - 1].append(client)

        # Only keep groups if they have more than one member. Empty groups or
        # groups with one member are trivial to decide, so there is no point
        # in keeping them.
        return [group for group in raw_groups if len(group) > 1]


class _ProblemDataBuilder:
    """
    read() helper that builds a ``ProblemData`` object from the instance
    attributes of the given parser.
    """

    def __init__(self, parser: _InstanceParser):
        self.parser = parser

    def data(self) -> ProblemData:
        clients = self._clients()
        depots = self._depots()
        vehicle_types = self._vehicle_types()
        distance_matrices = self._distance_matrices()
        groups = self._groups()

        return ProblemData(
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=distance_matrices,
            # VRPLIB instances typically do not have a duration data field, and
            # instead assume duration == distance.
            duration_matrices=distance_matrices,
            groups=groups,
        )

    def _depots(self) -> list[Depot]:
        num_depots = self.parser.num_depots
        depot_idcs = self.parser.depot_idcs()

        contiguous_lower_idcs = np.arange(num_depots)
        if num_depots == 0 or (depot_idcs != contiguous_lower_idcs).any():
            msg = """
            Source file should contain at least one depot in the contiguous
            lower indices, starting from 1.
            """
            raise ValueError(msg)

        coords = self.parser.coords()
        return [
            Depot(x=coords[idx][0], y=coords[idx][1])
            for idx in range(num_depots)
        ]

    def _clients(self) -> list[Client]:
        groups = self.parser.mutually_exclusive_groups()
        num_locs = self.parser.num_locations

        idx2group: list[Optional[int]] = [None for _ in range(num_locs)]
        for group, members in enumerate(groups):
            for client in members:
                idx2group[client] = group

        coords = self.parser.coords()
        demands = self.parser.demands()
        backhauls = self.parser.backhauls()
        service_duration = self.parser.service_times()
        time_windows = self.parser.time_windows()
        release_times = self.parser.release_times()
        prizes = self.parser.prizes()  # we interpret a zero-prize client as
        required = np.isclose(prizes, 0)  # required in the benchmark instances

        return [
            Client(
                x=coords[idx][0],
                y=coords[idx][1],
                delivery=demands[idx],
                pickup=backhauls[idx],
                service_duration=service_duration[idx],
                tw_early=time_windows[idx][0],
                tw_late=time_windows[idx][1],
                release_time=release_times[idx],
                prize=prizes[idx],
                required=required[idx] and idx2group[idx] is None,
                group=idx2group[idx],
            )
            for idx in range(self.parser.num_depots, num_locs)
        ]

    def _vehicle_types(self) -> list[VehicleType]:
        num_vehicles = self.parser.num_vehicles
        vehicles_data = (
            self.parser.capacities(),
            self.parser.allowed_clients(),
            self.parser.vehicles_depots(),
            self.parser.max_distances(),
            self.parser.max_durations(),
        )

        if any(len(attr) != num_vehicles for attr in vehicles_data):
            msg = """
            The number of elements in the vehicles data attributes should be
            equal to the number of vehicles in the problem.
            """
            raise ValueError(msg)

        # VRPLIB instances includes data for each available vehicle. We group
        # vehicles by their attributes to create unique vehicle types.
        type2idcs = defaultdict(list)
        for vehicle, veh_type in enumerate(zip(*vehicles_data)):
            type2idcs[veh_type].append(vehicle)

        client2profile = self._allowed2profile()
        time_windows = self.parser.time_windows()

        vehicle_types = []
        for attributes, vehicles in type2idcs.items():
            (capacity, clients, depot_idx, max_dist, max_duration) = attributes

            vehicle_type = VehicleType(
                num_available=len(vehicles),
                capacity=capacity,
                start_depot=depot_idx,
                end_depot=depot_idx,
                # The literature specifies depot time windows. We do not have
                # depot time windows but instead set those on the vehicles.
                tw_early=time_windows[depot_idx][0],
                tw_late=time_windows[depot_idx][1],
                max_duration=max_duration,
                max_distance=max_dist,
                profile=client2profile[clients],
                # A bit hacky, but this csv-like name is really useful to track
                # the actual vehicles that make up this vehicle type.
                name=",".join(map(str, vehicles)),
            )
            vehicle_types.append(vehicle_type)

        return vehicle_types

    def _distance_matrices(self) -> list[np.ndarray]:
        distances = self.parser.edge_weight()

        if self.parser.type() == "VRPB":
            # In VRPB, linehauls must be served before backhauls. This can be
            # enforced by setting a high value for the distance/duration from
            # depot to backhaul (forcing linehaul to be served first) and a
            # large value from backhaul to linehaul (avoiding linehaul after
            # backhaul clients).
            linehaul = self.parser.demands() > 0
            backhaul = self.parser.backhauls() > 0
            distances[0, backhaul] = MAX_VALUE
            distances[np.ix_(backhaul, linehaul)] = MAX_VALUE

        allowed2profile = self._allowed2profile()
        num_profiles = len(allowed2profile)
        dist_mats = [distances.copy() for _ in range(num_profiles)]

        for allowed_clients, type_idx in allowed2profile.items():
            if len(allowed_clients) == self.parser.num_clients:
                # True if this feature is unused, and the distance matrix for
                # this profile does not have to be modified.
                continue

            num_depots = self.parser.num_depots
            num_locations = self.parser.num_locations

            # This profile is allowed to visit every depot and all clients in
            # its allowed clients section.
            allowed = np.zeros((num_locations,), dtype=bool)
            allowed[:num_depots] = True
            allowed[list(allowed_clients)] = True

            # Some dtype trickery to ensure the MAX_VALUE assignment below does
            # not overflow.
            dtype = np.promote_types(dist_mats[type_idx].dtype, np.int64)
            dist_mats[type_idx] = dist_mats[type_idx].astype(dtype)

            # Set MAX_VALUE to and from disallowed clients, preventing this
            # vehicle type from serving them.
            dist_mat = dist_mats[type_idx]
            dist_mat[:, ~allowed] = dist_mat[~allowed, :] = MAX_VALUE
            np.fill_diagonal(dist_mat, 0)

        if any(dist.max() > MAX_VALUE for dist in dist_mats):
            msg = """
            The maximum distance value is very large. This might impact 
            numerical stability. Consider rescaling your input data.
            """
            warn(msg, ScalingWarning)

        return dist_mats

    def _groups(self) -> list[ClientGroup]:
        groups = self.parser.mutually_exclusive_groups()
        return [ClientGroup(group) for group in groups]

    def _allowed2profile(self) -> dict[tuple[int, ...], int]:
        allowed_clients2profile_idx = {}
        profile_idx = count(0)
        for clients in self.parser.allowed_clients():
            if clients not in allowed_clients2profile_idx:
                allowed_clients2profile_idx[clients] = next(profile_idx)

        return allowed_clients2profile_idx
