from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

import numpy as np

from pyvrp._pyvrp import (
    Client,
    ClientGroup,
    Depot,
    ProblemData,
    VehicleType,
)
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import ScalingWarning
from pyvrp.solve import SolveParams, solve

if TYPE_CHECKING:
    from pyvrp.Result import Result
    from pyvrp.stop import StoppingCriterion


class Edge:
    """
    Stores an edge connecting two locations.

    Raises
    ------
    ValueError
        When either distance or duration is a negative value, or when self
        loops have nonzero distance or duration values.
    """

    __slots__ = ["frm", "to", "distance", "duration"]

    def __init__(
        self,
        frm: Union[Client, Depot],
        to: Union[Client, Depot],
        distance: int,
        duration: int,
    ):
        if distance < 0 or duration < 0:
            raise ValueError("Cannot have negative edge distance or duration.")

        if frm == to and (distance != 0 or duration != 0):
            raise ValueError("A self loop must have 0 distance and duration.")

        if max(distance, duration) > MAX_VALUE:
            msg = """
            The given distance or duration value is very large. This may impact
            numerical stability. Consider rescaling your input data.
            """
            warn(msg, ScalingWarning)

        self.frm = frm
        self.to = to
        self.distance = distance
        self.duration = duration


class Profile:
    """
    Stores a routing profile.

    A routing profile is a collection of edges with distance and duration
    attributes that together define a complete distance and duration matrix.
    These can be used to model, for example, the road uses of different types
    of vehicles, like trucks, cars, or bicyclists. Each
    :class:`~pyvrp._pyvrp.VehicleType` is associated with a routing profile.
    """

    edges: list[Edge]

    def __init__(self):
        self.edges = []

    def add_edge(
        self,
        frm: Union[Client, Depot],
        to: Union[Client, Depot],
        distance: int,
        duration: int = 0,
    ) -> Edge:
        """
        Adds a new edge to this routing profile.
        """
        edge = Edge(frm, to, distance, duration)
        self.edges.append(edge)
        return edge


class Model:
    """
    A simple interface for modelling vehicle routing problems with PyVRP.
    """

    def __init__(self) -> None:
        self._clients: list[Client] = []
        self._depots: list[Depot] = []
        self._edges: list[Edge] = []
        self._groups: list[ClientGroup] = []
        self._profiles: list[Profile] = []
        self._vehicle_types: list[VehicleType] = []

    @property
    def locations(self) -> list[Union[Client, Depot]]:
        """
        Returns all locations (depots and clients) in the current model. The
        clients in the routes of the solution returned by :meth:`~solve` can be
        used to index these locations.
        """
        return self._depots + self._clients

    @property
    def groups(self) -> list[ClientGroup]:
        """
        Returns all client groups currently in the model.
        """
        return self._groups

    @property
    def profiles(self) -> list[Profile]:
        """
        Returns all routing profiles currently in the model.
        """
        return self._profiles

    @property
    def vehicle_types(self) -> list[VehicleType]:
        """
        Returns the vehicle types in the current model. The routes of the
        solution returned by :meth:`~solve` have a property
        :meth:`~pyvrp._pyvrp.Route.vehicle_type()` that can be used to index
        these vehicle types.
        """
        return self._vehicle_types

    @classmethod
    def from_data(cls, data: ProblemData) -> "Model":
        """
        Constructs a model instance from the given data.

        Parameters
        ----------
        data
            Problem data to feed into the model.

        Returns
        -------
        Model
            A model instance representing the given data.
        """
        depots = data.depots()
        clients = data.clients()
        locs = depots + clients

        profiles = [Profile() for _ in range(data.num_profiles)]
        for idx, profile in enumerate(profiles):
            distances = data.distance_matrix(profile=idx)
            durations = data.duration_matrix(profile=idx)
            profile.edges = [
                Edge(
                    frm=locs[frm],
                    to=locs[to],
                    distance=distances[frm, to],
                    duration=durations[frm, to],
                )
                for frm in range(data.num_locations)
                for to in range(data.num_locations)
            ]

        self = Model()
        self._clients = clients
        self._depots = depots
        self._groups = data.groups()
        self._profiles = profiles
        self._vehicle_types = data.vehicle_types()

        return self

    def add_client(
        self,
        x: int,
        y: int,
        delivery: int = 0,
        pickup: int = 0,
        service_duration: int = 0,
        tw_early: int = 0,
        tw_late: int = np.iinfo(np.int64).max,
        release_time: int = 0,
        prize: int = 0,
        required: bool = True,
        group: Optional[ClientGroup] = None,
        *,
        name: str = "",
    ) -> Client:
        """
        Adds a client with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.Client` instance.

        Raises
        ------
        ValueError
            When ``group`` is not ``None``, and the given ``group`` is not part
            of this model instance, or when a required client is being added to
            a mutually exclusive client group.
        """
        if group is None:
            group_idx = None
        elif group in self._groups:
            group_idx = self._groups.index(group)
        else:
            raise ValueError("The given group is not in this model instance.")

        if required and group is not None and group.mutually_exclusive:
            # Required clients cannot be part of a mutually exclusive client
            # group, since then there's nothing to decide about.
            raise ValueError("Required client in mutually exclusive group.")

        client = Client(
            x=x,
            y=y,
            delivery=delivery,
            pickup=pickup,
            service_duration=service_duration,
            tw_early=tw_early,
            tw_late=tw_late,
            release_time=release_time,
            prize=prize,
            required=required,
            group=group_idx,
            name=name,
        )

        if group_idx is not None:
            client_idx = len(self._depots) + len(self._clients)
            self._groups[group_idx].add_client(client_idx)

        self._clients.append(client)
        return client

    def add_client_group(self, required: bool = True) -> ClientGroup:
        """
        Adds a new, possibly optional, client group to the model. Returns the
        created group.
        """
        group = ClientGroup(required=required)
        self._groups.append(group)
        return group

    def add_depot(
        self,
        x: int,
        y: int,
        *,
        name: str = "",
    ) -> Depot:
        """
        Adds a depot with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.Depot` instance.
        """
        depot = Depot(x=x, y=y, name=name)
        self._depots.append(depot)

        for group in self._groups:  # new depot invalidates client indices
            group.clear()

        for idx, client in enumerate(self._clients, len(self._depots)):
            if client.group is not None:
                self._groups[client.group].add_client(idx)

        return depot

    def add_edge(
        self,
        frm: Union[Client, Depot],
        to: Union[Client, Depot],
        distance: int,
        duration: int = 0,
        profile: Optional[Profile] = None,
    ) -> Edge:
        """
        Adds an edge :math:`(i, j)` between ``frm`` (:math:`i`) and ``to``
        (:math:`j`). The edge can be given distance and duration attributes.
        Distance is required, but the default duration is zero. Returns the
        created edge.

        .. note::

           If ``profile`` is not provided, the edge is a base edge that will be
           set for all profiles in the model. Any profile-specific edge takes
           precendence over a base edge with the same ``frm`` and ``to``
           locations.
        """
        if profile is not None:
            return profile.add_edge(frm, to, distance, duration)

        edge = Edge(frm=frm, to=to, distance=distance, duration=duration)
        self._edges.append(edge)
        return edge

    def add_profile(self) -> Profile:
        """
        Adds a new routing profile to the model.
        """
        profile = Profile()
        self._profiles.append(profile)
        return profile

    def add_vehicle_type(
        self,
        num_available: int = 1,
        capacity: int = 0,
        start_depot: Optional[Depot] = None,
        end_depot: Optional[Depot] = None,
        fixed_cost: int = 0,
        tw_early: int = 0,
        tw_late: int = np.iinfo(np.int64).max,
        max_duration: int = np.iinfo(np.int64).max,
        max_distance: int = np.iinfo(np.int64).max,
        unit_distance_cost: int = 1,
        unit_duration_cost: int = 0,
        profile: Optional[Profile] = None,
        *,
        name: str = "",
    ) -> VehicleType:
        """
        Adds a vehicle type with the given attributes to the model. Returns the
        created :class:`~pyvrp._pyvrp.VehicleType` instance.

        .. note::

           The vehicle type is assigned to the first depot if no depot
           information is provided.

        Raises
        ------
        ValueError
            When the given ``depot`` or ``profile`` arguments are not in this
            model instance.
        """
        if start_depot is None:
            start_idx = 0
        elif start_depot in self._depots:
            start_idx = self._depots.index(start_depot)
        else:
            raise ValueError("The given start depot is not in this model.")

        if end_depot is None:
            end_idx = 0
        elif end_depot in self._depots:
            end_idx = self._depots.index(end_depot)
        else:
            raise ValueError("The given end depot is not in this model.")

        if profile is None:
            profile_idx = 0
        elif profile in self._profiles:
            profile_idx = self._profiles.index(profile)
        else:
            raise ValueError("The given profile is not in this model.")

        vehicle_type = VehicleType(
            num_available=num_available,
            capacity=capacity,
            start_depot=start_idx,
            end_depot=end_idx,
            fixed_cost=fixed_cost,
            tw_early=tw_early,
            tw_late=tw_late,
            max_duration=max_duration,
            max_distance=max_distance,
            unit_distance_cost=unit_distance_cost,
            unit_duration_cost=unit_duration_cost,
            profile=profile_idx,
            name=name,
        )

        self._vehicle_types.append(vehicle_type)
        return vehicle_type

    def data(self) -> ProblemData:
        """
        Creates and returns a :class:`~pyvrp._pyvrp.ProblemData` instance
        from this model's attributes.
        """
        locs = self.locations
        loc2idx = {id(loc): idx for idx, loc in enumerate(locs)}

        # First we create the base distance and duration matrices. These are
        # shared by all routing profiles. If an edge was not specified, we use
        # a large default value here.
        base_distance = np.full((len(locs), len(locs)), MAX_VALUE, np.int64)
        base_duration = np.full((len(locs), len(locs)), MAX_VALUE, np.int64)
        np.fill_diagonal(base_distance, 0)
        np.fill_diagonal(base_duration, 0)

        for edge in self._edges:
            frm = loc2idx[id(edge.frm)]
            to = loc2idx[id(edge.to)]
            base_distance[frm, to] = edge.distance
            base_duration[frm, to] = edge.duration

        # Now we create the profile-specific distance and duration matrices.
        # These are based on the base matrices.
        distances = []
        durations = []
        for profile in self._profiles:
            prof_distance = base_distance.copy()
            prof_duration = base_duration.copy()

            for edge in profile.edges:
                frm = loc2idx[id(edge.frm)]
                to = loc2idx[id(edge.to)]
                prof_distance[frm, to] = edge.distance
                prof_duration[frm, to] = edge.duration

            distances.append(prof_distance)
            durations.append(prof_duration)

        # When the user has not provided any profiles, we create an implicit
        # first profile from the base matrices.
        if not self._profiles:
            distances = [base_distance]
            durations = [base_duration]

        return ProblemData(
            self._clients,
            self._depots,
            self.vehicle_types,
            distances,
            durations,
            self._groups,
        )

    def solve(
        self,
        stop: StoppingCriterion,
        seed: int = 0,
        collect_stats: bool = True,
        display: bool = True,
        params: SolveParams = SolveParams(),
    ) -> Result:
        """
        Solve this model.

        Parameters
        ----------
        stop
            Stopping criterion to use.
        seed
            Seed value to use for the random number stream. Default 0.
        collect_stats
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display
            Whether to display information about the solver progress. Default
            ``True``. Progress information is only available when
            ``collect_stats`` is also set, which it is by default.
        params
            Solver parameters to use. If not provided, a default will be used.

        Returns
        -------
        Result
            A Result object, containing statistics (if collected) and the best
            found solution.
        """
        return solve(self.data(), stop, seed, collect_stats, display, params)
