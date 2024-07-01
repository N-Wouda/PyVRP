from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyvrp._pyvrp import ProblemData

if TYPE_CHECKING:
    from pyvrp import ProblemData
    from pyvrp._pyvrp import Solution


@dataclass
class NeighbourhoodParams:
    """
    Configuration for calculating a granular neighbourhood.

    Attributes
    ----------
    weight_wait_time
        Penalty weight given to the minimum wait time aspect of the proximity
        calculation. A large wait time indicates the clients are far apart
        in duration/time.
    weight_time_warp
        Penalty weight given to the minimum time warp aspect of the proximity
        calculation. A large time warp indicates the clients are far apart in
        duration/time.
    nb_granular
        Number of other clients that are in each client's granular
        neighbourhood. This parameter determines the size of the overall
        neighbourhood.
    symmetric_proximity
        Whether to calculate a symmetric proximity matrix. This ensures edge
        :math:`(i, j)` is given the same weight as :math:`(j, i)`.
    symmetric_neighbours
        Whether to symmetrise the neighbourhood structure. This ensures that
        when edge :math:`(i, j)` is in, then so is :math:`(j, i)`. Note that
        this is *not* the same as ``symmetric_proximity``.

    Raises
    ------
    ValueError
        When ``nb_granular`` is non-positive.
    """

    weight_wait_time: float = 0.2
    weight_time_warp: float = 1.0
    nb_granular: int = 50
    symmetric_proximity: bool = True
    symmetric_neighbours: bool = False

    def __post_init__(self):
        if self.nb_granular <= 0:
            raise ValueError("nb_granular <= 0 not understood.")


def compute_neighbours(
    data: ProblemData, params: NeighbourhoodParams = NeighbourhoodParams()
) -> list[list[int]]:
    """
    Computes neighbours defining the neighbourhood for a problem instance.

    Parameters
    ----------
    data
        ProblemData for which to compute the neighbourhood.
    params
        NeighbourhoodParams that define how the neighbourhood is computed.

    Returns
    -------
    list
        A list of list of integers representing the neighbours for each client.
        The first lists in the lower indices are associated with the depots and
        are all empty.
    """
    proximity = _compute_proximity(
        data,
        params.weight_wait_time,
        params.weight_time_warp,
    )

    if params.symmetric_proximity:
        proximity = np.minimum(proximity, proximity.T)

    for group in data.groups():
        if group.mutually_exclusive:
            # Clients in mutually exclusive groups cannot neighbour each other,
            # since only one of them can be in the solution at any given time.
            # We use max float, not infty, to ensure these clients are ordered
            # before the depots: we want to avoid same group neighbours, but it
            # is not problematic if we need to have them.
            idcs = np.ix_(group.clients, group.clients)
            proximity[idcs] = np.finfo(np.float64).max

    np.fill_diagonal(proximity, np.inf)  # cannot be in own neighbourhood
    proximity[: data.num_depots, :] = np.inf  # depots have no neighbours
    proximity[:, : data.num_depots] = np.inf  # clients do not neighbour depots

    k = min(params.nb_granular, data.num_clients - 1)  # excl. self
    top_k = np.argsort(proximity, axis=1, kind="stable")[data.num_depots :, :k]

    if not params.symmetric_neighbours:
        return [[] for _ in range(data.num_depots)] + top_k.tolist()

    # Construct a symmetric adjacency matrix and return the adjacent clients
    # as the neighbourhood structure.
    adj = np.zeros_like(proximity, dtype=bool)
    rows = np.expand_dims(np.arange(data.num_depots, len(proximity)), axis=1)
    adj[rows, top_k] = True
    adj = adj | adj.transpose()

    return [np.flatnonzero(row).tolist() for row in adj]


def _compute_proximity(
    data: ProblemData, weight_wait_time: float, weight_time_warp: float
) -> np.ndarray[float]:
    """
    Computes proximity for neighborhood. Proximity is based on [1]_, with
    modification for additional VRP variants.

    Parameters
    ----------
    data
        ProblemData for which to compute proximity.
    params
        NeighbourhoodParams that define how proximity is computed.

    Returns
    -------
    np.ndarray[float]
        An array of size :py:attr:`~pyvrp._pyvrp.ProblemData.num_locations`
        by :py:attr:`~pyvrp._pyvrp.ProblemData.num_locations`.

    References
    ----------
    .. [1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). A
           hybrid genetic algorithm with adaptive diversity management for a
           large class of vehicle routing problems with time-windows.
           *Computers & Operations Research*, 40(1), 475 - 489.
    """
    early = np.zeros((data.num_locations,))
    early[data.num_depots :] = np.asarray([c.tw_early for c in data.clients()])

    late = np.zeros((data.num_locations,))
    late[data.num_depots :] = np.asarray([c.tw_late for c in data.clients()])

    service = np.zeros_like(early)
    service[data.num_depots :] = [c.service_duration for c in data.clients()]

    prize = np.zeros_like(early)
    prize[data.num_depots :] = [client.prize for client in data.clients()]

    # We first determine the elementwise minimum cost across all vehicle types.
    # This is the cheapest way any edge can be traversed.
    distances = data.distance_matrices()
    durations = data.duration_matrices()
    edge_costs = [  # edge costs per vehicle type
        veh_type.unit_distance_cost * distances[veh_type.profile]
        + veh_type.unit_duration_cost * durations[veh_type.profile]
        for veh_type in data.vehicle_types()
    ]

    # Minimum wait time and time warp of visiting j directly after i.
    min_duration = np.minimum.reduce(durations)
    min_wait = early[None, :] - min_duration - service[:, None] - late[:, None]
    min_tw = early[:, None] + service[:, None] + min_duration - late[None, :]

    # Proximity is based on edge costs (and rewards) and penalties for known
    # time-related violations.
    return (
        np.minimum.reduce(edge_costs, dtype=float)
        - prize[None, :]
        + weight_wait_time * np.maximum(min_wait, 0)
        + weight_time_warp * np.maximum(min_tw, 0)
    )


class NeighbourhoodManager:
    def __init__(
        self,
        data: ProblemData,
        neighbourhood: list[list[int]],
        params: NeighbourhoodParams = NeighbourhoodParams(),
    ):
        self._data = data
        self._neighbourhood = neighbourhood
        self._params = params

    def __call__(
        self, solution1: Solution, solution2: Solution
    ) -> list[list[int]]:
        """
        Finds client candidates for search. The candidates are those
        who are in routes that are different between solution1 and solution2.
        """
        modified = set()

        neighbours1 = solution1.neighbours()
        neighbours2 = solution2.neighbours()

        for idx in range(self._data.num_depots, self._data.num_locations):
            pred1, succ1 = neighbours1[idx]
            pred2, succ2 = neighbours2[idx]

            if neighbours1[idx] != neighbours2[idx]:
                modified.add(idx)
                modified.add(pred1)
                modified.add(succ1)
                modified.add(pred2)
                modified.add(succ2)

        nbhd = []
        for idx, clients in enumerate(self.neighbourhood):
            if idx in modified:
                nbhd.append(clients)
            else:
                nbhd.append([])
                # restricted = [idx for idx in clients if idx in modified]
                # nbhd.append(restricted)

        # if len([v for v in nbhd if v]) == 0:
        #     nbhd = self.neighbourhood

        return nbhd

    def update(self):
        pass

    @property
    def neighbourhood(self) -> list[list[int]]:
        return self._neighbourhood
