from typing import Tuple

import numpy as np

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)

from ._crossover import (
    heterogeneous_selective_route_exchange as _heterogeneous_srex,
)
from ._crossover import (
    selective_route_exchange as _srex,
)


def selective_route_exchange(
    parents: Tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    rng: RandomNumberGenerator,
) -> Solution:
    """
    This crossover operator due to Nagata and Kobayashi (2010) combines routes
    from both parents to generate a new offspring solution. It does this by
    carefully selecting routes from the second parent that could be exchanged
    with routes from the first parent. After exchanging these routes, the
    resulting offspring solution is repaired using a greedy repair strategy.

    The selective route exchange (SREX) selects a set of routes for each parent
    and replaces the selected routes of the first parent with those of the
    second parent. The routes are selected by maximimizing the overlap between
    the clients in the two sets of routes. This is achieved through a heuristic
    that iteratively shifts adjacent routes until no further improvement in
    maximizing the overlap is observed. Then, two offspring are generated by
    replacing the selected routes in two distinct ways, and the offspring with
    the lowest cost is returned.

    The standard SREX does not take into account vehicle types in the crossover
    i.e. the crossover is performed as if there are no vehicle types. Routes in
    the offspring get assigned the vehicle type of the original route in the
    parent. Due to the exchange, this may violate the number of available
    vehicles for a type, which is resolved by greedily reassigning some vehicle
    types for routes that were exchanged using the first available type.


    Parameters
    ----------
    parents
        The two parent solutions to create an offspring from.
    data
        The problem instance.
    cost_evaluator
        The cost evaluator to be used during the greedy repair step.
    rng
        The random number generator to use.

    Returns
    -------
    Solution
        A new offspring.

    References
    ----------
    .. [1] Nagata, Y., & Kobayashi, S. (2010). A Memetic Algorithm for the
           Pickup and Delivery Problem with Time Windows Using Selective Route
           Exchange Crossover. *Parallel Problem Solving from Nature*, PPSN XI,
           536 - 545.
    """
    first, second = parents

    if first.num_clients() == 0:
        return second

    if second.num_clients() == 0:
        return first

    idx1 = rng.randint(first.num_routes())
    idx2 = idx1 if idx1 < second.num_routes() else 0
    max_routes_to_move = min(first.num_routes(), second.num_routes())
    num_routes_to_move = rng.randint(max_routes_to_move) + 1

    return _srex(
        parents, data, cost_evaluator, (idx1, idx2), num_routes_to_move
    )


def heterogeneous_selective_route_exchange(
    parents: Tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    rng: RandomNumberGenerator,
) -> Solution:
    """
    SREX variant that handles heterogeneous vehicle types seperately. This
    version groups routes by vehicle types and performs the SREX crossover per
    vehicle type, ensuring exchanges only take place between routes of the same
    vehicle type. Start indices are optimized per vehicle type as well.

    Since a single vehicle type may have different sets of customers assigned
    in both parents, the greedy repair phase is not done per vehicle type but
    at the end over all routes.

    Parameters
    ----------
    parents
        The two parent solutions to create an offspring from.
    data
        The problem instance.
    cost_evaluator
        The cost evaluator to be used during the greedy repair step.
    rng
        The random number generator to use.

    Returns
    -------
    Solution
        A new offspring.

    References
    ----------
    .. [1] Nagata, Y., & Kobayashi, S. (2010). A Memetic Algorithm for the
           Pickup and Delivery Problem with Time Windows Using Selective Route
           Exchange Crossover. *Parallel Problem Solving from Nature*, PPSN XI,
           536 - 545.
    """
    first, second = parents
    num_routes_1 = np.bincount(
        [r.vehicle_type() for r in first.get_routes()],
        minlength=data.num_vehicle_types,
    )
    num_routes_2 = np.bincount(
        [r.vehicle_type() for r in second.get_routes()],
        minlength=data.num_vehicle_types,
    )
    max_routes_to_move = np.minimum(num_routes_1, num_routes_2)

    # Note: it is allowed to move 0 routes of one type, since in theory there
    # can be only one vehicle of a type in which case we want the route for
    # that type from either the one or the other parent.
    # As such, it can happen that the offspring exchanges 0 routes.
    num_routes_to_move = [
        rng.randint(max_r + 1) for max_r in max_routes_to_move
    ]
    start_indices = [
        0 if num_r == 0 else rng.randint(num_r) for num_r in num_routes_1
    ]

    return _heterogeneous_srex(
        parents, data, cost_evaluator, start_indices, num_routes_to_move
    )
