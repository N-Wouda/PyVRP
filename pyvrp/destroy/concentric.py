import random

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
)
from pyvrp.search import NeighbourhoodParams, compute_neighbours


def concentric(
    data: ProblemData,
    solution: Solution,
    cost_eval: CostEvaluator,
    rng: RandomNumberGenerator,
):
    params = NeighbourhoodParams(nb_granular=data.num_locations)
    neighbourhood = compute_neighbours(data, params)

    clients = [idx for route in solution.routes() for idx in route.visits()]
    num_destroy = rng.randint(50) + 1

    # Find all client indices to skip
    skip = set()

    random.shuffle(clients)
    current = random.choice(clients)
    skip.add(current)

    while len(skip) < num_destroy:
        candidate = random.choice(neighbourhood[current])
        if candidate in skip:
            continue

        skip.add(candidate)
        current = candidate

    # Rebuild the Solution but skip those clients
    routes = []
    for route in solution.routes():
        if visits := [idx for idx in route.visits() if idx not in skip]:
            routes.append(Route(data, visits, route.vehicle_type()))

    return Solution(data, routes)
