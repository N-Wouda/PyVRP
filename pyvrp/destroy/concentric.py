import random

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
)


def concentric(
    data: ProblemData,
    solution: Solution,
    cost_eval: CostEvaluator,
    rng: RandomNumberGenerator,
):
    num_destroy = random.randint(10, 20)

    # Find all client indices to skip
    client = rng.randint(data.num_clients) + 1
    closest = data.distance_matrix(0)[client].argsort().tolist()
    closest.remove(client)
    top_k = closest[:num_destroy]

    # Rebuild the Solution but skip those clients
    routes = []
    for route in solution.routes():
        if visits := [idx for idx in route.visits() if idx not in top_k]:
            routes.append(Route(data, visits, route.vehicle_type()))

    return Solution(data, routes)
