import random as rnd

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
)


def random(
    data: ProblemData,
    solution: Solution,
    cost_eval: CostEvaluator,
    rng: RandomNumberGenerator,
    perturbation_strength=30,
) -> Solution:
    """
    Randomly removes a number of clients from the solution.

    Parameters
    ----------
    TODO

    Returns
    -------
    Solution
        The destroyed solution.
    """
    clients = [idx for route in solution.routes() for idx in route.visits()]
    num_destroy = rng.randint(perturbation_strength) + 1

    # Find all client indices to keep
    rnd.shuffle(clients)
    keep = set(clients[num_destroy:])

    # Rebuild the Solution but skip those clients
    routes = []
    for route in solution.routes():
        if visits := [idx for idx in route.visits() if idx in keep]:
            routes.append(Route(data, visits, route.vehicle_type()))

    return Solution(data, routes)
