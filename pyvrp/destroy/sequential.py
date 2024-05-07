import random

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Route,
    Solution,
)


def sequential(
    data: ProblemData,
    solution: Solution,
    cost_eval: CostEvaluator,
    rng: RandomNumberGenerator,
) -> Solution:
    perturbation_strength = random.randint(10, 20)
    # Select a random route
    routes = solution.routes()
    current = routes.pop(rng.randint(len(routes)))
    cands = []
    pool = []

    # Continue until enough routes are destroyed
    while len(cands) < perturbation_strength:
        max_string_size = perturbation_strength - len(cands)
        new = remove_string(current, max_string_size, data, rng)
        if len(new) > 0:
            pool.append(new)

        cands.extend(
            [idx for idx in current.visits() if idx not in new.visits()]
        )

        if len(cands) >= perturbation_strength:
            break

        # Find the nearest route
        distances = [dist(new, route) for route in routes]
        idx = distances.index(min(distances))
        current = routes.pop(idx)

        # # Select random route
        # current = routes.pop(rng.randint(len(routes)))

    return Solution(data, routes + pool)


def dist(route1, route2):
    x1, y1 = route1.centroid()
    x2, y2 = route2.centroid()
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def remove_string(route, max_string_size, data, rng):
    """
    Remove a string that constains the passed-in customer.
    """
    # Find consecutive indices to remove that contain the customer
    size = rng.randint(min(len(route), max_string_size)) + 1
    start = rng.randint(len(route))
    skip = {(start + idx) % len(route) for idx in range(size)}

    visits = [
        client for idx, client in enumerate(route.visits()) if idx not in skip
    ]

    return Route(data, visits, route.vehicle_type())
