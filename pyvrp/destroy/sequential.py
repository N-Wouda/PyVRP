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
    perturbation_strength: int = 30,
    MAX_STRING_SIZE: int = 10,
) -> Solution:
    avg_route_size = solution.num_clients() // solution.num_routes()
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    num_strings = perturbation_strength // max_string_size
    max_string_removals = min(solution.num_routes(), num_strings)

    # TODO find seed

    # Select a random route
    routes = solution.routes()
    current = routes.pop(rng.randint(len(routes)))
    pool = []

    # Continue until enough routes are destroyed
    for _ in range(max_string_removals):
        new = remove_string(current, max_string_size, data, rng)
        if len(new) > 0:
            pool.append(new)

        if _ == max_string_removals - 1:
            break

        # Find the nearest route
        distances = [dist(new, route) for route in routes]
        idx = distances.index(min(distances))
        current = routes.pop(idx)

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

    if rng.rand() < 0.5:
        visits = [
            client
            for idx, client in enumerate(route.visits())
            if idx not in skip
        ]
    else:
        visits = [
            client for idx, client in enumerate(route.visits()) if idx in skip
        ]

    return Route(data, visits, route.vehicle_type())
