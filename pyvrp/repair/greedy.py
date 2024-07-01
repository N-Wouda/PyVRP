import random as random

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.repair import greedy_repair


def greedy(
    data: ProblemData,
    solution: Solution,
    cost_eval: CostEvaluator,
    rng: RandomNumberGenerator,
) -> Solution:
    """
    Small wrapper around ``greedy_repair`` to have Solution's as input and
    output.
    """
    visited = {idx for route in solution.routes() for idx in route.visits()}
    clients = range(data.num_depots, data.num_locations)
    unplanned = [idx for idx in clients if idx not in visited]
    random.shuffle(unplanned)

    routes = greedy_repair(solution.routes(), unplanned, data, cost_eval)
    not_empty = [route for route in routes if route.visits()]
    return Solution(data, not_empty)
