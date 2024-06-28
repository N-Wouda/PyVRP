import math
from dataclasses import dataclass

from pyvrp.Statistics import Statistics
from pyvrp._pyvrp import CostEvaluator, Solution


@dataclass
class Result:
    """
    Stores the outcomes of a single run. An instance of this class is returned
    once the GeneticAlgorithm completes.

    Parameters
    ----------
    best
        The best observed solution.
    stats
        A Statistics object containing runtime statistics.
    num_iterations
        Number of iterations performed by the genetic algorithm.
    runtime
        Total runtime of the main genetic algorithm loop.

    Raises
    ------
    ValueError
        When the number of iterations or runtime are negative.
    """

    best: Solution
    stats: Statistics
    num_iterations: int
    runtime: float

    def __post_init__(self):
        if self.num_iterations < 0:
            raise ValueError("Negative number of iterations not understood.")

        if self.runtime < 0:
            raise ValueError("Negative runtime not understood.")

    def cost(self) -> float:
        """
        Returns the cost (objective) value of the best solution. Returns inf
        if the best solution is infeasible.
        """
        if not self.best.is_feasible():
            return math.inf

        return CostEvaluator(0, 0, 0).cost(self.best)

    def is_feasible(self) -> bool:
        """
        Returns whether the best solution is feasible.
        """
        return self.best.is_feasible()

    def summary(self) -> str:
        """
        Returns a nicely formatted result summary.
        """
        obj_str = f"{self.cost()}" if self.is_feasible() else "INFEASIBLE"
        summary = [
            "Solution results",
            "================",
            f"    # routes: {self.best.num_routes()}",
            f"   # clients: {self.best.num_clients()}",
            f"   objective: {obj_str}",
            f"    distance: {self.best.distance()}",
            f"    duration: {self.best.duration()}",
            f"# iterations: {self.num_iterations}",
            f"    run-time: {self.runtime:.2f} seconds",
        ]

        return "\n".join(summary)

    def __str__(self) -> str:
        content = [
            self.summary(),
            "",
            "Routes",
            "------",
            str(self.best),
        ]

        return "\n".join(content)
