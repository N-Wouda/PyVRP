from typing import Protocol

from pyvrp._pyvrp import CostEvaluator, Solution


class SearchMethod(Protocol):  # pragma: no cover
    """
    Protocol that search methods must implement.
    """

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
    ) -> Solution:
        """
        Search around the given solution, and returns a new solution that is
        hopefully better.

        Parameters
        ----------
        solution
            The solution to improve.
        cost_evaluator
            Cost evaluator to use when evaluating improvements.

        Returns
        -------
        Solution
            The improved solution.
        """

    def register(
        self, current: Solution, perturbed: Solution, candidate: Solution
    ):
        """
        Registers information between the current, perturbed, and candidate
        solutions.
        """
