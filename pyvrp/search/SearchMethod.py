from typing import Optional, Protocol

from pyvrp._pyvrp import CostEvaluator, Solution


class SearchMethod(Protocol):  # pragma: no cover
    """
    Protocol that search methods must implement.
    """

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        neighbours: Optional[list[list[int]]] = None,
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
