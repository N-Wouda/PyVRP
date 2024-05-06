from typing import Protocol

from pyvrp._pyvrp import CostEvaluator, Solution


class PerturbMethod(Protocol):
    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
    ) -> Solution:
        """
        Perturbs the given solution and returns a new solution.

        Parameters
        ----------
        solution
            Solution to perturb.
        cost_evaluator
            Cost evaluator to use when perturbing.

        Returns
        -------
        Solution
            The perturbed solution.
        """
