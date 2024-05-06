from typing import Optional

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)


class DestroyRepair:
    def __init__(
        self,
        data: ProblemData,
        rng: RandomNumberGenerator,
        destroy_ops: list,
        repair_ops: list,
    ):
        pass

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        candidates: Optional[list[int]] = None,
    ) -> Solution:
        """
        This method uses the :meth:`~search` and :meth:`~intensify` methods to
        iteratively improve the given solution. First, :meth:`~search` is
        applied. Thereafter, :meth:`~intensify` is applied. This repeats until
        no further improvements are found. Finally, the improved solution is
        returned.

        Parameters
        ----------
        solution
            The solution to improve through local search.
        cost_evaluator
            Cost evaluator to use.
        candidates
            TODO

        Returns
        -------
        Solution
            The improved solution. This is not the same object as the
            solution that was passed in.
        """
        pass

    def register(
        self, current: Solution, perturbed: Solution, candidate: Solution
    ):
        pass
