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
        self._data = data
        self._rng = rng
        self._destroy_ops = destroy_ops
        self._repair_ops = repair_ops

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        neighbours: Optional[list[list[int]]] = None,
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
        neighbours
            New granular neighbourhood if passed.

        Returns
        -------
        Solution
            The improved solution. This is not the same object as the
            solution that was passed in.
        """
        d_idx = self._rng.randint(len(self._destroy_ops))
        r_idx = self._rng.randint(len(self._repair_ops))

        destroy_op = self._destroy_ops[d_idx]
        repair_op = self._repair_ops[r_idx]

        destroyed = destroy_op(self._data, solution, cost_evaluator, self._rng)
        return repair_op(self._data, destroyed, cost_evaluator, self._rng)

    def register(
        self, current: Solution, perturbed: Solution, candidate: Solution
    ):
        pass
