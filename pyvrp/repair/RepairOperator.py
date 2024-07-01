from typing import Protocol

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)


class RepairOperator(Protocol):
    def __call__(
        self,
        data: ProblemData,
        solution: Solution,
        cost_eval: CostEvaluator,
        rng: RandomNumberGenerator,
    ) -> Solution:
        """
        TODO
        """
