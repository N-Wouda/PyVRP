from pyvrp.PenaltyManager import PenaltyManager
from pyvrp.ProblemData import ProblemData

class MoveTwoClientsReversed(NodeOperator):
    def __init__(
        self, data: ProblemData, penalty_manager: PenaltyManager
    ) -> None: ...

class NodeOperator:
    def __init__(self, *args, **kwargs) -> None: ...
