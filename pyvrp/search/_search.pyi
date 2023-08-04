from typing import Iterator, List, Optional

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)

class NodeOperator:
    def __init__(self, *args, **kwargs) -> None: ...

class RouteOperator:
    def __init__(self, *args, **kwargs) -> None: ...

class Exchange10(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange11(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange20(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange21(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange22(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange30(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange31(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange32(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Exchange33(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class LocalSearch:
    def __init__(
        self,
        data: ProblemData,
        neighbours: List[List[int]],
    ) -> None: ...
    def add_node_operator(self, op: NodeOperator) -> None: ...
    def add_route_operator(self, op: RouteOperator) -> None: ...
    def set_neighbours(self, neighbours: List[List[int]]) -> None: ...
    def get_neighbours(self) -> List[List[int]]: ...
    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
    ) -> Solution: ...
    def shuffle(self, rng: RandomNumberGenerator) -> None: ...
    def intensify(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        overlap_tolerance: float = 0.05,
    ) -> Solution: ...
    def search(
        self, solution: Solution, cost_evaluator: CostEvaluator
    ) -> Solution: ...

class MoveTwoClientsReversed(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class RelocateStar(RouteOperator):
    def __init__(self, data: ProblemData) -> None: ...

class SwapStar(RouteOperator):
    def __init__(self, data: ProblemData) -> None: ...

class TwoOpt(NodeOperator):
    def __init__(self, data: ProblemData) -> None: ...

class Route:
    def __init__(
        self, data: ProblemData, idx: int, vehicle_type: int
    ) -> None: ...
    @property
    def idx(self) -> int: ...
    @property
    def vehicle_type(self) -> int: ...
    def __getitem__(self, idx: int) -> Node: ...
    def __iter__(self) -> Iterator[Node]: ...
    def __len__(self) -> int: ...
    def append(self, node: Node) -> None: ...
    def insert(self, idx: int, node: Node) -> None: ...
    def update(self) -> None: ...

class Node:
    def __init__(self, loc: int) -> None: ...
    @property
    def client(self) -> int: ...
    @property
    def idx(self) -> int: ...
    @property
    def route(self) -> Optional[Route]: ...
    def is_depot(self) -> bool: ...
