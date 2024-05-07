from .RepairOperator import RepairOperator as RepairOperator
from ._repair import greedy_repair as greedy_repair
from ._repair import nearest_route_insert as nearest_route_insert
from .greedy import greedy as greedy
from .nearest import nearest as nearest

REPAIR_OPERATORS: list[RepairOperator] = [
    greedy,
    # nearest,
]
