from .DestroyOperator import DestroyOperator as DestroyOperator
from .concentric import concentric as concentric
from .sequential import sequential as sequential

DESTROY_OPERATORS = [
    concentric,
    sequential,
]
