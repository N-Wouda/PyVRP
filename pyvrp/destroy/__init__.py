from .DestroyOperator import DestroyOperator as DestroyOperator
from .concentric import concentric as concentric
from .random import random as random
from .sequential import sequential as sequential

DESTROY_OPERATORS: list[DestroyOperator] = [
    concentric,
    random,
    sequential,
]
