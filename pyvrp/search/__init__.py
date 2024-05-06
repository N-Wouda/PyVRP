from typing import Type

from .DestroyRepair import DestroyRepair as DestroyRepair
from .LocalSearch import LocalSearch as LocalSearch
from .SearchMethod import SearchMethod as SearchMethod
from ._search import Exchange10 as Exchange10
from ._search import Exchange11 as Exchange11
from ._search import Exchange20 as Exchange20
from ._search import Exchange21 as Exchange21
from ._search import Exchange22 as Exchange22
from ._search import Exchange30 as Exchange30
from ._search import Exchange31 as Exchange31
from ._search import Exchange32 as Exchange32
from ._search import Exchange33 as Exchange33
from ._search import NodeOperator as NodeOperator
from ._search import RouteOperator as RouteOperator
from ._search import SwapRoutes as SwapRoutes
from ._search import SwapStar as SwapStar
from ._search import SwapTails as SwapTails
from .neighbourhood import NeighbourhoodParams as NeighbourhoodParams
from .neighbourhood import compute_neighbours as compute_neighbours

NODE_OPERATORS: list[Type[NodeOperator]] = [
    Exchange10,
    Exchange20,
    Exchange30,
    Exchange11,
    Exchange21,
    Exchange31,
    Exchange22,
    Exchange32,
    Exchange33,
    SwapTails,
]

ROUTE_OPERATORS: list[Type[RouteOperator]] = [
    SwapRoutes,
    SwapStar,
]
