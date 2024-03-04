from typing import Callable, Iterator, Optional, Union, overload

import numpy as np

class CostEvaluator:
    def __init__(
        self, capacity_penalty: int = 0, tw_penalty: int = 0
    ) -> None: ...
    def load_penalty(self, load: int, capacity: int) -> int: ...
    def tw_penalty(self, time_warp: int) -> int: ...
    def penalised_cost(self, solution: Solution) -> int: ...
    def cost(self, solution: Solution) -> int: ...

class DynamicBitset:
    def __init__(self, num_bits: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __getitem__(self, idx: int) -> bool: ...
    def __setitem__(self, idx: int, value: bool) -> None: ...
    def all(self) -> bool: ...
    def any(self) -> bool: ...
    def none(self) -> bool: ...
    def count(self) -> int: ...
    def __len__(self) -> int: ...
    def __or__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __and__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __xor__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __invert__(self) -> DynamicBitset: ...
    def reset(self) -> DynamicBitset: ...

class Client:
    x: int
    y: int
    delivery: int
    pickup: int
    service_duration: int
    tw_early: int
    tw_late: int
    release_time: int
    prize: int
    required: bool
    group: Optional[int]
    name: str
    def __init__(
        self,
        x: int,
        y: int,
        delivery: int = 0,
        pickup: int = 0,
        service_duration: int = 0,
        tw_early: int = 0,
        tw_late: int = ...,
        release_time: int = 0,
        prize: int = 0,
        required: bool = True,
        group: Optional[int] = None,
        name: str = "",
    ) -> None: ...

class Depot:
    x: int
    y: int
    tw_early: int
    tw_late: int
    name: str
    def __init__(
        self,
        x: int,
        y: int,
        tw_early: int = 0,
        tw_late: int = ...,
        name: str = "",
    ) -> None: ...

class VehicleType:
    num_available: int
    depot: int
    capacity: int
    fixed_cost: int
    tw_early: int
    tw_late: int
    max_duration: int
    name: str
    def __init__(
        self,
        num_available: int = 1,
        capacity: int = 0,
        depot: int = 0,
        fixed_cost: int = 0,
        tw_early: int = 0,
        tw_late: int = ...,
        max_duration: int = ...,
        name: str = "",
    ) -> None: ...

class ProblemData:
    def __init__(
        self,
        clients: list[Client],
        depots: list[Depot],
        vehicle_types: list[VehicleType],
        distance_matrix: np.ndarray[int],
        duration_matrix: np.ndarray[int],
    ) -> None: ...
    def location(self, idx: int) -> Union[Client, Depot]: ...
    def clients(self) -> list[Client]: ...
    def depots(self) -> list[Depot]: ...
    def vehicle_types(self) -> list[VehicleType]: ...
    def replace(
        self,
        clients: Optional[list[Client]] = None,
        depots: Optional[list[Depot]] = None,
        vehicle_types: Optional[list[VehicleType]] = None,
        distance_matrix: Optional[np.ndarray[int]] = None,
        duration_matrix: Optional[np.ndarray[int]] = None,
    ) -> ProblemData: ...
    def centroid(self) -> tuple[float, float]: ...
    def vehicle_type(self, vehicle_type: int) -> VehicleType: ...
    def dist(self, first: int, second: int) -> int: ...
    def duration(self, first: int, second: int) -> int: ...
    def distance_matrix(self) -> np.ndarray[int]: ...
    def duration_matrix(self) -> np.ndarray[int]: ...
    @property
    def num_clients(self) -> int: ...
    @property
    def num_depots(self) -> int: ...
    @property
    def num_locations(self) -> int: ...
    @property
    def num_vehicles(self) -> int: ...
    @property
    def num_vehicle_types(self) -> int: ...

class Route:
    def __init__(
        self, data: ProblemData, visits: list[int], vehicle_type: int
    ) -> None: ...
    def __getitem__(self, idx: int) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def is_feasible(self) -> bool: ...
    def has_excess_load(self) -> bool: ...
    def has_time_warp(self) -> bool: ...
    def delivery(self) -> int: ...
    def pickup(self) -> int: ...
    def excess_load(self) -> int: ...
    def distance(self) -> int: ...
    def duration(self) -> int: ...
    def visits(self) -> list[int]: ...
    def time_warp(self) -> int: ...
    def start_time(self) -> int: ...
    def end_time(self) -> int: ...
    def slack(self) -> int: ...
    def service_duration(self) -> int: ...
    def travel_duration(self) -> int: ...
    def wait_duration(self) -> int: ...
    def release_time(self) -> int: ...
    def prizes(self) -> int: ...
    def centroid(self) -> tuple[float, float]: ...
    def vehicle_type(self) -> int: ...
    def depot(self) -> int: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple, /) -> None: ...

class Solution:
    def __init__(
        self,
        data: ProblemData,
        routes: Union[list[Route], list[list[int]]],
    ) -> None: ...
    @classmethod
    def make_random(
        cls, data: ProblemData, rng: RandomNumberGenerator
    ) -> Solution: ...
    def neighbours(self) -> list[Optional[tuple[int, int]]]: ...
    def routes(self) -> list[Route]: ...
    def has_excess_load(self) -> bool: ...
    def has_time_warp(self) -> bool: ...
    def distance(self) -> int: ...
    def excess_load(self) -> int: ...
    def fixed_vehicle_cost(self) -> int: ...
    def time_warp(self) -> int: ...
    def prizes(self) -> int: ...
    def uncollected_prizes(self) -> int: ...
    def is_feasible(self) -> bool: ...
    def is_complete(self) -> bool: ...
    def num_routes(self) -> int: ...
    def num_clients(self) -> int: ...
    def num_missing_clients(self) -> int: ...
    def __copy__(self) -> Solution: ...
    def __deepcopy__(self, memo: dict) -> Solution: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple, /) -> None: ...

class PopulationParams:
    generation_size: int
    lb_diversity: float
    min_pop_size: int
    nb_close: int
    nb_elite: int
    ub_diversity: float
    def __init__(
        self,
        min_pop_size: int = 25,
        generation_size: int = 40,
        nb_elite: int = 4,
        nb_close: int = 5,
        lb_diversity: float = 0.1,
        ub_diversity: float = 0.5,
    ) -> None: ...
    @property
    def max_pop_size(self) -> int: ...

class SubPopulation:
    def __init__(
        self,
        diversity_op: Callable[[Solution, Solution], float],
        params: PopulationParams,
    ) -> None: ...
    def add(
        self, solution: Solution, cost_evaluator: CostEvaluator
    ) -> None: ...
    def purge(self, cost_evaluator: CostEvaluator) -> None: ...
    def update_fitness(self, cost_evaluator: CostEvaluator) -> None: ...
    def __getitem__(self, idx: int) -> SubPopulationItem: ...
    def __iter__(self) -> Iterator[SubPopulationItem]: ...
    def __len__(self) -> int: ...

class SubPopulationItem:
    @property
    def fitness(self) -> float: ...
    @property
    def solution(self) -> Solution: ...
    def avg_distance_closest(self) -> float: ...

class DistanceSegment:
    def __init__(
        self,
        idx_first: int,
        idx_last: int,
        distance: int,
    ) -> None: ...
    @overload
    @staticmethod
    def merge(
        distance_matrix: np.ndarray[int],
        first: DistanceSegment,
        second: DistanceSegment,
    ) -> DistanceSegment: ...
    @overload
    @staticmethod
    def merge(
        distance_matrix: np.ndarray[int],
        first: DistanceSegment,
        second: DistanceSegment,
        third: DistanceSegment,
    ) -> DistanceSegment: ...
    def distance(self) -> int: ...

class LoadSegment:
    def __init__(
        self,
        delivery: int,
        pickup: int,
        load: int,
    ) -> None: ...
    @overload
    @staticmethod
    def merge(
        first: LoadSegment,
        second: LoadSegment,
    ) -> LoadSegment: ...
    @overload
    @staticmethod
    def merge(
        first: LoadSegment,
        second: LoadSegment,
        third: LoadSegment,
    ) -> LoadSegment: ...
    def delivery(self) -> int: ...
    def pickup(self) -> int: ...
    def load(self) -> int: ...

class DurationSegment:
    def __init__(
        self,
        idx_first: int,
        idx_last: int,
        duration: int,
        time_warp: int,
        tw_early: int,
        tw_late: int,
        release_time: int,
    ) -> None: ...
    @overload
    @staticmethod
    def merge(
        duration_matrix: np.ndarray[int],
        first: DurationSegment,
        second: DurationSegment,
    ) -> DurationSegment: ...
    @overload
    @staticmethod
    def merge(
        duration_matrix: np.ndarray[int],
        first: DurationSegment,
        second: DurationSegment,
        third: DurationSegment,
    ) -> DurationSegment: ...
    def duration(self) -> int: ...
    def tw_early(self) -> int: ...
    def tw_late(self) -> int: ...
    def time_warp(self, max_duration: int = ...) -> int: ...

class RandomNumberGenerator:
    @overload
    def __init__(self, seed: int) -> None: ...
    @overload
    def __init__(self, state: list[int]) -> None: ...
    @staticmethod
    def max() -> int: ...
    @staticmethod
    def min() -> int: ...
    def rand(self) -> float: ...
    def randint(self, high: int) -> int: ...
    def __call__(self) -> int: ...
    def state(self) -> list[int]: ...
