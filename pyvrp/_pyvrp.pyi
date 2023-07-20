from typing import Callable, Iterator, List, Tuple, Union, overload

class CostEvaluator:
    def __init__(
        self, capacity_penalty: int = 0, tw_penalty: int = 0
    ) -> None: ...
    def load_penalty(self, load: int, vehicle_capacity: int) -> int: ...
    def tw_penalty(self, time_warp: int) -> int: ...
    def penalised_cost(self, solution: Solution) -> int: ...
    def cost(self, solution: Solution) -> int: ...

class DynamicBitset:
    def __init__(self, num_bits: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def count(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> bool: ...
    def __setitem__(self, idx: int, value: bool) -> None: ...
    def __or__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __and__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __xor__(self, other: DynamicBitset) -> DynamicBitset: ...
    def __invert__(self) -> DynamicBitset: ...

class Matrix:
    @overload
    def __init__(self, dimension: int) -> None: ...
    @overload
    def __init__(self, n_rows: int, n_cols: int) -> None: ...
    @overload
    def __init__(self, data: List[List[int]]) -> None: ...
    @property
    def num_cols(self) -> int: ...
    @property
    def num_rows(self) -> int: ...
    def max(self) -> int: ...
    def size(self) -> int: ...
    def __getitem__(self, idx: Tuple[int, int]) -> int: ...
    def __setitem__(self, idx: Tuple[int, int], value: int) -> None: ...

class Client:
    x: int
    y: int
    demand: int
    service_duration: int
    tw_early: int
    tw_late: int
    release_time: int
    prize: int
    required: bool
    def __init__(
        self,
        x: int,
        y: int,
        demand: int = 0,
        service_duration: int = 0,
        tw_early: int = 0,
        tw_late: int = 0,
        release_time: int = 0,
        prize: int = 0,
        required: bool = True,
    ) -> None: ...

class VehicleType:
    capacity: int
    num_available: int
    depot: int
    def __init__(self, capacity: int, num_available: int) -> None: ...

class ProblemData:
    def __init__(
        self,
        clients: List[Client],
        vehicle_types: List[VehicleType],
        distance_matrix: List[List[int]],
        duration_matrix: List[List[int]],
    ): ...
    def client(self, client: int) -> Client: ...
    def centroid(self) -> Tuple[float, float]: ...
    def vehicle_type(self, vehicle_type: int) -> VehicleType: ...
    def dist(self, first: int, second: int) -> int: ...
    def duration(self, first: int, second: int) -> int: ...
    @property
    def num_clients(self) -> int: ...
    @property
    def num_vehicles(self) -> int: ...
    @property
    def num_vehicle_types(self) -> int: ...

class Route:
    def __init__(
        self, data: ProblemData, visits: List[int], vehicle_type: int
    ) -> None: ...
    def __getitem__(self, idx: int) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...
    def is_feasible(self) -> bool: ...
    def has_excess_load(self) -> bool: ...
    def has_time_warp(self) -> bool: ...
    def demand(self) -> int: ...
    def excess_load(self) -> int: ...
    def distance(self) -> int: ...
    def duration(self) -> int: ...
    def visits(self) -> List[int]: ...
    def service_duration(self) -> int: ...
    def time_warp(self) -> int: ...
    def wait_duration(self) -> int: ...
    def release_time(self) -> int: ...
    def prizes(self) -> int: ...
    def centroid(self) -> Tuple[float, float]: ...
    def vehicle_type(self) -> int: ...

class Solution:
    def __init__(
        self,
        data: ProblemData,
        routes: Union[List[Route], List[List[int]]],
    ) -> None: ...
    @classmethod
    def make_random(cls, data: ProblemData, rng: XorShift128) -> Solution:
        """
        Creates a randomly generated solution.

        Parameters
        ----------
        data
            Data instance.
        rng
            Random number generator to use.

        Returns
        -------
        Solution
            The randomly generated solution.
        """
    def get_neighbours(self) -> List[Tuple[int, int]]: ...
    def get_routes(self) -> List[Route]: ...
    def has_excess_load(self) -> bool: ...
    def has_time_warp(self) -> bool: ...
    def distance(self) -> int: ...
    def excess_load(self) -> int: ...
    def time_warp(self) -> int: ...
    def prizes(self) -> int: ...
    def uncollected_prizes(self) -> int: ...
    def is_feasible(self) -> bool: ...
    def num_routes(self) -> int: ...
    def num_clients(self) -> int: ...
    def __copy__(self) -> Solution: ...
    def __deepcopy__(self, memo: dict) -> Solution: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

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
    ) -> None:
        """
        Creates a SubPopulation instance.

        This subpopulation manages a collection of solutions, and initiates
        survivor selection (purging) when their number grows large. A
        subpopulation's solutions can be accessed via indexing and iteration.
        Each solution is stored as a tuple of type ``_Item``, which stores
        the solution itself, a fitness score (higher is worse), and a list
        of proximity values to the other solutions in the subpopulation.

        Parameters
        ----------
        diversity_op
            Operator to use to determine pairwise diversity between solutions.
        params
            Population parameters.
        """
    def add(self, solution: Solution, cost_evaluator: CostEvaluator) -> None:
        """
        Adds the given solution to the subpopulation. Survivor selection is
        automatically triggered when the population reaches its maximum size.

        Parameters
        ----------
        solution
            Solution to add to the subpopulation.
        cost_evaluator
            CostEvaluator to use to compute the cost.
        """
    def purge(self, cost_evaluator: CostEvaluator) -> None:
        """
        Performs survivor selection: solutions in the subpopulation are
        purged until the population is reduced to the ``min_pop_size``.
        Purging happens to duplicate solutions first, and then to solutions
        with high biased fitness.

        Parameters
        ----------
        cost_evaluator
            CostEvaluator to use to compute the cost.
        """
    def update_fitness(self, cost_evaluator: CostEvaluator) -> None:
        """
        Updates the biased fitness scores of solutions in the subpopulation.
        This fitness depends on the quality of the solution (based on its cost)
        and the diversity w.r.t. to other solutions in the subpopulation.

        .. warning::

           This function must be called before accessing the
           :meth:`~SubPopulationItem.fitness` attribute.
        """
    def __getitem__(self, idx: int) -> SubPopulationItem: ...
    def __iter__(self) -> Iterator[SubPopulationItem]: ...
    def __len__(self) -> int: ...

class SubPopulationItem:
    @property
    def fitness(self) -> float:
        """
        Fitness value for this SubPopulationItem.

        Returns
        -------
        float
            Fitness value for this SubPopulationItem.

        .. warning::

           This is a cached property that is not automatically updated. Before
           accessing the property, :meth:`~SubPopulation.update_fitness` should
           be called unless the population has not changed since the last call.
        """
    @property
    def solution(self) -> Solution:
        """
        Solution for this SubPopulationItem.

        Returns
        -------
        Solution
            Solution for this SubPopulationItem.
        """
    def avg_distance_closest(self) -> float:
        """
        Determines the average distance of the solution wrapped by this item
        to a number of solutions that are most similar to it. This provides a
        measure of the relative 'diversity' of the wrapped solution.

        Returns
        -------
        float
            The average distance/diversity of the wrapped solution.
        """

class TimeWindowSegment:
    def __init__(
        self,
        idx_first: int,
        idx_last: int,
        duration: int,
        time_warp: int,
        tw_early: int,
        tw_late: int,
        release_time: int,
    ) -> None:
        """
        Creates a time window segment.

        Parameters
        ----------
        idx_first
            Index of the first client in the route segment.
        idx_last
            Index of the last client in the route segment.
        duration
            Total duration, including waiting time.
        time_warp
            Total time warp on the route segment.
        tw_early
            Earliest visit moment of the first client.
        tw_late
            Latest visit moment of the first client.
        release_time
            Earliest moment to start the route segment.
        """
    @overload
    @staticmethod
    def merge(
        duration_matrix: Matrix,
        first: TimeWindowSegment,
        second: TimeWindowSegment,
    ) -> TimeWindowSegment:
        """
        Merges two time window segments, in order.
        """
    @overload
    @staticmethod
    def merge(
        duration_matrix: Matrix,
        first: TimeWindowSegment,
        second: TimeWindowSegment,
        third: TimeWindowSegment,
    ) -> TimeWindowSegment:
        """
        Merges three time window segments, in order.
        """
    def total_time_warp(self) -> int:
        """
        Returns the total time warp on this route segment.

        Returns
        -------
        int
            Total time warp on this route segment.
        """

class XorShift128:
    def __init__(self, seed: int) -> None: ...
    @staticmethod
    def max() -> int: ...
    @staticmethod
    def min() -> int: ...
    def rand(self) -> float: ...
    def randint(self, high: int) -> int: ...
    def __call__(self) -> int: ...
