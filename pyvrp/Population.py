from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from pyvrp._lib.hgspy import (
    Individual,
    PenaltyManager,
    ProblemData,
    XorShift128,
)


@dataclass
class _Wrapper:
    individual: Individual
    fitness: float

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _Wrapper) and self.individual == other.individual
        )

    def __hash__(self) -> int:
        # Wrapper wraps the *individual*, the fitness score is just metadata.
        # There is one wrapper for each individual (identified by a unique
        # memory location).
        return hash(id(self.individual))

    def __iter__(self):
        return iter((self.individual, self.fitness))

    def __lt__(self, other: _Wrapper):
        return self.fitness < other.fitness

    def cost(self) -> float:
        return self.individual.cost()


SubPop = List[_Wrapper]
DiversityMeasure = Callable[[ProblemData, Individual, Individual], float]
ProxType = Dict[_Wrapper, List[Tuple[float, _Wrapper]]]


@dataclass
class PopulationParams:
    min_pop_size: int = 25
    generation_size: int = 40
    nb_elite: int = 4
    nb_close: int = 5
    lb_diversity: float = 0.1
    ub_diversity: float = 0.5

    def __post_init__(self):
        if not 0 <= self.lb_diversity <= 1.0:
            raise ValueError("lb_diversity must be in [0, 1].")

        if not 0 <= self.ub_diversity <= 1.0:
            raise ValueError("ub_diversity must be in [0, 1].")

        if self.ub_diversity <= self.lb_diversity:
            raise ValueError("ub_diversity <= lb_diversity not understood.")

        if self.nb_elite < 0:
            raise ValueError("nb_elite < 0 not understood.")

        if self.nb_close < 0:
            raise ValueError("nb_close < 0 not understood.")

        if self.min_pop_size < 0:
            # It'll be pretty difficult to do crossover without individuals,
            # but that should not matter much to the Population. Perhaps the
            # user wants to add the pops by themselves somehow.
            raise ValueError("min_pop_size < 0 not understood")

        if self.generation_size < 0:
            raise ValueError("generation_size < 0 not understood.")

    @property
    def max_pop_size(self) -> int:
        return self.min_pop_size + self.generation_size


class Population:
    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: XorShift128,
        diversity_op: DiversityMeasure,
        params: PopulationParams = PopulationParams(),
    ):
        """
        Creates a Population instance.

        Parameters
        ----------
        data
            Data object describing the problem to be solved.
        penalty_manager
            Penalty manager to use.
        rng
            Random number generator.
        diversity_op
            Operator to use to determine pairwise diversity between solutions.
        params, optional
            Population parameters. If not provided, a default will be used.
        """
        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._op = diversity_op
        self._params = params

        self._feas: SubPop = []
        self._infeas: SubPop = []

        self._prox: ProxType = {}
        self._best = Individual(data, penalty_manager, rng)

        for _ in range(params.min_pop_size):
            self.add(Individual(data, penalty_manager, rng))

    @property
    def feasible_subpopulation(self) -> SubPop:
        """
        Returns the feasible subpopulation maintained by this population
        instance.

        Returns
        -------
        SubPop
            Feasible subpopulation.
        """
        return self._feas

    @property
    def infeasible_subpopulation(self) -> SubPop:
        """
        Returns the infeasible subpopulation maintained by this population
        instance.

        Returns
        -------
        SubPop
            Infeasible subpopulation.
        """
        return self._infeas

    @property
    def proximity_structure(self) -> ProxType:
        """
        Returns the proximity structure maintained by this population instance.
        The proximity structure is used for diversity and fitness calculations.

        Returns
        -------
        ProxType
            Proximity structure.
        """
        return self._prox

    def __len__(self) -> int:
        """
        Returns the current population size, that is, the size of its feasible
        and infeasible subpopulations.

        Returns
        -------
        int
            Population size.
        """
        return len(self._feas) + len(self._infeas)

    def add(self, individual: Individual):
        """
        Adds the given individual to the population. Survivor selection is
        automatically triggered when the population reaches its maximum size.

        Parameters
        ----------
        individual
            Individual to add to the population.
        """
        # Copy-construct another individual so we can take ownership.
        individual = Individual(individual)
        is_feasible = individual.is_feasible()
        cost = individual.cost()

        pop = self._feas if is_feasible else self._infeas
        wrapper = _Wrapper(individual, 0.0)

        self._prox[wrapper] = []

        for other in pop:
            dist = self._op(self._data, individual, other.individual)
            bisect.insort_left(self._prox[wrapper], (dist, other))
            bisect.insort_left(self._prox[other], (dist, wrapper))

        pop.append(wrapper)
        self.update_fitness(pop)

        if len(pop) > self._params.max_pop_size:
            self.purge(pop)

        if is_feasible and cost < self._best.cost():
            self._best = individual

    def select(self) -> Tuple[Individual, Individual]:
        """
        Selects two (if possible non-identical) parents by binary tournament,
        subject to a diversity restriction.

        Returns
        -------
        tuple
            A pair of individuals (parents).
        """
        first = self.get_binary_tournament()
        second = self.get_binary_tournament()

        diversity = self._op(self._data, first, second)
        lb = self._params.lb_diversity
        ub = self._params.ub_diversity

        tries = 1
        while not (lb <= diversity <= ub) and tries <= 10:
            tries += 1
            second = self.get_binary_tournament()
            diversity = self._op(self._data, first, second)

        return first, second

    def purge(self, pop: SubPop):
        """
        Performs survivor selection: individuals in the given sub-population
        are purged until the population is reduced to the ``minPopSize``.
        Purging happens to solutions with high biased fitness.

        Parameters
        ----------
        pop
            Sub-population to purge.
        """

        def remove(individual: _Wrapper):
            for _, others in self._prox.items():
                for idx, (_, other) in enumerate(others):
                    if other == individual:
                        del others[idx]
                        break

            del self._prox[individual]
            pop.remove(individual)

        # TODO remove duplicates

        while len(pop) > self._params.min_pop_size:
            self.update_fitness(pop)
            remove(max(pop))

    def update_fitness(self, pop: SubPop):
        by_cost = np.argsort([wrapper.cost() for wrapper in pop])
        diversity = []

        for rank in range(len(pop)):
            wrapper = pop[by_cost[rank]]
            avg_diversity = self.avg_distance_closest(wrapper)
            diversity.append((avg_diversity, rank))

        diversity.sort(reverse=True)
        nb_elite = min(self._params.nb_elite, len(pop))

        for div_rank, (_, cost_rank) in enumerate(diversity):
            div_weight = 1 - nb_elite / len(pop)
            fitness = (cost_rank + div_weight * div_rank) / len(pop)
            pop[cost_rank].fitness = fitness

    def avg_distance_closest(self, wrapper: _Wrapper) -> float:
        """
        Determines the average distance of the given (wrapped) individual to
        a number of individuals that are most similar to it. This provides a
        measure of the relative 'diversity' of this individual.

        Parameters
        ----------
        wrapper
            Wrapped individual whose average distance/diversity to calculate.

        Returns
        -------
        float
            The average distance/diversity of the given individual relative to
            the total population.
        """
        # TODO do we need nb_close? Why not all?
        closest = self._prox[wrapper][: self._params.nb_close]
        return np.mean([div for div, _ in closest]) if closest else 0.0

    def get_binary_tournament(self) -> Individual:
        """
        Select an individual from this population by binary tournament.

        Returns
        -------
        Individual
            The selected individual.
        """

        def select():
            num_feas = len(self._feas)
            idx = self._rng.randint(len(self))

            if idx < num_feas:
                return self._feas[idx]

            return self._infeas[idx - num_feas]

        indiv1, fitness1 = select()
        indiv2, fitness2 = select()

        return indiv1 if fitness1 < fitness2 else indiv2

    def get_best_found(self) -> Individual:
        """
        Returns the best found solution so far. In early iterations, this
        solution might not be feasible yet.

        Returns
        -------
        Individual
            The best solution found so far.
        """
        return self._best
