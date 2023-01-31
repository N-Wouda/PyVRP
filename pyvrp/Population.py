from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, List, Tuple

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
        return hash(self.individual)

    def __iter__(self):
        return iter((self.individual, self.fitness))

    def __lt__(self, other: _Wrapper):
        return self.fitness < other.fitness

    def cost(self) -> float:
        return self.individual.cost()


_SubPop = List[_Wrapper]
_DiversityMeasure = Callable[[ProblemData, Individual, Individual], float]
_ProxType = DefaultDict[_Wrapper, List[Tuple[float, _Wrapper]]]


class Population:
    class Params:
        min_pop_size = 25
        generation_size = 40
        nb_elite = 4
        nb_close = 5
        lb_diversity = 0.1
        ub_diversity = 0.5

        # TODO parameter validation

        @property
        def max_pop_size(self) -> int:
            return self.min_pop_size + self.generation_size

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: XorShift128,
        diversity_op: _DiversityMeasure,
        params: Params = Params(),
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

        self._feas: _SubPop = []
        self._infeas: _SubPop = []

        self._prox: _ProxType = defaultdict(list)

        self._best = Individual(data, penalty_manager, rng)

        for _ in range(params.min_pop_size):
            self.add(Individual(data, penalty_manager, rng))

    def add(self, individual: Individual):
        """
        Adds the given individual to the population. Survivor selection is
        automatically triggered when the population reaches its maximum size.

        Parameters
        ----------
        individual
            Individual to add to the population.
        """
        is_feasible = individual.is_feasible()
        cost = individual.cost()

        pop = self._feas if is_feasible else self._infeas
        wrapper = _Wrapper(individual, 0.0)

        for other in pop:
            dist = self._op(self._data, individual, other.individual)
            bisect.insort_left(self._prox[wrapper], (dist, other))
            bisect.insort_left(self._prox[other], (dist, wrapper))

        pop.append(wrapper)
        self.update_fitness(pop)

        if len(pop) > self._params.max_pop_size:
            self.purge(pop)

        if is_feasible and cost < self.get_best_found().cost():
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

    def purge(self, pop: _SubPop):
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

        # TODO remove duplicates?

        while len(pop) > self._params.min_pop_size:
            self.update_fitness(pop)
            remove(max(pop))

    def update_fitness(self, pop: _SubPop):
        by_cost = np.argsort([wrapper.cost() for wrapper in pop])
        diversity = []

        for rank in range(len(pop)):
            individual = pop[by_cost[rank]]
            closest = self._prox[individual][: self._params.nb_close]

            if closest:
                dist = np.mean([div for div, _ in closest])
                diversity.append((dist, rank))
            else:
                diversity.append((0, rank))

        diversity.sort(reverse=True)
        nb_elite = min(self._params.nb_elite, len(pop))

        for div_rank, (_, cost_rank) in enumerate(diversity):
            div_weight = 1 - nb_elite / len(pop)
            fitness = (cost_rank + div_weight * div_rank) / len(pop)
            pop[cost_rank].fitness = fitness

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
            num_infeas = len(self._infeas)
            idx = self._rng.randint(num_feas + num_infeas)

            if idx < len(self._feas):
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
