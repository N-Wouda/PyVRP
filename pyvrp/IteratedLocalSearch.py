from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

if TYPE_CHECKING:
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp._pyvrp import (
        CostEvaluator,
        ProblemData,
        RandomNumberGenerator,
        Solution,
    )
    from pyvrp.accept import AcceptanceCriterion
    from pyvrp.search.SearchMethod import SearchMethod
    from pyvrp.stop.StoppingCriterion import StoppingCriterion


@dataclass
class IteratedLocalSearchParams:
    """
    Parameters for the iterated local search algorithm.

    Parameters
    ----------
    repair_probability
        Probability (in :math:`[0, 1]`) of repairing an infeasible solution.
    nb_iter_no_improvement # TODO
        Number of iterations without any improvement needed before a restart
        occurs.

    Attributes
    ----------
    repair_probability
        Probability of repairing an infeasible solution.
    nb_iter_no_improvement
        Number of iterations without improvement before a restart occurs.

    Raises
    ------
    ValueError
        When ``repair_probability`` is not in :math:`[0, 1]`, or
        ``nb_iter_no_improvement`` is negative.
    """

    repair_probability: float = 0.80
    nb_iter_no_improvement: int = 20_000

    def __post_init__(self):
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be in [0, 1].")


class IteratedLocalSearch:
    """
    Creates an IteratedLocalSearch instance.

    Parameters
    ----------
    data
        The problem data instance.
    penalty_manager
        Penalty manager to use.
    rng
        Random number generator.
    perturb_method
        Perturb method to use.
    search_method
        Search method to use.
    acceptance_criterion
        Acceptance criterion to use.
    params
        Iterated local search parameters to use. If not provided, a default
        will be used.
    """

    def __init__(
        self,
        data: ProblemData,
        penalty_manager: PenaltyManager,
        rng: RandomNumberGenerator,
        perturb_method: SearchMethod,
        search_method: SearchMethod,
        acceptance_criterion: AcceptanceCriterion,
        params: IteratedLocalSearchParams,
    ):
        self._data = data
        self._pm = penalty_manager
        self._rng = rng
        self._perturb = perturb_method
        self._search = search_method
        self._accept = acceptance_criterion
        self._params = params

    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.cost_evaluator()

    def run(
        self,
        stop: StoppingCriterion,
        initial_solution: Solution,
        collect_stats: bool,
        display: bool,
    ) -> Result:
        """
        Runs the iterated local search algorithm.

        Parameters
        ----------
        stop
            Stopping criterion to use. The algorithm runs until the first time
            the stopping criterion returns ``True``.
        initial_solution
            The initial solution to use in the first iteration.
        collect_stats # TODO
            Whether to collect statistics about the solver's progress. Default
            ``True``.
        display # TODO
            Whether to display information about the solver progress. Default
            ``False``. Progress information is only available when
            ``collect_stats`` is also set.
        """
        best = current = initial_solution

        start = time.perf_counter()
        # stats = Statistics(collect_stats=collect_stats) # TODO
        iters = 0
        # iters_no_improvement = 1  # TODO

        while not stop(self._cost_evaluator.cost(best)):
            iters += 1

            perturbed = self._perturb(current, self._cost_evaluator)
            candidate = self._search(perturbed, self._cost_evaluator)

            self._pm.register(candidate)

            cost = self._cost_evaluator.cost(candidate)
            best_cost = self._cost_evaluator.cost(best)

            if cost < best_cost:
                elapsed = round(time.perf_counter() - start, 2)
                print("Direct", cost, elapsed)
                best, current = candidate, candidate
                continue

            # Possibly repair if candidate solution is infeasible. In that
            # case, we penalise infeasibility more using a penalty booster.
            if (
                not candidate.is_feasible()
                and self._rng.rand() < self._params.repair_probability
            ):
                booster_cost_eval = self._pm.booster_cost_evaluator()
                candidate = self._search(candidate, booster_cost_eval)
                self._pm.register(candidate)

                cost = self._cost_evaluator.cost(candidate)
                if cost < best_cost:
                    elapsed = round(time.perf_counter() - start, 2)
                    print("Repair", cost, elapsed)
                    best, current = candidate, candidate
                    continue

        # Accept based on penalised costs.
        current_cost = self._cost_evaluator.penalised_cost(current)
        candidate_cost = self._cost_evaluator.penalised_cost(candidate)

        if self._accept(best_cost, current_cost, candidate_cost):
            current = candidate

        end = time.perf_counter()
        runtime = end - start

        return Result(
            best,
            Statistics(),  # TODO stats # type: ignore
            iters,
            runtime,
        )
