from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from warnings import warn

import numpy as np

from pyvrp._pyvrp import CostEvaluator, ProblemData, Solution
from pyvrp.exceptions import PenaltyBoundWarning


@dataclass
class PenaltyParams:
    """
    The penalty manager parameters.

    Parameters
    ----------
    repair_booster
        A repair booster value :math:`r \\ge 1`. This value is used to
        temporarily multiply the current penalty terms, to force feasibility.
        See also
        :meth:`~pyvrp.PenaltyManager.PenaltyManager.booster_cost_evaluator`.
    solutions_between_updates
        Number of feasibility registrations between penalty value updates. The
        penalty manager updates the penalty terms every once in a while based
        on recent feasibility registrations. This parameter controls how often
        such updating occurs.
    penalty_increase
        Amount :math:`p_i \\ge 1` by which the current penalties are
        increased when insufficient feasible solutions (see
        ``target_feasible``) have been found amongst the most recent
        registrations. The penalty values :math:`v` are updated as
        :math:`v \\gets p_i v`.
    penalty_decrease
        Amount :math:`p_d \\in [0, 1]` by which the current penalties are
        decreased when sufficient feasible solutions (see ``target_feasible``)
        have been found amongst the most recent registrations. The penalty
        values :math:`v` are updated as :math:`v \\gets p_d v`.
    target_feasible
        Target percentage :math:`p_f \\in [0, 1]` of feasible registrations
        in the last ``solutions_between_updates`` registrations. This
        percentage is used to update the penalty terms: when insufficient
        feasible solutions have been registered, the penalties are increased;
        similarly, when too many feasible solutions have been registered, the
        penalty terms are decreased. This ensures a balanced population, with a
        fraction :math:`p_f` feasible and a fraction :math:`1 - p_f` infeasible
        solutions.

    Attributes
    ----------
    repair_booster
        A repair booster value.
    solutions_between_updates
        Number of feasibility registrations between penalty value updates.
    penalty_increase
        Amount :math:`p_i \\ge 1` by which the current penalties are
        increased when insufficient feasible solutions (see
        ``target_feasible``) have been found amongst the most recent
        registrations.
    penalty_decrease
        Amount :math:`p_d \\in [0, 1]` by which the current penalties are
        decreased when sufficient feasible solutions (see ``target_feasible``)
        have been found amongst the most recent registrations.
    target_feasible
        Target percentage :math:`p_f \\in [0, 1]` of feasible registrations
        in the last ``solutions_between_updates`` registrations.
    """

    repair_booster: int = 12
    solutions_between_updates: int = 50
    penalty_increase: float = 1.34
    penalty_decrease: float = 0.32
    target_feasible: float = 0.43

    def __post_init__(self):
        if not self.repair_booster >= 1:
            raise ValueError("Expected repair_booster >= 1.")

        if not self.solutions_between_updates >= 1:
            raise ValueError("Expected solutions_between_updates >= 1.")

        if not self.penalty_increase >= 1.0:
            raise ValueError("Expected penalty_increase >= 1.")

        if not (0.0 <= self.penalty_decrease <= 1.0):
            raise ValueError("Expected penalty_decrease in [0, 1].")

        if not (0.0 <= self.target_feasible <= 1.0):
            raise ValueError("Expected target_feasible in [0, 1].")


class PenaltyManager:
    """
    Creates a PenaltyManager instance.

    This class manages time warp and load penalties, and provides penalty terms
    for given time warp and load values. It updates these penalties based on
    recent history, and can be used to provide a temporary penalty booster
    object that increases the penalties for a short duration.

    .. note::

       Consider initialising using :meth:`~init_from` to compute initial
       penalty values that are scaled according to the data instance.

    Parameters
    ----------
    params
        PenaltyManager parameters. If not provided, a default will be used.
    initial_penalties
        Initial penalty values for unit load (idx 0), duration (1), and
        distance (2) violations. Defaults to ``(20, 6, 6)`` for backwards
        compatibility. These values are clipped to the range ``[MIN_PENALTY,
        MAX_PENALTY]``.
    """

    MIN_PENALTY = 1
    MAX_PENALTY = 100_000
    FEAS_TOL = 0.05

    def __init__(
        self,
        params: PenaltyParams = PenaltyParams(),
        initial_penalties: tuple[int, int, int] = (20, 6, 6),
    ):
        self._params = params
        self._penalties = np.clip(
            initial_penalties,
            self.MIN_PENALTY,
            self.MAX_PENALTY,
        )

        self._feas_lists: list[list[bool]] = [
            [],  # tracks recent load feasibility
            [],  # track recent time feasibility
            [],  # track recent distance feasibility
        ]

    @classmethod
    def init_from(
        cls,
        data: ProblemData,
        params: PenaltyParams = PenaltyParams(),
    ) -> PenaltyManager:
        """
        Initialises from the given data instance and parameter object. The
        initial penalty values are computed from the problem data.

        Parameters
        ----------
        data
            Data instance to use when computing penalty values.
        params
            PenaltyManager parameters. If not provided, a default will be used.
        """
        distances = data.distance_matrices()
        durations = data.duration_matrices()
        edge_costs = [  # edge costs per vehicle type
            veh_type.unit_distance_cost * distances[veh_type.profile]
            + veh_type.unit_duration_cost * durations[veh_type.profile]
            for veh_type in data.vehicle_types()
        ]

        # Best edge cost/distance/duration over all vehicle types and profiles,
        # and then average that for the entire matrix to obtain an "average
        # best" edge cost/distance/duration.
        avg_cost = np.minimum.reduce(edge_costs).mean()
        avg_distance = np.minimum.reduce(distances).mean()
        avg_duration = np.minimum.reduce(durations).mean()

        avg_load = 0
        if data.num_clients != 0:
            pickups = np.array([c.pickup for c in data.clients()])
            deliveries = np.array([c.delivery for c in data.clients()])
            avg_load = np.maximum(pickups, deliveries).mean()

        # Initial penalty parameters are meant to weigh an average increase
        # in the relevant value by the same amount as the average edge cost.
        init_load = round(avg_cost / max(avg_load, 1))
        init_tw = round(avg_cost / max(avg_duration, 1))
        init_dist = round(avg_cost / max(avg_distance, 1))
        return cls(params, (init_load, init_tw, init_dist))

    def _compute(self, penalty: int, feas_percentage: float) -> int:
        # Computes and returns the new penalty value, given the current value
        # and the percentage of feasible solutions since the last update.
        diff = self._params.target_feasible - feas_percentage

        if abs(diff) < self.FEAS_TOL:
            return penalty

        # +/- 1 to ensure we do not get stuck at the same integer values.
        if diff > 0:
            new_penalty = self._params.penalty_increase * penalty + 1
        else:
            new_penalty = self._params.penalty_decrease * penalty - 1

        clipped = int(np.clip(new_penalty, self.MIN_PENALTY, self.MAX_PENALTY))

        if clipped == self.MAX_PENALTY:
            msg = """
            A penalty parameter has reached its maximum value. This means PyVRP
            struggles to find a feasible solution for the instance that's being
            solved, either because the instance has no feasible solution, or it
            is very hard to find one. Check the instance carefully to determine
            if a feasible solution exists.
            """
            warn(msg, PenaltyBoundWarning)

        return clipped

    def _register(self, feas_list: list[bool], penalty: int, is_feas: bool):
        feas_list.append(is_feas)

        if len(feas_list) != self._params.solutions_between_updates:
            return penalty

        avg = fmean(feas_list)
        feas_list.clear()
        return self._compute(penalty, avg)

    def register(self, sol: Solution):
        """
        Registers the feasibility dimensions of the given solution.
        """
        args = [
            not sol.has_excess_load(),
            not sol.has_time_warp(),
            not sol.has_excess_distance(),
        ]

        for idx, is_feas in enumerate(args):
            feas_list = self._feas_lists[idx]
            penalty = self._penalties[idx]
            self._penalties[idx] = self._register(feas_list, penalty, is_feas)

    def cost_evaluator(self) -> CostEvaluator:
        """
        Get a cost evaluator using the current penalty values.
        """
        return CostEvaluator(*self._penalties)

    def booster_cost_evaluator(self) -> CostEvaluator:
        """
        Get a cost evaluator using the boosted current penalty values.
        """
        return CostEvaluator(*(self._penalties * self._params.repair_booster))
