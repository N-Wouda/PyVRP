from typing import Protocol


class AcceptanceCriterion(Protocol):
    """
    Protocol describing an acceptance criterion.
    """

    def __call__(
        self,
        best_cost: float,
        current_cost: float,
        candidate_cost: float,
    ) -> bool:
        """
        Determines whether to accept the proposed, candidate solution based on
        this acceptance criterion and the other solution states.

        Parameters
        ----------
        best_cost
            The cost of the best solution observed so far.
        current_cost
            The cost of the current solution.
        candidate_cost
            The cost of the proposed solution.

        Returns
        -------
        bool
            Whether to accept the candidate solution (True), or not (False).
        """
