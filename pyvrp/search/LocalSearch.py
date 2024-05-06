from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search._search import LocalSearch as _LocalSearch
from pyvrp.search._search import NodeOperator, RouteOperator


class LocalSearch:
    """
    Local search method. This search method explores a granular neighbourhood
    in a very efficient manner using user-provided node and route operators.
    This quickly results in much improved solutions.

    Parameters
    ----------
    data
        Data object describing the problem to be solved.
    rng
        Random number generator.
    neighbours
        List of lists that defines the local search neighbourhood.
    """

    def __init__(
        self,
        data: ProblemData,
        rng: RandomNumberGenerator,
        neighbours: list[list[int]],
    ):
        self._ls = _LocalSearch(data, neighbours)
        self._rng = rng

    def add_node_operator(self, op: NodeOperator):
        """
        Adds a node operator to this local search object. The node operator
        will be used by :meth:`~search` to improve a solution.

        Parameters
        ----------
        op
            The node operator to add to this local search object.
        """
        self._ls.add_node_operator(op)

    def add_route_operator(self, op: RouteOperator):
        """
        Adds a route operator to this local search object. The route operator
        will be used by :meth:`~intensify` to improve a solution using more
        expensive route operators.

        Parameters
        ----------
        op
            The route operator to add to this local search object.
        """
        self._ls.add_route_operator(op)

    def set_neighbours(self, neighbours: list[list[int]]):
        """
        Convenience method to replace the current granular neighbourhood used
        by the local search object.

        Parameters
        ----------
        neighbours
            A new granular neighbourhood.
        """
        self._ls.set_neighbours(neighbours)

    def neighbours(self) -> list[list[int]]:
        """
        Returns the granular neighbourhood currently used by the local search.
        """
        return self._ls.neighbours()

    def __call__(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
    ) -> Solution:
        """
        This method uses the :meth:`~search` and :meth:`~intensify` methods to
        iteratively improve the given solution. First, :meth:`~search` is
        applied. Thereafter, :meth:`~intensify` is applied. This repeats until
        no further improvements are found. Finally, the improved solution is
        returned.

        Parameters
        ----------
        solution
            The solution to improve through local search.
        cost_evaluator
            Cost evaluator to use.

        Returns
        -------
        Solution
            The improved solution. This is not the same object as the
            solution that was passed in.
        """
        self._ls.shuffle(self._rng)
        return self._ls(solution, cost_evaluator)

    def intensify(
        self,
        solution: Solution,
        cost_evaluator: CostEvaluator,
        overlap_tolerance: float = 0.05,
    ) -> Solution:
        """
        This method uses the intensifying route operators on this local search
        object to improve the given solution. To limit the computational
        demands of intensification, the  ``overlap_tolerance`` argument
        can be used to limit the number of route pairs that are evaluated.

        Parameters
        ----------
        solution
            The solution to improve.
        cost_evaluator
            Cost evaluator to use.
        overlap_tolerance
            This method evaluates improving moves between route pairs. To limit
            computational efforts, by default not all route pairs are
            considered: only those route pairs that share some overlap when
            considering their center's angle to the center of all clients.
            This parameter controls the amount of overlap needed before two
            routes are evaluated.

        Returns
        -------
        Solution
            The improved solution. This is not the same object as the
            solution that was passed in.
        """
        self._ls.shuffle(self._rng)
        return self._ls.intensify(solution, cost_evaluator, overlap_tolerance)

    def search(
        self, solution: Solution, cost_evaluator: CostEvaluator
    ) -> Solution:
        """
        This method uses the node operators on this local search object to
        improve the given solution.

        Parameters
        ----------
        solution
            The solution to improve.
        cost_evaluator
            Cost evaluator to use.

        Returns
        -------
        Solution
            The improved solution. This is not the same object as the
            solution that was passed in.
        """
        self._ls.shuffle(self._rng)
        return self._ls.search(solution, cost_evaluator)
