from collections import deque
from statistics import mean
from typing import Deque, List


class MovingAverageThreshold:
    """
    The Adaptive Threshold (AT) criterion of [1]. This criterion accepts a
    candidate solution if it is better than an adaptive threshold value. The
    adaptive threshold is computed as:

    .. math::

       f(s^*) + \\eta \\left(
          \\sum_{i = 1}^\\gamma \\frac{f(s^i)}{\\gamma} - f(s^*)
       \\right)

    where :math:`s^*` is the best solution observed in the last :math:`\\gamma`
    iterations, :math:`f(\\cdot)` indicates the objective function,
    :math:`\\eta \\in [0, 1]` and :math:`\\gamma \\in \\mathbb{N}` are
    parameters, and each :math:`s^i` is a recently observed solution. The
    recently observed solutions are stored in a ``history`` attributed of size
    at most :math:`\\gamma`.

    Parameters
    ----------
    eta: float
        Used to determine the threshold value. Larger values of :math:`\\eta`
        result in more accepted candidate solutions. Must be in [0, 1].
    gamma: int
        History size. Must be positive.

    References
    ----------
    .. [1] Máximo, V.R. and M.C.V. Nascimento. 2021. A hybrid adaptive iterated
           local search with diversification control to the capacitated vehicle
           routing problem, *European Journal of Operational Research* 294 (3):
           1108 - 1119.
    """

    def __init__(self, eta: float, gamma: int):
        if not (0 <= eta <= 1):
            raise ValueError("eta must be in [0, 1].")

        if gamma <= 0:
            raise ValueError("gamma must be positive.")

        self._eta = eta
        self._gamma = gamma
        self._history: Deque[float] = deque(maxlen=gamma)

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def gamma(self) -> int:
        return self._gamma

    @property
    def history(self) -> List[float]:
        return list(self._history)

    def __call__(self, best, current, candidate) -> bool:
        self._history.append(candidate)
        recent_best = min(self._history)
        recent_avg = mean(self._history)

        threshold = recent_best + self._eta * (recent_avg - recent_best)
        return candidate <= threshold
