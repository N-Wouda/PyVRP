import time


class RecordToRecord:
    def __init__(self, start_pct: float, end_pct: float, max_runtime: float):
        self._start_pct = start_pct
        self._end_pct = end_pct
        self._max_runtime = max_runtime

        self._delta_pct = start_pct - end_pct
        self._start_time = time.perf_counter()

    @property
    def start_pct(self) -> float:
        return self._start_pct

    @property
    def end_pct(self) -> float:
        return self._end_pct

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    def __call__(self, best: float, curr: float, cand: float):
        elapsed = (time.perf_counter() - self._start_time) / self.max_runtime
        threshold = (self._delta_pct * (1 - elapsed) + self.end_pct) * best
        threshold = max(0, threshold)  # Ensure threshold is non-negative

        return cand - best <= threshold
