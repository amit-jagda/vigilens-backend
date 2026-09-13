from collections import deque, defaultdict
from typing import Optional, Tuple

class MatchTracker:
    def __init__(self, window: int = 5, min_hits: int = 3):
        self.window = window
        self.min_hits = min_hits
        self.history: defaultdict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))

    def add_match(self, track_id: int, employee_id: Optional[int]):
        """Record a candidate employee ID (or None) for the given track."""
        self.history[track_id].append(employee_id)

    def get_consensus(self, track_id: int, min_hits: Optional[int] = None) -> Tuple[Optional[int], float, int]:
        """Return (most_common_id, ratio, hits).
        If total valid hits for the employee is less than min_hits, consensus cannot be reached.
        This prevents immediate false positives on early frames (e.g. 1/1 = 100%).
        """
        required_hits = min_hits if min_hits is not None else self.min_hits
        dq = self.history.get(track_id)
        if not dq or len(dq) < required_hits:
            return None, 0.0, 0
        counts = {}
        for eid in dq:
            if eid is None:
                continue
            counts[eid] = counts.get(eid, 0) + 1
        if not counts:
            return None, 0.0, 0
        most_common_id = max(counts, key=counts.get)
        hits = counts[most_common_id]
        if hits < required_hits:
            return None, 0.0, hits
        ratio = hits / len(dq)
        return most_common_id, ratio, hits
