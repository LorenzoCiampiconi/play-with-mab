from collections import defaultdict
from typing import Dict

from distribution import DistributionABC


class MabProblem:
    def __init__(self, arms: Dict[int, DistributionABC]):
        self._arms = arms
        self._results = defaultdict(list)

    def pull(self, arm_id, save_results=True):
        result = self._arms[arm_id].sample()

        if save_results:
            self._results[arm_id].append(result)

        return result
