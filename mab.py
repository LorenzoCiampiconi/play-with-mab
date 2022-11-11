from collections import defaultdict
from typing import Dict, Optional

from distribution import DistributionABC, GaussianDistribution


class MabProblem:
    def __init__(self, arms: Optional[Dict[int, DistributionABC]] = None):
        if arms is None:
            arms = self._get_default_arms()
        self._arms: Dict[int, DistributionABC] = arms
        self._results = defaultdict(list)

    @staticmethod
    def _get_default_arms():
        g_1 = GaussianDistribution(0, 1)
        g_2 = GaussianDistribution(0.5, 2)
        g_3 = GaussianDistribution(4, 10)
        g_4 = GaussianDistribution(3, 5)

        arms = {1: g_1, 2: g_2, 3: g_3, 4: g_4}

        return arms

    def pull(self, arm_id, save_results=True):
        result = round(self._arms[arm_id].sample(),2)

        if save_results:
            self._results[arm_id].append(result)

        return result

    def display_results(self):
        print(dict(self._results))
