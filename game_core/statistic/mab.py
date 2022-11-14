from collections import defaultdict
from typing import Dict, Optional, Tuple, List

from game_core.statistic.distribution import DistributionABC, GaussianDistribution, PositiveGaussianDistribution


class MABProblem:
    def __init__(self, arms: Optional[Dict[str, DistributionABC]] = None):
        if arms is None:
            arms = self._get_default_arms()
        self._arms: Dict[str, DistributionABC] = arms
        self._rewards: Dict[str, List[float]] = defaultdict(list)
        self._all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0, reward_squared=0))
        self._cumulative_reward: float = 0
        self.total_actions = 0

    def reset(self):
        self._rewards: Dict[int, List[float]] = defaultdict(list)
        self._all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0, reward_squared=0))
        self._cumulative_reward: float = 0
        self.total_actions = 0

    @staticmethod
    def _get_default_arms():
        g_1 = PositiveGaussianDistribution(5, 1)
        g_2 = PositiveGaussianDistribution(10, 2)
        g_3 = PositiveGaussianDistribution(15, 5)
        g_4 = PositiveGaussianDistribution(20, 10)

        arms = {"1": g_1, "2": g_2, "3": g_3, "4": g_4}

        return arms

    def pull(self, arm_id, save_results=True):
        reward = round(self._arms[arm_id].sample(), 2)

        if save_results:
            self._rewards[arm_id].append(reward)

            self.total_actions += 1
            self._cumulative_reward += reward
            self._all_rewards.append(reward)
            self.record[arm_id]['actions'] += 1
            self.record[arm_id]['reward'] += reward
            self.record[arm_id]['reward_squared'] += reward ** 2

        return reward

    @property
    def arms_ids(self) -> Tuple[str]:
        return tuple(self._arms.keys())

    def display_results(self, playing_window):
        for arm_id in self.arms_ids:
            arm_string_results = [str(r) for r in self._rewards[arm_id]]
            playing_window[f"arm_text{arm_id}"].update("\n".join(arm_string_results))
