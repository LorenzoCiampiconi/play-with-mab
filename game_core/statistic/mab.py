from collections import defaultdict
from typing import Dict, Optional, Tuple, List

from game_core.statistic.distribution import (
    DistributionABC,
    GaussianDistribution,
    PositiveGaussianDistribution,
    BernoulliDistribution,
)


class MABProblem:
    def __init__(self, arms: Optional[Dict[str, DistributionABC]] = None):
        if arms is None:
            arms = self._get_default_arms()
        self._arms: Dict[str, DistributionABC] = arms
        self.__reset__()

    def __reset__(self):
        self._rewards: Dict[int, List[float]] = defaultdict(list)
        self._all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0, reward_squared=0))
        self._cumulative_reward: float = 0
        self.total_actions = 0
        self._history_of_play = []
        self._history_of_cumulative_reward = []
        self._history_of_cumulative_reward_by_id = defaultdict(lambda: [])

    def reset(self):
        self.__reset__()

    @staticmethod
    def _get_default_arms():
        b_1 = BernoulliDistribution(0.2, 99999)
        b_2 = BernoulliDistribution(0.4, 87)
        b_3 = BernoulliDistribution(0.8, 100)

        arms = {
            "1": b_1,
            "2": b_2,
            "3": b_3,
        }

        return arms

    def history_of_cumulative_reward(self):
        return self._history_of_cumulative_reward

    def history_of_cumulative_reward_by_id(self):
        return self._history_of_cumulative_reward_by_id

    def pull(self, arm_id, save_results=True):
        reward = round(self._arms[arm_id].sample(), 2)

        if save_results:
            self._rewards[arm_id].append(reward)

            self.total_actions += 1
            self._cumulative_reward += reward
            self._all_rewards.append(reward)
            self.record[arm_id]["actions"] += 1
            self.record[arm_id]["reward"] += reward
            self.record[arm_id]["reward_squared"] += reward**2
            self._history_of_play.append(arm_id)
            self._history_of_cumulative_reward.append(self._cumulative_reward)
            for arm in self.arms_ids:
                self._history_of_cumulative_reward_by_id[arm].append(self.record[arm]["reward"])
            print(self._history_of_cumulative_reward_by_id.items())

        return reward

    @property
    def arms_ids(self) -> Tuple[str]:
        return tuple(self._arms.keys())

    def display_results(self, playing_window):
        for arm_id in self.arms_ids:
            arm_string_results = [str(r) for r in self._rewards[arm_id]]
            playing_window[f"arm_text{arm_id}"].update("\n".join(arm_string_results))

    @property
    def rewards(self):
        return self._rewards

    def get_reward_by_id(self, arm_id: str) -> List[float]:
        return self.rewards[arm_id]

    @property
    def cumulative_reward(self):
        return self._cumulative_reward

    @property
    def history_of_cumulative_reward(self):
        return self._history_of_cumulative_reward

    @property
    def history_of_play(self):
        return self._history_of_play
