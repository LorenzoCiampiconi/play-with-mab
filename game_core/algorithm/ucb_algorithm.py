from collections import defaultdict
from random import sample
from typing import Union, Dict

import numpy as np
from matplotlib import pyplot as plt

from game_core.algorithm.base_algorithm import MABAlgorithm


class UpperConfidenceBound1(MABAlgorithm):
    algorithm_label = "UCB-1"

    def __init__(self, reset=False, **kwargs):
        super().__init__(**kwargs)
        self._last_played_arm: Union[None, str] = None
        self._upper_confidence_bounds: Dict[str, Union[float]] = defaultdict(float)

        if reset:
            self.mab_problem.reset()

    def _compute_c(self, arm):
        arm_record = self.mab_problem.record[arm]
        n = arm_record["actions"]
        return np.sqrt(2 * np.log(self.mab_problem.total_actions) / n)

    def _update_upper_confidence_bound(self, arm):
        arm_record = self.mab_problem.record[arm]
        n = arm_record["actions"]

        if n != 0:
            sample_mean = arm_record["reward"] / n
            c = self._compute_c(arm)

            ucb = sample_mean + c
            self._upper_confidence_bounds[arm] = ucb

        else:
            self._upper_confidence_bounds[arm] = np.inf

    def _update_upper_confidence_bounds(self):
        if self._last_played_arm is not None:
            for arm_id in self.mab_problem.arms_ids:
                self._update_upper_confidence_bound(arm_id)

    def select_arm(self) -> str:
        self._update_upper_confidence_bounds()

        bounds_map_to_arms = {v: k for k, v in self._upper_confidence_bounds.items()}
        selected_arm = (
            bounds_map_to_arms[max(bounds_map_to_arms.keys())]
            if self._last_played_arm is not None
            else sample(self.mab_problem.arms_ids, 1)[0]
        )

        self._last_played_arm = selected_arm

        return selected_arm

    def info(self) -> str:
        return f"INFO - (bounds for arms are {self._upper_confidence_bounds})"


class UpperConfidenceBound1Tuned(UpperConfidenceBound1):
    algorithm_label = "UCB-1-tuned"

    def _compute_c(self, arm):
        arm_record = self.mab_problem.record[arm]
        n = arm_record["actions"]
        sample_mean = arm_record["reward"] / n

        variance_bound = arm_record["reward_squared"] / n - sample_mean**2
        variance_bound += np.sqrt(2 * np.log(self.mab_problem.total_actions.total_actions) / n)
        return np.sqrt(np.min([variance_bound, 1 / 4]) * np.log(self.mab_problem.total_actions) / n)
