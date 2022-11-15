import abc
import random
from collections import defaultdict
from random import sample
from typing import Union, Mapping, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import operator
from game_core.statistic.distribution import BetaDistribution
from game_core.statistic.mab import MABProblem


class MABAlgorithm(metaclass=abc.ABCMeta):
    algorithm_label = "NotImplemented"
    mab_problem: MABProblem

    def __init__(self, *, mab_problem: MABProblem):
        self._mab_problem: MABProblem = mab_problem

    @property
    def mab_problem(self):
        assert self._mab_problem is not None
        return self._mab_problem

    @abc.abstractmethod
    def select_arm(self) -> str:
        pass

    @abc.abstractmethod
    def info(self) -> str:
        pass

    @abc.abstractmethod
    def plot_stats(self) -> plt.Figure:
        pass


class RandomAlgorithm(MABAlgorithm):
    algorithm_label = "RANDOM"

    def select_arm(self) -> str:
        selected_arm_id = sample(self.mab_problem.arms_ids, 1)[0]
        return f"arm{selected_arm_id}"

    def info(self) -> str:
        "(no info available)"


class UpperConfidenceBound1(MABAlgorithm):
    algorithm_label = "UCB-1"

    def __init__(self, reset=False, **kwargs):
        super().__init__(**kwargs)
        self._last_played_arm: Union[None, str] = None
        self._upper_confidence_bounds: Dict[str, Union[float]] = defaultdict(float)

        if reset:
            self.mab_problem.reset()

    def _update_upper_confidence_bound(self, arm):
        arm_record = self.mab_problem.record[arm]
        n = arm_record["actions"]

        if n != 0:
            sample_mean = arm_record["reward"] / arm_record["actions"] if n != 0 else np.inf
            c = np.sqrt(2 * np.log(self.mab_problem.total_actions) / arm_record["actions"])

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

    def plot_stats(self) -> plt.Figure:
        return plt.figure()


class ThompsonSampling(MABAlgorithm):
    algorithm_label = "Thompson Sampling"

    def __init__(self, reset=False, **kwargs):
        super().__init__(**kwargs)
        self.beta_distributions_parameters: Dict[str, Tuple[int, int]] = defaultdict(lambda: (1, 1))
        self._last_played_arm: Union[None, str] = None
        if reset:
            self.mab_problem.reset()

    def _update_beta_distributions(self):
        for arm in self.mab_problem.arms_ids:
            successes = self.mab_problem.record[arm]['reward'] if self.mab_problem.record[arm]['reward'] > 0 else 1
            failures = self.mab_problem.record[arm]['actions'] - self.mab_problem.record[arm]['reward'] if self.mab_problem.record[arm]['actions'] - self.mab_problem.record[arm]['reward'] > 0 else 1
            self.beta_distributions_parameters[arm] = (successes, failures)

    def select_arm(self) -> str:
        self._update_beta_distributions()
        samples = {arm: BetaDistribution(self.beta_distributions_parameters[arm][0], self.beta_distributions_parameters[arm][1]).sample() for arm in self.mab_problem.arms_ids}
        selected_arm = max(samples.items(), key=operator.itemgetter(1))[0]
        self._last_played_arm = selected_arm
        return selected_arm

    def plot_stats(self) -> plt.Figure:
        return plt.figure()

    def info(self) -> str:
        return f"INFO - (parameters for distributions are {self.beta_distributions_parameters.items()})"