import abc
import random
from collections import defaultdict
from random import sample
from scipy.stats import beta
from typing import Union, Mapping, Dict, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator
from game_core.statistic.distribution import BetaDistribution
from game_core.statistic.mab import MABProblem
from game_core.configs.configs import color_list, CB_Lastminute


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
    def plot_stats(self, figsize, max_plays=100) -> plt.Figure:
        pass

    def plot_iteration_histogram(self, axs: plt.Axes, max_plays=100):
        axs.set_xlim(0, max_plays)
        axs.set(xlabel="time_steps", ylabel="arm")
        axs.set_title("Number of plays per arm", loc="right", fontsize=10, fontweight="bold")
        labels = []
        plays = []
        for arm in self.mab_problem.arms_ids:
            labels.append(arm)
            plays.append(self.mab_problem.record[arm]["actions"])
        sns.barplot(x=plays, y=labels, axes=axs)


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
        self.iteration = 0
        if reset:
            self.mab_problem.reset()

    def _update_beta_distributions(self):
        for arm in self.mab_problem.arms_ids:
            successes = self.mab_problem.record[arm]["reward"] if self.mab_problem.record[arm]["reward"] > 0 else 1
            failures = (
                self.mab_problem.record[arm]["actions"] - self.mab_problem.record[arm]["reward"]
                if self.mab_problem.record[arm]["actions"] - self.mab_problem.record[arm]["reward"] > 0
                else 1
            )
            self.beta_distributions_parameters[arm] = (successes, failures)

    @property
    def beta_dist_of_arms(self) -> Dict[str, BetaDistribution]:
        return {
            arm: BetaDistribution(
                self.beta_distributions_parameters[arm][0], self.beta_distributions_parameters[arm][1]
            )
            for arm in self.mab_problem.arms_ids
        }

    def select_arm(self) -> str:
        if self.iteration < 3:
            selected_arm = self.iteration + 1
            self._last_played_arm = selected_arm
        else:
            self._update_beta_distributions()
            samples = {arm: beta_dist.sample() for arm, beta_dist in self.beta_dist_of_arms.items()}
            selected_arm = max(samples.items(), key=operator.itemgetter(1))[0]
            self._last_played_arm = selected_arm
        self.iteration += 1
        return selected_arm

    def plot_stats(self, figsize, max_plays=100) -> plt.Figure:
        fig, axs = plt.subplots(2, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

        beta_dists_of_arms: dict = self.beta_dist_of_arms

        low_lim = min(beta.ppf(0.01, beta_dist.a, beta_dist.b) for beta_dist in beta_dists_of_arms.values())
        up_lim = 1

        fig.suptitle("Algorithm parameters", fontsize=12, fontweight="bold", color=CB_Lastminute)

        axs[0].set_ylim(0, 15)
        axs[0].set_xlim(0, up_lim)

        x = np.linspace(low_lim, up_lim, 1000)

        for arm, beta_dist in beta_dists_of_arms.items():
            pdf = beta_dist.pdf(x)
            axs[0].plot(x, pdf, label=f"Arm {arm}")

        axs[0].set_title("Beta Distributions of arms", loc="right", fontsize=10, fontweight="bold")
        axs[0].legend(frameon=False)

        self.plot_iteration_histogram(axs[1], max_plays=max_plays)

        plt.tight_layout()

        return fig

    def info(self) -> str:
        return f"INFO - (parameters for distributions are {self.beta_distributions_parameters.items()})"
