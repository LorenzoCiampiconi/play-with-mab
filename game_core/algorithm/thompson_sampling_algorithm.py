from collections import defaultdict
from scipy.stats import beta
from typing import Union, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import operator

from game_core.algorithm.base_algorithm import MABAlgorithm
from game_core.statistic.distribution import BetaDistribution
from game_core.configs.configs import CB_Lastminute


class ThompsonSampling(MABAlgorithm):
    algorithm_label = "Thompson Sampling"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta_distributions_parameters: Dict[str, Tuple[int, int]] = defaultdict(lambda: (1, 1))
        self._last_played_arm: Union[None, str] = None
        self.iteration = 0

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
            axs[0].plot(x, pdf, label=f"Arm {arm}", linewidth=3)
            axs[0].fill_between(x, pdf, step="pre", alpha=0.3)

        axs[0].set_title("Beta Distributions of arms", loc="right", fontsize=10, fontweight="bold")
        axs[0].legend(frameon=False)

        self.plot_iteration_histogram(axs[1], max_plays=max_plays)

        plt.tight_layout()

        return fig

    def info(self) -> str:
        return f"INFO - (parameters for distributions are {self.beta_distributions_parameters.items()})"
