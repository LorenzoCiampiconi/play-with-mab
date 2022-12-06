import abc
from random import sample

import seaborn as sns
from matplotlib import pyplot as plt

from game_core.statistic.mab import MABProblem


class MABAlgorithm(metaclass=abc.ABCMeta):
    algorithm_label = "NotImplemented"
    mab_problem: MABProblem

    default_kwargs = {}

    def __init__(self, *, mab_problem: MABProblem):
        self._mab_problem: MABProblem = mab_problem
        self.set_mab_problem_best_arm()  # todo refactor

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

    def plot_stats(self, figsize, max_plays=100) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        self.plot_iteration_histogram(ax, max_plays=max_plays)

        plt.tight_layout()

        return fig

    def set_mab_problem_best_arm(self):
        arms_w_expected_value = [v for v in self.mab_problem.arms.values()]
        arms_w_expected_value.sort(key=lambda x: x.expected_value, reverse=True)
        self._mab_problem.set_best_arm(arms_w_expected_value[0])

    def plot_iteration_histogram(self, axs: plt.Axes, max_plays=100):
        axs.set_xlim(0, max_plays)
        axs.set_xlabel("Time Steps", fontdict=dict(weight="bold"))
        axs.set_ylabel("Arm", fontdict=dict(weight="bold"))
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
