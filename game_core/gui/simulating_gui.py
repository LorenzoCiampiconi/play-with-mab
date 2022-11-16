import abc
import time
from typing import Optional, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from game_core.gui.bmab_gui import BarcelonaMabGUI
from game_core.simulation.algorithm import MABAlgorithm
from game_core.statistic.mab import MABProblem

import PySimpleGUI as sg


class SimulatingGUIMixinABC(metaclass=abc.ABCMeta):
    max_simulation_steps = 100
    mab_problem: MABProblem

    _cumulative_reward_fig_label = "cumulative_reward_fig"

    def __init__(self, *, simulate: Optional[bool] = False, simulation_interval=1, **kwargs):
        super().__init__(**kwargs)
        self._figures = {}
        self._simulate = simulate
        self._simulation_interval = simulation_interval
        self._last_simulation_step = 0
        self._total_simulation_steps = 0
        self._simulation_window = None

    @property
    def is_time_to_simulate(self):
        now = time.time()
        return (
            self._last_simulation_step == 0 or now > self._simulation_interval + self._last_simulation_step
        ) and self._total_simulation_steps <= self.max_simulation_steps

    def start_simulation(self, simulation_step=max_simulation_steps):
        self.max_simulation_steps = simulation_step
        self._simulate = True
        self._last_simulation_step = 0
        self._total_simulation_steps = 0

    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        pass

    @property
    def timeout(self):
        return self._simulation_interval / 2

    def _get_simulation_window_layout(self):
        return [[sg.Canvas(key=self._cumulative_reward_fig_label)]]

    def open_simulation_window(self):
        layout = self._get_simulation_window_layout()
        self._simulation_window = sg.Window(
            "Simulation Window",
            layout,
            size=(800, 900),
            finalize=True,
            background_color="white",
            element_justification="c",
        )

    def draw_figure_on_window_canvas(
        self, window: sg.Window, canvas_id: str, figure: plt.Figure, figure_agg_label: str
    ):
        if figure_agg_label in self._figures:
            self._figures[figure_agg_label].get_tk_widget().forget()

        canvas = window[canvas_id].TKCanvas
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)

        figure_canvas_agg.draw_idle()
        figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        return figure_canvas_agg

    def update_cumulative_rewards(self):
        cumulative_reward = self.mab_problem.history_of_cumulative_reward
        cumulative_reward_by_id = self.mab_problem.history_of_cumulative_reward_by_id()
        figure = plt.figure(figsize=(7, 4))
        plt.clf()
        plt.ylim(0, self.max_simulation_steps)
        plt.xlim(0, self.max_simulation_steps)
        for arm in self.mab_problem.arms_ids:
            plt.plot(cumulative_reward_by_id[arm], label=f"Arm {arm}", linewidth=2.5)
        plt.plot(cumulative_reward, label="Total", linewidth=3)

        plt.title("Cumulative Rewards", fontsize="12")
        plt.xlabel("time-steps")
        plt.ylabel("Cumulative reward ($)")
        plt.legend(frameon=False)
        plt.tight_layout()

        self._figures[self._cumulative_reward_fig_label] = self.draw_figure_on_window_canvas(
            self._simulation_window, self._cumulative_reward_fig_label, figure, self._cumulative_reward_fig_label
        )

    def update_simulation_window(self):
        self.update_cumulative_rewards()

    def read_simulation_window(self):
        event, values = self._simulation_window.read(timeout=0.01)
        if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
            self._simulation_window = None
            self._simulate = False

    def reset_environment(self):
        self.mab_problem.reset()

    def event_loop_stem(self, event, window):
        super().event_loop_stem(event, window)

        if self._simulation_window is not None:
            self.read_simulation_window()

        if event == "Simulate":
            self._simulate = True
            self.reset_environment()
            self.open_simulation_window()

        if self._simulate and self.is_time_to_simulate:
            self.simulate(window)
            self.update_simulation_window()


class AlgorithmEmployingSimulatingGUIMixin(SimulatingGUIMixinABC):
    mab_problem: MABProblem
    _algorithm_stats_fig_label = "_algorithm_stats_fig"

    def __init__(self, algorithm_class: Type[MABAlgorithm], algorithm_kwargs, **kwargs):
        super().__init__(**kwargs)
        algorithm_kwargs["mab_problem"] = self.mab_problem
        self._algorithm = algorithm_class(**algorithm_kwargs)

    def simulate(self, window):
        print(f"asking {self._algorithm.algorithm_label} algorithm to play... {self._algorithm.info()}")

        if self._algorithm._mab_problem is not self.mab_problem:
            self._algorithm._mab_problem = self.mab_problem

        arm = f"arm_{self._algorithm.select_arm()}"
        window.Element(arm).TKButton.invoke()
        self._last_simulation_step = time.time()
        self._total_simulation_steps += 1

    def _get_simulation_window_layout(self):
        col = [[sg.Canvas(key=self._cumulative_reward_fig_label)], [sg.Canvas(key=self._algorithm_stats_fig_label)]],

        return [
            col
        ]

    def update_algorithm_stats(self):
        figure = self._algorithm.plot_stats()

        self._figures[self._algorithm_stats_fig_label] = self.draw_figure_on_window_canvas(
            self._simulation_window, self._algorithm_stats_fig_label, figure, self._algorithm_stats_fig_label
        )

    def update_simulation_window(self):
        super().update_simulation_window()
        self.update_algorithm_stats()


class BarcelonaMABAlgorithmSimulatingGUI(AlgorithmEmployingSimulatingGUIMixin, BarcelonaMabGUI):
    ...
