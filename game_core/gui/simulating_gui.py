import abc
import time
from typing import Optional, Type, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from game_core.gui import img_path
from game_core.gui.bmab_gui import BarcelonaMabGUI, BarcelonaMABGUINewLayout
from game_core.simulation.algorithm import (
    MABAlgorithm,
    UpperConfidenceBound1,
    RandomAlgorithm,
    ThompsonSampling,
    EpsilonGreedy,
)
from game_core.statistic.mab import MABProblem
from game_core.configs.configs import color_list, CB_Lastminute, CB_Gold

import PySimpleGUI as sg


class SimulatingGUIMixinABC(metaclass=abc.ABCMeta):
    max_simulation_steps = 100
    mab_problem: MABProblem
    play_image_file = img_path / "play_button.png"
    pause_image_file = img_path / "pause_button.png"
    regret_image_file = img_path / "pepe_regret.png"
    expectation_image_file = img_path / "pepe_expectation.png"
    sim_button_size = (2, 2)

    _cumulative_reward_fig_label = "cumulative_reward_fig"

    def __init__(self, *, simulate: Optional[bool] = False, simulation_interval=1, **kwargs):
        super().__init__(**kwargs)
        self._figures = {}
        self._simulate = simulate
        self._simulation_interval = simulation_interval
        self._last_simulation_step = 0
        self._total_simulation_steps = 0
        self._simulation_window = None
        self._plot_regret = False
        self._plot_expected_reward = False
        self._plot_figsize = (6, 6)
        self._cumulative_reward_plot_zoom = 1
        self._cumulative_reward_plot_x_shift = 0
        self._cumulative_reward_plot_y_shift = 0

    @property
    def is_time_to_simulate(self):
        now = time.time()
        return (
            self._last_simulation_step == 0 or now > self._simulation_interval + self._last_simulation_step
        ) and self._total_simulation_steps <= self.max_simulation_steps

    def _reset_zoom(self):
        self._cumulative_reward_plot_zoom = 1
        self._cumulative_reward_plot_x_shift = 0
        self._cumulative_reward_plot_y_shift = 0

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
        return [[sg.Canvas(key=self._cumulative_reward_fig_label)]]  # todo objectify

    def open_simulation_window(self):
        layout = self._get_simulation_window_layout()
        self._simulation_window = sg.Window(
            "Simulation Window",
            layout,
            size=(800, 1100),
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
        figure = plt.figure(figsize=(7, 3))
        plt.clf()
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)

        base_xlim = 0, self.max_simulation_steps
        base_ylim = (0, 80)

        xlim = np.divide(
            (
                base_xlim[0] + self._cumulative_reward_plot_x_shift / self._cumulative_reward_plot_zoom,
                base_xlim[1] + self._cumulative_reward_plot_x_shift / self._cumulative_reward_plot_zoom,
            ),
            self._cumulative_reward_plot_zoom,
        )
        ylim = np.divide(
            (
                base_ylim[0] + self._cumulative_reward_plot_y_shift / self._cumulative_reward_plot_zoom,
                base_ylim[1] + self._cumulative_reward_plot_y_shift / self._cumulative_reward_plot_zoom,
            ),
            self._cumulative_reward_plot_zoom,
        )

        plt.xlim(xlim)
        plt.ylim(ylim)

        for arm in self.mab_problem.arms_ids:
            plt.plot(cumulative_reward_by_id[arm], label=f"Arm {arm}", linewidth=2.5)

        plt.plot(cumulative_reward, label="Total", linewidth=3)

        expected_reward = self.mab_problem.best_arm.get_cumulate_expected_value_for_steps(
            self.mab_problem.total_actions
        )
        if self._plot_expected_reward and self.mab_problem.best_arm is not None:
            plt.plot(expected_reward, label=f"Expected reward", linewidth=3, color=CB_Gold)
        if self._plot_regret and self.mab_problem.best_arm is not None:
            regret = self.mab_problem.regret_history
            plt.plot(regret, label=f"Regret", linewidth=4)

        plt.title("Cumulative Rewards", fontsize="12", fontweight="bold", color=CB_Lastminute)
        plt.xlabel("Time Steps", fontweight="bold")
        plt.ylabel("$", fontweight="bold", fontsize="15", color=CB_Gold, rotation=0)
        plt.legend(frameon=False)
        plt.tight_layout()

        self._figures[self._cumulative_reward_fig_label] = self.draw_figure_on_window_canvas(
            self._simulation_window, self._cumulative_reward_fig_label, figure, self._cumulative_reward_fig_label
        )

    def update_simulation_window(self):
        self.update_cumulative_rewards()

    def read_simulation_window(self):
        event, values = self._simulation_window.read(timeout=0.01)
        if event == "Play":
            self._simulate = True
        elif event == "Pause":
            self._simulate = False
        elif event == "Regret":
            self._plot_regret = not self._plot_regret
            self.update_simulation_window()
        elif event == "Expectation":
            self._plot_expected_reward = not self._plot_expected_reward
            self.update_simulation_window()
        elif event == "+zoom":
            self._cumulative_reward_plot_zoom += 1
            self.update_simulation_window()
        elif event == "-zoom":
            self._cumulative_reward_plot_zoom -= (
                1 if self._cumulative_reward_plot_zoom > 1 else self._cumulative_reward_plot_zoom / 10
            )
            self.update_simulation_window()
        elif event == "reset-zoom":
            self._reset_zoom()
            self.update_simulation_window()
        elif event == "+x":
            self._cumulative_reward_plot_x_shift += 10
            self.update_simulation_window()
        elif event == "-x":
            self._cumulative_reward_plot_x_shift -= 10
            self.update_simulation_window()
        elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
            self._simulation_window = None
            self._simulate = False

    def reset_environment(self):
        self.mab_problem.reset()

    def event_loop_stem(self, event, window):
        super().event_loop_stem(event, window)

        if self._simulation_window is not None:
            self.read_simulation_window()

        if event == "Open Simulation":
            self._simulate = False
            self.reset_environment()
            self.open_simulation_window()

        if self._simulate and self.is_time_to_simulate:
            self.simulate(window)
            self.update_simulation_window()


class AlgorithmEmployingSimulatingGUIMixin(SimulatingGUIMixinABC):
    mab_problem: MABProblem
    _algorithm_stats_fig_label = "_algorithm_stats_fig"

    algorithm_class_dict: Dict[str, Type[MABAlgorithm]] = dict(
        epsilon_greedy=EpsilonGreedy,
        random_policy=RandomAlgorithm,
        thompson_sampling=ThompsonSampling,
        ucb_1=UpperConfidenceBound1,
    )

    def __init__(self, algorithm_type: str, algorithm_kwargs: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        algorithm_kwargs["mab_problem"] = self.mab_problem

        self.__set_mab_algorithm__(algorithm_type, algorithm_kwargs)

    def __set_mab_algorithm__(self, algorithm_type, algorithm_kwargs: Optional[dict] = None):
        algorithm_class = self.algorithm_class_dict[algorithm_type]
        self._algorithm_type = algorithm_type

        if algorithm_kwargs is None:
            algorithm_kwargs = algorithm_class.default_kwargs

        self._algorithm: MABAlgorithm = algorithm_class(**algorithm_kwargs)

    def set_mab_algorithm(self, algorithm_type, algorithm_kwargs: Optional[dict] = None):
        self.__set_mab_algorithm__(algorithm_type, algorithm_kwargs)

    @property
    def algorithm_type(self) -> str:
        return self._algorithm_type

    def simulate(self, window):
        print(f"asking {self._algorithm.algorithm_label} algorithm to play... {self._algorithm.info()}")

        if self._algorithm._mab_problem is not self.mab_problem:
            self._algorithm._mab_problem = self.mab_problem

        arm = f"arm_{self._algorithm.select_arm()}"
        window.Element(arm).TKButton.invoke()
        self._last_simulation_step = time.time()
        self._total_simulation_steps += 1

    def _get_simulation_window_layout(self):
        play_img = self.get_byte_64_image(self.play_image_file, size=(15, 15))
        pause_img = self.get_byte_64_image(self.pause_image_file, size=(15, 15))
        regret_img = self.get_byte_64_image(self.regret_image_file, size=(53, 53))
        expectation_img = self.get_byte_64_image(self.expectation_image_file, size=(103, 53))
        col = (
            [
                [sg.Canvas(key=self._cumulative_reward_fig_label)],
                [
                    sg.Button("+zoom", button_color=CB_Lastminute),
                    sg.Button("-zoom", button_color=CB_Lastminute),
                    sg.Button("reset-zoom", button_color=CB_Lastminute),
                    sg.Button("+x", button_color=CB_Lastminute),
                    sg.Button("-x", button_color=CB_Lastminute),
                ],
                [sg.Canvas(key=self._algorithm_stats_fig_label)],
            ],
        )

        return [
            col,
            [
                sg.Button("Play", size=self.sim_button_size, image_data=play_img, button_color=CB_Lastminute),
                sg.Button("Pause", size=self.sim_button_size, image_data=pause_img, button_color=CB_Lastminute),
                sg.Button("Regret", image_data=regret_img, button_color=CB_Lastminute),
                sg.Button("Expectation", image_data=expectation_img, button_color=CB_Lastminute),
            ],
        ]

    def update_algorithm_stats(self):
        figure = self._algorithm.plot_stats(self._plot_figsize)

        self._figures[self._algorithm_stats_fig_label] = self.draw_figure_on_window_canvas(
            self._simulation_window, self._algorithm_stats_fig_label, figure, self._algorithm_stats_fig_label
        )

    def update_simulation_window(self):
        super().update_simulation_window()
        self.update_algorithm_stats()


class BarcelonaMABAlgorithmSimulatingGUI(AlgorithmEmployingSimulatingGUIMixin, BarcelonaMABGUINewLayout):
    ...
