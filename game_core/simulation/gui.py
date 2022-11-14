import abc
import random
import time
from typing import Optional, Type

from game_core.gui.bmab_gui import BarcelonaMabGUI
from game_core.simulation.algorithm import MABAlgorithm
from game_core.statistic.mab import MABProblem


class SimulatingGUIMixinABC(metaclass=abc.ABCMeta):
    max_simulation_steps = 30

    def __init__(self, *, simulate: Optional[bool] = False, simulation_interval=1, **kwargs):
        super().__init__(**kwargs)
        self._simulate = simulate
        self._simulation_interval = simulation_interval
        self._last_simulation_step = 0
        self._total_simulation_steps = 0

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

    def event_loop_stem(self, event, window):
        super().event_loop_stem(event, window)

        if self._simulate and self.is_time_to_simulate:
            self.simulate(window)


class AlgorithmEmployingSimulatingGUIMixin(SimulatingGUIMixinABC):
    mab_problem: MABProblem

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
