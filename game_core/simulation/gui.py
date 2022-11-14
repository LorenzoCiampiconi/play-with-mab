import abc
import random
import time
from typing import Optional

from game_core.gui.bmab_gui import BarcelonaMabGUI


class SimulatingGUIMixinABC(metaclass=abc.ABCMeta):
    max_simulation_steps = 10

    def __init__(self, simulate: Optional[bool] = False, simulation_interval=1):
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


class RandomSimulatingGUIMixin(SimulatingGUIMixinABC):
    def simulate(self, window):
        print("simulating play with default play")
        arm = random.randint(1, 4)
        window.Element(f"arm{arm}").TKButton.invoke()
        self._last_simulation_step = time.time()
        self._total_simulation_steps += 1
