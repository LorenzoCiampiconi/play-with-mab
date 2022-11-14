from pathlib import Path
from typing import Union, Tuple

import io
from PIL import Image
import PySimpleGUI as sg

from game_core.gui import img_path
from game_core.gui.base_gui import BaseGUIABC
from game_core.mab import MabProblem


class AnalysisGUI(BaseGUIABC):

    def __init__(self):
        #todo: parameters
        # self.mab_problem = mab_problem
        self.parameters = []


    def get_play_layout(self):
        layout = [
            [sg.Text(key="test")]
        ]

        return layout

    def prepare_for_play(self):
        pass

    def window_layout_post_process(self, window):
       pass

    def event_loop_stem(self, event, window):
        pass
        # window["test"] = self.mab_problem.get_results_by_arm(1)