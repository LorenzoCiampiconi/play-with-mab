from typing import Union

import PySimpleGUI as sg

from mab import MabProblem


class BarcelonaMabGUI:
    def __init__(self):
        self.mab_problem: Union[None, MabProblem] = None

    def instantiate_problem(self, configuration=None):
        if configuration is not None:
            arms = dict()
            for id in configuration:
                arms[id] = configuration[id]["dist_class"](configuration[id]["dist_params"])
            self.mab_problem = MabProblem(arms)
        else:
            self.mab_problem = MabProblem()

    def pull_by_event(self, event_str: str):
        if "arm" in event_str:
            arm_code = int(event_str[-1])
            return self.mab_problem.pull(arm_code)

    @staticmethod
    def get_play_layout():
        layout = [[sg.Text("Qui ci sará la slot")], [sg.Button("arm_1")], [sg.Button("arm_2")], [sg.Button("arm_3")]]
        return layout

    def play_window_process(self):
        self.instantiate_problem()
        layout = self.get_play_layout()
        window = sg.Window("Get rich with our MAB", layout)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            self.pull_by_event(event)
            self.mab_problem.display_results()



