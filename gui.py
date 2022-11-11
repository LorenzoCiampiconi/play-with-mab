from pathlib import Path
from typing import Union, Tuple

import io
from PIL import Image
import PySimpleGUI as sg

from mab import MabProblem

path = Path(__file__).parent


class BarcelonaMabGUI:
    slot_img = path / "slot.png"
    arm_button_size = (5, 1.3)
    play_window_size = (1250, 800)
    results_font_size = ("helvetica", 25)

    def __init__(self):
        self.mab_problem: Union[None, MabProblem] = None

    @staticmethod
    def get_byte_64_image(img: str, dim: Tuple[int,int]=(200, 200)):
        image = Image.open(img)
        image.thumbnail(dim)
        bio = io.BytesIO()
        image.save(bio, format="PNG")

        return bio.getvalue()

    def instantiate_problem(self, configuration=None):
        if configuration is not None:
            arms = dict()
            for id in configuration:
                arms[id] = configuration[id]["dist_class"](configuration[id]["dist_params"])
            self.mab_problem = MabProblem(arms)
        else:
            self.mab_problem = MabProblem()

    def pull_by_event(self, event_str: str, playing_window):
        if "arm" in event_str:
            arm_code = int(event_str[-1])
            return self.mab_problem.pull(arm_code)

    def _get_layout_col_by_arm_id(self, arm_id):
        arm_col = [
            [sg.Image(self.get_byte_64_image(self.slot_img))],
            [sg.Button(f'arm{arm_id}', size=self.arm_button_size)],
            [sg.Text('', key=f"arm_text{arm_id}", font=self.results_font_size)]
        ]
        return arm_col

    def get_play_layout(self):
        layout = [
            [
                sg.Column(self._get_layout_col_by_arm_id(arm_id), vertical_alignment = 't')
                for arm_id in self.mab_problem.arms_ids
            ]
        ]

        return layout

    def play_window_process(self):
        self.instantiate_problem()
        layout = self.get_play_layout()

        window = sg.Window(
            "Get rich with our MAB",
            layout,
            size=self.play_window_size
        )

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            self.pull_by_event(event, playing_window=window)
            self.mab_problem.display_results(playing_window=window)
