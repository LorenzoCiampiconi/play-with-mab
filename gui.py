from pathlib import Path
from typing import Union, Tuple

import io
from PIL import Image
import PySimpleGUI as sg

from mab import MabProblem

img_path = Path(__file__).parent / "img"


class BarcelonaMabGUI:
    slot_img_file = img_path / "slot.png"
    background = img_path / "background.png"
    button_img_file = img_path / "button3.png"
    arm_button_size = (3.5, 3.5)
    slot_img_size = (300,200)
    play_window_size = (1250, 800)
    results_font_size = ("helvetica", 25)

    def __init__(self):
        self.mab_problem: Union[None, MabProblem] = None

    @staticmethod
    def get_byte_64_image(img: str, size: Tuple[int, int]=(200, 200)):
        image = Image.open(img)
        image.thumbnail(size)
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

    def _get_layout_col_by_arm_id(self, arm_id) -> sg.Column:
        slot_img = self.get_byte_64_image(self.slot_img_file, size=self.slot_img_size)

        arm_col = [
            [sg.Button(f'arm{arm_id}', size=self.arm_button_size, image_data=slot_img)],
            [sg.Text('', key=f"arm_text{arm_id}", font=self.results_font_size)]
        ]
        return sg.Column(arm_col, vertical_alignment='t', element_justification='c', expand_x=True, expand_y=True, key=f"col_{arm_id}")

    def get_play_layout(self):
        layout = [
            [
                self._get_layout_col_by_arm_id(arm_id)
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
            element_justification='c',
            size=self.play_window_size,
            finalize=True
        )
        for arm_id in self.mab_problem.arms_ids:
            window[f'col_{arm_id}'].Widget.configure(borderwidth=2, relief=sg.RELIEF_SOLID)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            self.pull_by_event(event, playing_window=window)
            self.mab_problem.display_results(playing_window=window)
