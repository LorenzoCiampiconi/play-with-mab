from typing import Union, Tuple, Optional

import io
from PIL import Image
import PySimpleGUI as sg

from game_core.gui import img_path
from game_core.gui.base_gui import BaseGUIABC
from game_core.statistic.mab import MABProblem
from game_core.configs.configs import color_list


class BarcelonaMabGUI(BaseGUIABC):
    slot_img_file = img_path / "slot.png"
    arm_button_size = (0.1, 0.1)
    slot_img_size = (300, 200)
    results_font_size = ("helvetica", 25)
    binary_win_label = "$"
    binary_loss_label = "X"
    mapping = {0: binary_loss_label, 1: binary_win_label}

    def __init__(self, *, configuration=None, max_length_col: Optional[int] = 14, **kwargs):
        super().__init__(**kwargs)
        self.mab_problem: Union[None, MABProblem] = None
        self.instantiate_problem(configuration=configuration)
        self._max_length_col = max_length_col

    @staticmethod
    def get_byte_64_image(img: str, size: Tuple[int, int] = (200, 200)):
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
            self.mab_problem = MABProblem(arms)
        else:
            self.mab_problem = MABProblem()

    def pull_by_event(self, event_str: str):
        if "arm" in event_str:
            arm_code = event_str.split("_")[-1]
            return self.mab_problem.pull(arm_code)

    def _get_layout_col_by_arm_id(self, arm_id) -> sg.Column:
        slot_img = self.get_byte_64_image(self.slot_img_file, size=self.slot_img_size)

        arm_col = [
            [
                sg.Text(
                    f"Arm {arm_id}",
                    font=self.results_font_size,
                    background_color="#35654d",
                    text_color=color_list[int(arm_id) - 1],
                )
            ],
            [sg.Button(f"arm_{arm_id}", size=self.arm_button_size, image_data=slot_img, button_color="#35674d")],
            [
                sg.Text(
                    "",
                    key=f"arm_text{arm_id}",
                    font=self.results_font_size,
                    background_color="#35654d",
                    justification="center",
                )
            ],
        ]
        return sg.Column(
            arm_col,
            vertical_alignment="t",
            element_justification="c",
            expand_x=True,
            expand_y=True,
            key=f"col_{arm_id}",
            background_color="#35654d",
        )

    def get_play_layout(self):
        layout = [
            [self._get_layout_col_by_arm_id(arm_id) for arm_id in self.mab_problem.arms_ids],
            [sg.Button("Open Simulation", size=(20, 1.3))],
        ]

        return layout

    def event_loop_stem(self, event, window):
        self.pull_by_event(event)
        self.update_on_screen_mab_history(window)

    def _process_rewards_for_ui(self, rewards):
        if all(r in {0, 1} for r in rewards):
            s_rewards = [self.mapping[r] for r in rewards]
            if len(rewards) > self._max_length_col - 1:
                shown_rewards = s_rewards[-self._max_length_col - 1 :]
                collapsed_rewards = [
                    f"... {sum(rewards[:-self._max_length_col - 1])}{self.binary_win_label}, "
                    f"{len(rewards) - sum(rewards[:-self._max_length_col - 1])}{self.binary_loss_label} ..."
                ]
                s_rewards = collapsed_rewards + shown_rewards

            return s_rewards
        else:
            return [str(r) for r in rewards]

    def update_on_screen_mab_history(self, window):
        for arm_id in self.mab_problem.arms_ids:
            arm_string_results = self._process_rewards_for_ui(self.mab_problem.rewards[arm_id])

            window[f"arm_text{arm_id}"].update("\n".join(arm_string_results))

    def prepare_for_play(self):
        self.instantiate_problem()

    def window_layout_post_process(self, window):
        for arm_id in self.mab_problem.arms_ids:
            window[f"col_{arm_id}"].Widget.configure(borderwidth=2, relief=sg.RELIEF_SOLID)
