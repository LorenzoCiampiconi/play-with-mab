import abc
import logging
from pathlib import Path
from typing import Union, Tuple

import io
from PIL import Image
import PySimpleGUI as sg

logger = logging.getLogger(__name__)


class BaseGUIABC(metaclass=abc.ABCMeta):
    play_window_size = (1000, 600)
    menu_window_size = (1000, 600)

    play_event = "Play"
    switch_mab_algorithm_event = "Switch MAB Algorithm"
    algorithm_self_selection_event = "Algorithm self selection"

    @staticmethod
    def collapse(layout, key):
        """
        Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
        :param layout: The layout for the section
        :param key: Key used to make this seciton visible / invisible
        :return: A pinned column that can be placed directly into your layout
        :rtype: sg.pin
        """
        return sg.pin(sg.Column(layout, key=key))

    @staticmethod
    def get_byte_64_image(img: str, size: Tuple[int, int] = (200, 200)):
        image = Image.open(img)
        image.thumbnail(size)
        bio = io.BytesIO()
        image.save(bio, format="PNG")

        return bio.getvalue()

    def prepare_for_play(self):
        pass

    @abc.abstractmethod
    def window_layout_post_process(self, window):
        pass

    @abc.abstractmethod
    def get_play_layout(self):
        ...

    @abc.abstractmethod
    def event_loop_stem(self, event, window):
        pass

    @property
    def timeout(self):
        return None

    @property
    @abc.abstractmethod
    def algorithm_type(self) -> str:
        pass

    def switch_mab_algorithm(self):
        logger.warning("Not Implemented feature")

    def get_menu_layout(self):
        layout = [
            [
                sg.Button(BaseGUIABC.play_event, size=(20, 1.3)),
                sg.Button(BaseGUIABC.switch_mab_algorithm_event, size=(20, 1.3)),
                sg.Button(BaseGUIABC.algorithm_self_selection_event, size=(20, 1.3)),
            ],
        ]

        return layout

    def get_menu_window(self) -> sg.Window:
        layout = self.get_menu_layout()

        window = sg.Window(
            "Get rich with our MAB - Made with <3 by Lore & Luke",
            layout,
            element_justification="c",
            size=self.menu_window_size,
            finalize=True,
            resizable=True,
        )

        return window

    def menu_window_process(self):
        window = self.get_menu_window()
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            elif event == BaseGUIABC.play_event:
                window.close()
                self.play_window_process()
                self.menu_window_process()

            elif event == BaseGUIABC.switch_mab_algorithm_event:
                self.switch_mab_algorithm()

    def play_window_process(self):
        self.prepare_for_play()
        layout = self.get_play_layout()

        window = sg.Window(
            "Get rich with our MAB - Made with <3 by Lore & Luke",
            layout,
            element_justification="c",
            size=self.play_window_size,
            finalize=True,
            resizable=True,
        )

        self.window_layout_post_process(window)

        while True:
            event, values = window.read(timeout=self.timeout)
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            self.event_loop_stem(event, window)
