import abc
from pathlib import Path
from typing import Union, Tuple

import io
from PIL import Image
import PySimpleGUI as sg


class BaseGUIABC(metaclass=abc.ABCMeta):
    play_window_size = (1250, 800)

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

    @abc.abstractmethod
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

    def play_window_process(self):
        self.prepare_for_play()
        layout = self.get_play_layout()

        window = sg.Window(
            "Get rich with our MAB",
            layout,
            element_justification='c',
            size=self.play_window_size,
            finalize=True
        )

        self.window_layout_post_process(window)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Cancel":  # if user closes window or clicks cancel
                break

            self.event_loop_stem(event, window)
