from pathlib import Path

img_path = Path(__file__).parent / "img"

from game_core.gui.bmab_gui import BarcelonaMABGUINewLayout
from game_core.gui.simulating_gui import AlgorithmEmployingSimulatingGUIMixin


class BarcelonaMABAlgorithmSimulatingGUI(AlgorithmEmployingSimulatingGUIMixin, BarcelonaMABGUINewLayout):
    ...
