from game_core.gui import BarcelonaMABAlgorithmSimulatingGUI

if __name__ == "__main__":
    gui = BarcelonaMABAlgorithmSimulatingGUI(simulate=False, algorithm_type="thompson_sampling", algorithm_kwargs={})
    gui.menu_window_process()
