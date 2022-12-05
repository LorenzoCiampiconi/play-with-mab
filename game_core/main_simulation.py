from game_core.gui.simulating_gui import BarcelonaMABAlgorithmSimulatingGUI
from game_core.simulation.algorithm import RandomAlgorithm, UpperConfidenceBound1, ThompsonSampling

if __name__ == "__main__":
    algorithm_class = ThompsonSampling
    gui = BarcelonaMABAlgorithmSimulatingGUI(simulate=False, algorithm_class=algorithm_class, algorithm_kwargs={})
    gui.menu_window_process()
