from game_core.gui.simulating_gui import BarcelonaMABAlgorithmSimulatingGUI
from game_core.simulation.algorithm import RandomAlgorithm, UpperConfidenceBound1

if __name__ == "__main__":
    algorithm_class = UpperConfidenceBound1
    gui = BarcelonaMABAlgorithmSimulatingGUI(simulate=False, algorithm_class=algorithm_class, algorithm_kwargs={})
    gui.play_window_process()
