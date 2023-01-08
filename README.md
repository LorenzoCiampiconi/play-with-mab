[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Play with Mab

This project is built to explain the popular Reinforcement Learning framework of Multi Armed Bandit (MAB) with a easy UI.

You can either play the "MAB envinroment" yourself, by playing with the available arms, or simulate one of the available algorithm which are:

 - Thompson Sampling
 - Upper Confidence Bound 1 (tuned)
 - Epsilon-Greedy
 - Random Policy
 
Parameter updates of the algorithm will be shown at each time steps until convergence is reached. Currently there's a fixed time horizon of 100 steps.

## Setup
if you want you can setup a python virtual environment:
`pip -m venv play_with_mab_venv`

in case you did you should activate it:
`. play_with_mab_venv/bin/activate`

then

`pip install -r requirements.txt`

and just run the main

`python3 game_core/__main__.py`
