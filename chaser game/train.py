from game import Game
import numpy as np
from strategy import *

box_width = 0.5
n_chasers = 3
chase_game = Game(box_width, n_chasers, spread_three_quarter_line_init_chasers, 
                  random_runner_xdot, heat_seeking_chasers_xdot())
while True:
    game_state = chase_game.one_step()
    if game_state.stop:
        break