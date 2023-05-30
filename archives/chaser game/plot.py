import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from game import Game, N_DIMS, N_CHASERS
from strategy import *

radius = 0.2
chase_game = Game(0.5, circle_init_chasers(np.array([[0.25, 0.8]]), radius = radius), avoid_and_forward_runner_xdot, 
                  heat_seeking_chasers_xdot(foresight = 5))

fig, ax = plt.subplots()
n = 300
runner_positions = np.empty((n, N_DIMS))
chaser_positions = [np.empty((n, N_DIMS)) for _ in range(N_CHASERS)]
frame_count = n
for i in range(n):
    game_state = chase_game.one_step()
    runner_positions[i] = chase_game.runner_x
    for j in range(N_CHASERS):
        chaser_positions[j][i] = chase_game.chasers_x[j]
    if game_state.stop:
        print("The runner won" if game_state.win else "The chasers won")
        frame_count = i + 1
        break

def animate(i):
    ax.clear()
    plt_runner = plt.Circle(tuple(runner_positions[i]), 0.01, color = 'cyan')
    plt_chasers = [plt.Circle(tuple(chaser_positions[j][i]), 0.01, color='fuchsia') for j in range(N_CHASERS)]
    ax.add_patch(plt_runner)
    for j in range(N_CHASERS):
        ax.add_patch(plt_chasers[j])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

animation = FuncAnimation(fig, animate, frames = frame_count, interval = 1, repeat = False)
plt.show()