import numpy as np
from collections import namedtuple

GameState = namedtuple('GameState', ['stop', 'win'])

TIMESTEPS_PER_SEC = 30
TIMESTEP_LENGTH = 1 / TIMESTEPS_PER_SEC
N_DIMS = 2
N_CHASERS = 1

CHASER_XDOT_LIMIT = 0.5
RUNNER_XDOT_LIMIT = CHASER_XDOT_LIMIT * 1.5

class Game:
    player_radius = 0.01
    box_length = 1

    def __init__(
            self, 
            box_width, 
            init_chasers, 
            compute_runner_xdot, 
            compute_chasers_xdot
        ):
        self.box_width = box_width
        self.xmin = self.player_radius
        self.xmax = self.box_width - self.player_radius

        self.ymin = self.player_radius
        self.ymax = self.box_length - self.player_radius

        self.runner_x = np.array([[self.box_width / 2, self.player_radius]])
        self.runner_xdot = np.zeros((1, N_DIMS))
        self.chasers_x = init_chasers(self.box_length, box_width)
        assert self.chasers_x.shape == (N_CHASERS, N_DIMS)
        self.chasers_xdot = np.zeros((N_CHASERS, N_DIMS))
        self.compute_runner_xdot = compute_runner_xdot
        self.compute_chasers_xdot = compute_chasers_xdot
        self.chasers_state = None

    def one_step(self):
        # compute new xdot and update x
        args = (self.runner_x, self.runner_xdot, self.chasers_x, self.chasers_xdot)
        self.runner_xdot = self.compute_runner_xdot(*args)
        self.chasers_xdot, self.chasers_state = self.compute_chasers_xdot(*args, self.chasers_state)
        self.runner_x += self.runner_xdot * TIMESTEP_LENGTH
        self.chasers_x += self.chasers_xdot * TIMESTEP_LENGTH

        # check for win
        if self.runner_x[0, 1] >= self.ymax:
            return GameState(stop = True, win = True)

        # check for loss
        closest_chaser_dist = np.min(np.linalg.norm(self.runner_x - self.chasers_x, axis = 1)) 
        if closest_chaser_dist < 2 * self.player_radius:
            return GameState(stop = True, win = False)

        # clip x and y values
        self.runner_x[:, 0] = np.clip(self.runner_x[:, 0], self.xmin, self.xmax)
        self.chasers_x[:, 0] = np.clip(self.chasers_x[:, 0], self.xmin, self.xmax)
        self.runner_x[:, 1] = np.clip(self.runner_x[:, 1], self.ymin, self.ymax)
        self.chasers_x[:, 1] = np.clip(self.chasers_x[:, 1], self.ymin, self.ymax)

        # handle collisions between chasers
        # yields n by n by 2 matrix of distance vectors between chasers
        diff_matrix = np.expand_dims(self.chasers_x, axis = 1) - self.chasers_x
        # index i,j contains the L2 distance between chaser i and chaser j
        dist_matrix = np.linalg.norm(diff_matrix, axis = 2)
        # add large value along diagonal to avoid swapping with self
        dist_matrix += np.eye(N_CHASERS) * 3 * self.player_radius
        for i, j in list(zip(*np.where(dist_matrix < 2 * self.player_radius))):
            # avoid swapping twice (i > j)
            if i < j:
                # elastic collision between objects of equal mass just swaps velocities
                self.chasers_xdot[i], self.chasers_xdot[j] = self.chasers_xdot[j], self.chasers_xdot[i]
                self.chasers_x[i] += self.chasers_xdot[i] * TIMESTEP_LENGTH
                self.chasers_x[j] += self.chasers_xdot[j] * TIMESTEP_LENGTH
        return GameState(stop = False, win = True)
