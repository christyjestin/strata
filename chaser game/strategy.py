import numpy as np
from game import TIMESTEP_LENGTH, N_CHASERS, CHASER_XDOT_LIMIT, RUNNER_XDOT_LIMIT

def rescale(x_dot, is_chasers):
    limit = CHASER_XDOT_LIMIT if is_chasers else RUNNER_XDOT_LIMIT
    return x_dot / np.linalg.norm(x_dot, axis = 1, keepdims = True) * limit

predict_runner_x = lambda runner_x, runner_xdot, foresight: runner_x + TIMESTEP_LENGTH * runner_xdot * foresight

def spread_three_quarter_line_init_chasers(box_length, box_width):
    return np.array([[(0.5 + i) / N_CHASERS * box_width, 0.75 * box_length] for i in range(N_CHASERS)])

def heat_seeking_chasers_xdot(foresight = 1):
    def compute_x_dot(runner_x, runner_xdot, chasers_x, chasers_xdot, state):
        runner_next_x = predict_runner_x(runner_x, runner_xdot, foresight)
        chasers_xdot = runner_next_x - chasers_x
        return rescale(chasers_xdot, is_chasers = True), state
    return compute_x_dot

def random_runner_xdot(runner_x, runner_xdot, chasers_x, chasers_xdot):
    return rescale(np.array([[np.random.rand() * 2 - 1, 1]]), is_chasers = False)

def avoid_and_forward_runner_xdot(runner_x, runner_xdot, chasers_x, chasers_xdot):
    dists = np.linalg.norm(runner_x - chasers_x, axis = 1)
    closest_chaser_index = np.argmin(dists)
    x_diff = (runner_x[0, 0] - chasers_x[closest_chaser_index, 0])
    if x_diff == 0:
        x = -10 if np.random.rand() > 0.5 else 10
    else:
        x = 0.03 / (runner_x[0, 0] - chasers_x[closest_chaser_index, 0])
    return rescale(np.array([[x, 1]]), is_chasers = False)

def circle_init_chasers(center, radius):
    def init_chasers(box_length, box_width):
        thetas = (np.arange(N_CHASERS) / N_CHASERS) * 2 * np.pi
        circle = np.array([[np.cos(th), np.sin(th)] for th in thetas]) * radius
        return circle + center
    return init_chasers

def box_chasers_xdot(foresight, radius):
    heat_seeking_compute = heat_seeking_chasers_xdot(foresight = foresight)
    def compute_chasers_xdot(runner_x, runner_xdot, chasers_x, chasers_xdot, state):
        # init state
        if state == None:
            state = {'is_heat_seeking': False}

        if state['is_heat_seeking']:
            return heat_seeking_compute(runner_x, runner_xdot, chasers_x, chasers_xdot, state)
        else:
            centroid = np.mean(chasers_x, axis = 0)
            if np.linalg.norm(runner_x - centroid) < radius:
                state['is_heat_seeking'] = True
                return heat_seeking_compute(runner_x, runner_xdot, chasers_x, chasers_xdot, state)
            runner_next_x = predict_runner_x(runner_x, runner_xdot, foresight)
            chasers_xdot = np.tile(runner_next_x - centroid, (N_CHASERS, 1))
            return rescale(chasers_xdot, is_chasers = True), state
    return compute_chasers_xdot
