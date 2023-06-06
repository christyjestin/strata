import numpy as np
from model_constants import NUM_WEAPON_TOKENS

def one_hot(num_classes, class_index):
    assert class_index < num_classes
    arr = np.zeros(num_classes)
    arr[class_index] = 1
    return arr

def blast_trajectory_factory(duration, range, starting_offset, blast_radius):
    def func(player_position, player_angle):
        # the motion of the center of the blast is linear in the direction of player_angle, and the 
        # weapon tokens (i.e. blast_points) are simply a ring of points around the center points
        v = np.array([np.cos(player_angle), np.sin(player_angle)])
        blast_center = np.linspace(starting_offset, range, duration).reshape(-1, 1) * v
        th = np.linspace(0, 2 * np.pi, NUM_WEAPON_TOKENS, endpoint = False)
        blast_points = blast_radius * np.column_stack((np.cos(th), np.sin(th)))
        output: np.ndarray = np.expand_dims(blast_center, axis = 1) + np.expand_dims(blast_points, axis = 0)
        output += player_position

        assert output.shape == (duration, NUM_WEAPON_TOKENS, 2)
        return output

    return func


def scythe_trajectory_factory(duration, radius, blade_angle, sweep_angle):
    # the scythe is an arc of blade_angle degrees, and the weapon tokens (i.e. scythe_points) are points on this arc
    # the center of this arc has a rotation of sweep_angle degrees with the center of this sweep being player_angle
    scythe_center_angle = np.linspace(-sweep_angle / 2, sweep_angle / 2, duration)
    scythe_points = np.linspace(-blade_angle / 2, blade_angle / 2, NUM_WEAPON_TOKENS)
    angles = np.expand_dims(scythe_center_angle, axis = 1) + np.expand_dims(scythe_points, axis = 0)

    def func(player_position, player_angle):
        new_angles = player_angle + angles
        output: np.ndarray = radius * np.dstack((np.cos(new_angles), np.sin(new_angles)))
        output += player_position

        assert output.shape == (duration, NUM_WEAPON_TOKENS, 2)
        return output

    return func


def stabbing_trajectory_factory(duration, blade_length, blade_width, range, hilt_offset):
    assert NUM_WEAPON_TOKENS % 2 == 0, "the number of weapon tokens must be even for this function to work"
    assert duration % 2 == 1, "the duration must be odd for stabbing attacks because of how the motion works"

    # the blade extends out and then retracts
    # e.g. if hilt_offset = 1, duration = 11, blade_length = 9, and range = 15, then r = [1, 2,..., 5, 6, 5,..., 1]
    r = np.linspace(hilt_offset, range - blade_length, duration // 2, endpoint = False)
    hilt_radius = np.concatenate((r, np.array([range - blade_length]), r[::-1]))

    # linspace from 0 to blade_length but not including 0; the reason we're excluding 0 is that we only get 3
    # points to represent the sword, and points at the hilt aren't very useful anyway since they're unlikely to hit
    blade_point_radius = -np.linspace(-blade_length, 0, NUM_WEAPON_TOKENS // 2, endpoint = False)[::-1]
    radii = np.expand_dims(hilt_radius, axis = 1) + np.expand_dims(blade_point_radius, axis = 0)
    radii = np.expand_dims(radii, axis = 2)

    # the points representing the blade are a box around the edge of the blade, so we
    # compute the centerline and then extend perpendicularly to form the box
    def func(player_position, player_angle):
        center_line_points = radii * np.array([np.cos(player_angle), np.sin(player_angle)])
        perp_angle = player_angle + np.pi / 2
        perp_offset = blade_width / 2 * np.array([np.cos(perp_angle), np.sin(perp_angle)])
        output: np.ndarray = np.concatenate((center_line_points + perp_offset, center_line_points - perp_offset), axis = 1)
        output += player_position

        assert output.shape == (duration, NUM_WEAPON_TOKENS, 2)
        return output

    return func
