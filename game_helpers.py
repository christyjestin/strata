import numpy as np

def one_hot(num_classes, class_index):
    assert class_index < num_classes
    arr = np.zeros(num_classes)
    arr[class_index] = 1
    return arr

# outputs stabbing trajectory of shape num_timesteps x (2 * num_points) x 2
def stabbing_trajectory_maker(num_timesteps, length, width, num_points, separation):
    assert num_timesteps % 2 == 1, "the number of timesteps must be odd"
    # e.g. num_timesteps = 21, length = 10 yields 0, 1,..., 9, 10, 9,..., 1, 0
    points = np.linspace(0, length, (num_timesteps // 2) + 1)
    points_over_time = np.concatenate((points, points[-2::-1]))

    # expand to have multiple points per timestep
    points_per_timestep = np.arange(num_points) * separation
    x = points_over_time.reshape(-1, 1) + points_per_timestep.reshape(1, -1)
    y = np.array([width / 2, -width / 2]) # top and bottom edge
    # expand x and y to match
    output = np.dstack((np.repeat(x, 2, axis = 1), np.tile(y, (num_timesteps, num_points))))

    assert output.shape == (num_timesteps, 2 * num_points, 2)
    return output

# outputs sweeping circular trajectory of shape num_timesteps x num_points x 2
def circular_trajectory_maker(num_timesteps, radius, num_points, separation):
    # sweep from 0 degrees to 270 degrees
    angles_over_time = np.linspace(0, 3/2 * np.pi, num_timesteps)
    angles = np.array([np.roll(angles_over_time, -i * separation) for i in range(num_points)]).T
    output = radius * np.dstack((np.cos(angles), np.sin(angles)))

    assert output.shape == (num_timesteps, num_points, 2)
    return output