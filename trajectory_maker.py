import numpy as np
import math

def stabbing_trajectory_maker(num_timesteps, length, width, numpoints, separation):
    
    points = np.linspace(0,1,num_timesteps+1)*length

    points = list(points)
    points += points[::-1]
    points = np.array(points)

    y_offset_high = np.zeros_like(points) + width / 2
    y_offset_low = np.zeros_like(points) - width / 2

    pointshigh = np.vstack((points, y_offset_high)).T
    pointslow = np.vstack((points, y_offset_low)).T
    
    allPoints = [pointshigh, pointslow]

    for i in range (numpoints - 1):
        newpointshigh = np.vstack((points + (i+1)*separation, y_offset_high)).T
        newpointslow = np.vstack((points + (i+1)*separation, y_offset_low)).T
        allPoints.append(newpointshigh)
        allPoints.append(newpointslow)


    return np.array(allPoints)


def circular_trajectory_maker(num_timesteps, radius, numpoints, separation):
    angles = np.linspace(0, np.pi, num_timesteps + 1)
    #print(angles)

    allPoints = []

    for i in range(numpoints):
        shifted_angles = np.roll(angles, i * separation)  # Apply the delay effect by rolling the angles
        #print(shifted_angles)
        x_shifted = radius * np.cos(shifted_angles)
        y_shifted = radius * np.sin(shifted_angles)
        newpoints = np.vstack((x_shifted, y_shifted)).T
        allPoints.append(newpoints)

    return np.array(allPoints)






print(circular_trajectory_maker(10, 1, 2, -1)[:,0])


