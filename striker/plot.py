import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from constants import *
from game import Fight

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
ALPHA = np.linspace(0, 1, 10)

# box must be formatted as bottom 4 corners and corresponding top 4 corners in same order
def plot_box(ax, box, color = 'gray'):
    assert box.shape == (8, 3)
    for i, j in BOX_EDGES:
        line = np.outer(ALPHA, box[i]) + np.outer(1 - ALPHA, box[j])
        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color)

def plot_line(ax, a, b, color = 'gray'):
    line = np.outer(ALPHA, a.flatten()) + np.outer(1 - ALPHA, b.flatten())
    ax.plot3D(line[:, 0], line[:, 1], line[:, 2], color)

def plot_ring(ax, color = 'black'):
    for th in (2 * np.pi * np.arange(8) / 8):
        zspace = np.linspace(0, RING_HEIGHT, 10)
        ax.plot3D(np.tile(np.sin(th) * RING_RADIUS, 10), np.tile(np.cos(th) * RING_RADIUS, 10), zspace, color)

n = 360
fight = Fight(n, RIGHTY, RIGHTY)
fig = plt.figure()
ax = plt.axes(projection='3d')
box_positions_red = []
head_positions_red = []
box_positions_blue = []
head_positions_blue = []

right_knee_positions_red = []
right_foot_positions_red = []
left_knee_positions_red = []
left_foot_positions_red = []
right_hip_positions_red = []
left_hip_positions_red = []

right_knee_positions_blue = []
right_foot_positions_blue = []
left_knee_positions_blue = []
left_foot_positions_blue = []
right_hip_positions_blue = []
left_hip_positions_blue = []

right_elbow_positions_red = []
right_hand_positions_red = []
left_elbow_positions_red = []
left_hand_positions_red = []
right_shoulder_positions_red = []
left_shoulder_positions_red = []

right_elbow_positions_blue = []
right_hand_positions_blue = []
left_elbow_positions_blue = []
left_hand_positions_blue = []
right_shoulder_positions_blue = []
left_shoulder_positions_blue = []

for _ in range(n):
    box_positions_red.append(fight.a.torso_vertices())
    head_positions_red.append(fight.a.head_vertices())
    box_positions_blue.append(fight.b.torso_vertices())
    head_positions_blue.append(fight.b.head_vertices())

    right_knee_positions_red.append(fight.a.right_knee)
    right_foot_positions_red.append(fight.a.right_foot)
    left_knee_positions_red.append(fight.a.left_knee)
    left_foot_positions_red.append(fight.a.left_foot)
    right_hip_positions_red.append(fight.a.right_hip)
    left_hip_positions_red.append(fight.a.left_hip)

    right_knee_positions_blue.append(fight.b.right_knee)
    right_foot_positions_blue.append(fight.b.right_foot)
    left_knee_positions_blue.append(fight.b.left_knee)
    left_foot_positions_blue.append(fight.b.left_foot)
    right_hip_positions_blue.append(fight.b.right_hip)
    left_hip_positions_blue.append(fight.b.left_hip)

    right_elbow_positions_red.append(fight.a.right_elbow)
    right_hand_positions_red.append(fight.a.right_hand)
    left_elbow_positions_red.append(fight.a.left_elbow)
    left_hand_positions_red.append(fight.a.left_hand)
    right_shoulder_positions_red.append(fight.a.right_shoulder)
    left_shoulder_positions_red.append(fight.a.left_shoulder)

    right_elbow_positions_blue.append(fight.b.right_elbow)
    right_hand_positions_blue.append(fight.b.right_hand)
    left_elbow_positions_blue.append(fight.b.left_elbow)
    left_hand_positions_blue.append(fight.b.left_hand)
    right_shoulder_positions_blue.append(fight.b.right_shoulder)
    left_shoulder_positions_blue.append(fight.b.left_shoulder)
    fight.one_step()

def animate(i):
    ax.clear()
    plot_box(ax, box_positions_red[i], color = 'red')
    plot_box(ax, head_positions_red[i], color = 'red')
    plot_box(ax, box_positions_blue[i], color = 'blue')
    plot_box(ax, head_positions_blue[i], color = 'blue')
    plot_ring(ax)

    plot_line(ax, right_hip_positions_red[i].flatten(), right_knee_positions_red[i].flatten(), 'red')
    plot_line(ax, right_knee_positions_red[i].flatten(), right_foot_positions_red[i].flatten(), 'red')
    plot_line(ax, left_hip_positions_red[i].flatten(), left_knee_positions_red[i].flatten(), 'red')
    plot_line(ax, left_knee_positions_red[i].flatten(), left_foot_positions_red[i].flatten(), 'red')

    plot_line(ax, right_hip_positions_blue[i].flatten(), right_knee_positions_blue[i].flatten(), 'blue')
    plot_line(ax, right_knee_positions_blue[i].flatten(), right_foot_positions_blue[i].flatten(), 'blue')
    plot_line(ax, left_hip_positions_blue[i].flatten(), left_knee_positions_blue[i].flatten(), 'blue')
    plot_line(ax, left_knee_positions_blue[i].flatten(), left_foot_positions_blue[i].flatten(), 'blue')

    plot_line(ax, right_shoulder_positions_red[i].flatten(), right_elbow_positions_red[i].flatten(), 'red')
    plot_line(ax, right_elbow_positions_red[i].flatten(), right_hand_positions_red[i].flatten(), 'red')
    plot_line(ax, left_shoulder_positions_red[i].flatten(), left_elbow_positions_red[i].flatten(), 'red')
    plot_line(ax, left_elbow_positions_red[i].flatten(), left_hand_positions_red[i].flatten(), 'red')

    plot_line(ax, right_shoulder_positions_blue[i].flatten(), right_elbow_positions_blue[i].flatten(), 'blue')
    plot_line(ax, right_elbow_positions_blue[i].flatten(), right_hand_positions_blue[i].flatten(), 'blue')
    plot_line(ax, left_shoulder_positions_blue[i].flatten(), left_elbow_positions_blue[i].flatten(), 'blue')
    plot_line(ax, left_elbow_positions_blue[i].flatten(), left_hand_positions_blue[i].flatten(), 'blue')
    ax.set_box_aspect([2 * RING_RADIUS, 2 * RING_RADIUS, RING_HEIGHT])

animation = FuncAnimation(fig, animate, frames = n, interval = 1, repeat = True)
plt.show()