from collections import namedtuple
from enum import Enum
import numpy as np
import math

GameState = namedtuple('GameState', ['stop', 'win'])

TIMESTEPS_PER_SEC = 30
TIMESTEP_LENGTH = 1 / TIMESTEPS_PER_SEC

class Mode(Enum):
    NORMAL = 1
    ATTACKING = 2
    SHIELDING = 3
    # STUNNED = 4

class Attacks(Enum):
    STAB = 1
    SWEEP = 2
    BLAST = 3

def stabbing_trajectory_maker(num_timesteps, length, width, numpoints, separation):
    points = np.linspace(0, 1, int(num_timesteps / 2) + 1) * length
    points = list(points)
    points += points[::-1]
    points = np.array(points)

    y_offset_high = np.zeros_like(points) + width / 2
    y_offset_low = np.zeros_like(points) - width / 2

    pointshigh = np.vstack((points, y_offset_high)).T
    pointslow = np.vstack((points, y_offset_low)).T
    allPoints = [pointshigh, pointslow]

    for i in range (numpoints - 1):
        newpointshigh = np.vstack((points + (i + 1) * separation, y_offset_high)).T
        newpointslow = np.vstack((points + (i + 1) * separation, y_offset_low)).T
        allPoints.append(newpointshigh)
        allPoints.append(newpointslow)
    return np.array(allPoints)

def circular_trajectory_maker(num_timesteps, radius, numpoints, separation):
    angles = np.linspace(0, 3/2 * np.pi, num_timesteps + 1)
    allPoints = []
    for i in range(numpoints):
        shifted_angles = np.roll(angles, -i * separation) 
        x_shifted = radius * np.cos(shifted_angles)
        y_shifted = radius * np.sin(shifted_angles)
        newpoints = np.vstack((x_shifted, y_shifted)).T
        allPoints.append(newpoints)
    return np.array(allPoints)

class Attack:
    def __init__(self, damage, length, trajectory):
        self.damage = damage
        self.length = length
        self.time_step = 0
        self.trajectory = trajectory
        self.position = trajectory[:, 0]

    def one_step(self):
        # Attack ends
        if self.time_step == self.length:
            self.time_step = 0
            self.position = self.trajectory[:, 0]
            return True

        # Else Attack follows trajectory
        self.time_step += 1
        self.position = self.trajectory[:, self.time_step]
        return False


STAB_TRAJECTORY = stabbing_trajectory_maker(50, 30, 6, 8, 5)
SWEEP_TRAJECTORY = circular_trajectory_maker(75, 80, 10, 3)
BLAST_TRAJECTORY = stabbing_trajectory_maker(120, 1200, 20, 5, 5)

STAB_ATTACK = Attack(0.1, 50, STAB_TRAJECTORY)
BLAST_ATTACK = Attack(5, 60, BLAST_TRAJECTORY)
SWEEP_TRAJECTORY = Attack(0.2, 50, SWEEP_TRAJECTORY)

class Game:
    arena_length = 1920
    arena_width = 960

    def __init__(self, hero, adversary, round_length):
        self.hero = hero
        self.adversary = adversary
        self.round_length = round_length
        self.time_left = round_length

        # Gamestate vector is length of round, initially empty
        # total length = round_length ( in seconds) * TIMESTEPS_PER_SEC
        self.game_state_list = []


    def one_step(self):
        self.hero.one_step()
        self.adversary.one_step()
        self.calculate_damage()

        # check for win
        if self.adversary.health <= 0:
            return GameState(stop = True, win = True)
        # check for loss
        if self.hero.health <= 0:
            return GameState(stop = True, win = False)
        return GameState(stop = False, win = True)

# Damage calculation is currently a radial distance estimate
    def calculate_damage(self):
        hero_pos = self.hero.position
        adversary_pos = self.adversary.position

        hero_distance = np.linalg.norm(hero_pos - self.adversary.damagepoints, axis=1)
        adversary_distance = np.linalg.norm(adversary_pos - self.hero.damagepoints, axis=1)
        print(adversary_distance)
        # Hero takes damage
        if (np.min(hero_distance) < self.hero.hitbox_width - 15):
            self.hero.health -= self.adversary.damage
        # Adversary takes damage
        if (np.min(adversary_distance) < self.adversary.hitbox_width - 15):
            self.adversary.health -= self.hero.damage
            print(self.hero.health)

# ALL HITBOXES ARE ADDED TO PLAYER POSITION! so if players position 
# is [0,0] hitbox extends from [0,0] to [width, height]
class Player:
    mode = Mode.NORMAL

    def __init__(self, hitbox, max_movement_speed, max_rotation_speed, max_health, position, theta, arena_width, 
                 arena_length, is_hero = False):
        self.hitbox_width = hitbox['width']
        self.hitbox_height = hitbox['height']
        self.max_health = max_health
        self.health = max_health
        self.position = position
        self.theta = theta
        self.max_movement_speed = max_movement_speed
        self.max_rotation_speed = max_rotation_speed
        self.shieldcooldown = 0
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.mode = Mode.NORMAL

        self.Attack = None
        self.damagepoints = np.array([self.position])
        self.damage = 0
        self.is_hero = is_hero

    def one_step(self):
        # Player chooses movement and attack/defensive action
        # self.position = self.position + (np.random.rand(2) - 0.5)
        if not self.is_hero:
            self.theta = self.theta + 1

        if self.theta > 360:
            self.theta -= 360

        if self.theta < -360:
            self.theta += 360

        theta_radians = np.pi* self.theta / (180)
        rot = np.array([[math.cos(-theta_radians), -math.sin(-theta_radians)], 
                        [math.sin(-theta_radians), math.cos(-theta_radians)]])


        # Mode Stuff
        if self.mode == Mode.ATTACKING:
            self.damage = self.Attack.damage
            damagepoints = self.Attack.position
            damagepoints = (rot @ damagepoints.T).T

            self.damagepoints = self.position + damagepoints +  0 # potentially add offset

            done_attacking = self.Attack.one_step()

            if done_attacking:
                self.mode = Mode.NORMAL
                self.damagepoints = np.array([self.position])

        if self.mode == Mode.NORMAL:
            self.damagepoints = np.array([self.position])
            self.damage = 0

            # if choose to attack....
            self.Attack = STAB_ATTACK # model input
            self.mode = Mode.ATTACKING

            # if choose to shield...
            # self.mode = Mode.SHIELDING


Default_Hitbox = {
    'width': 32,
    'height': 32
}

hero = Player(Default_Hitbox, 1, 1, 100, np.array([600, 300]), 0, 1920, 960, True)
adversary = Player(Default_Hitbox, 1, 1, 100, np.array([600, 600]), 0, 1920, 960)
game = Game(hero, adversary, 10)

game.one_step()
game.one_step()