from collections import namedtuple
from enum import Enum

GameState = namedtuple('GameState', ['stop', 'win'])


TIMESTEPS_PER_SEC = 30
TIMESTEP_LENGTH = 1 / TIMESTEPS_PER_SEC
N_DIMS = 2

CHASER_XDOT_LIMIT = 0.5
RUNNER_XDOT_LIMIT = CHASER_XDOT_LIMIT * 1.5

class Mode(Enum):
    NORMAL = 1
    ATTACKING = 2
    SHIELDING = 3
    STUNNED = 4

class Game:
    arena_length = 100
    arena_width = 100

    # im just gonna make comments for features to add

    def __init__(self, hero, adversary, round_length):
        self.hero = hero
        self.adversary = adversary
        self.round_length = round_length
        self.time_left = round_length

        # let both players access the position and health of the other
        self.hero.connect(adversary)
        self.adversary.connect(hero)


    def one_step(self):
        # maybe just let them have modes like shield raised and trust each agent to correctly do damage on the other     like a state?
        self.hero.one_step()
        self.adversary.one_step()

        # check for win
        if self.adversary.health <= 0:
            return GameState(stop = True, win = True)

        # check for loss
        if self.hero.health <= 0:
            return GameState(stop = True, win = False)
        return GameState(stop = False, win = True)



class Hero:
    # keep it square?
    # Might want to make bosses nonsquare!!
    # oh right this is not the arena
    def __init__(self, hitbox_width, hitbox_height, movement_speed, rotation_speed, max_health, position):
        self.hitbox_width = hitbox_width
        self.hitbox_height = hitbox_height
        self.max_health = max_health
        self.health = max_health
        self.position = position
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed



class Adversary:
    def __init__(self, hitbox_width, hitbox_height, movement_speed, rotation_speed, max_health, position):
        self.hitbox_width = hitbox_width
        self.hitbox_height = hitbox_height
        self.max_health = max_health
        self.health = max_health
        self.position = position
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed


