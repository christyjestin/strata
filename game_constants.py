from collections import namedtuple
from enum import Enum
from model_constants import *

ARENA_LENGTH = 1920
ARENA_WIDTH = 960

GameState = namedtuple('GameState', ['game_over', 'won', 'time_left'])
Hitbox = namedtuple('Hitbox', ['width', 'height'], defaults = [32, 32])

TIMESTEPS_PER_SEC = 30
TIMESTEP_LENGTH = 1 / TIMESTEPS_PER_SEC

DONE = 'DONE'
LEGAL_ACTION = 'LEGAL ACTION'
ILLEGAL_ACTION = 'ILLEGAL_ACTION'

class Mode(Enum):
    NORMAL_MODE = 0
    ATTACK_MODE = 1
    SHIELD_MODE = 2

class AttackType(Enum):
    STAB_ATTACK = 0
    SWEEP_ATTACK = 1
    BLAST_ATTACK = 2

assert len(Mode) == NUM_PLAYER_MODES
assert len(AttackType) == NUM_ATTACKS