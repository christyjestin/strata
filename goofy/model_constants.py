from collections import namedtuple

# position (2), orientation as sin and cos (2), cooldown timers for shield and blast attack (2), one hot of mode (3)
NUM_PLAYER_MODES = 3
NUM_TIMERS = 2
PLAYER_TOKEN_LENGTH = 2 + 2 + NUM_TIMERS + NUM_PLAYER_MODES
PLAYER_L2_WEIGHT = [1, 1, 10, 10, 5, 5]
assert len(PLAYER_L2_WEIGHT) == PLAYER_TOKEN_LENGTH - NUM_PLAYER_MODES

# position (2), damage (1), one hot of type (3)
NUM_WEAPON_TYPES = 3
WEAPON_TOKEN_LENGTH = 3 + NUM_WEAPON_TYPES
WEAPON_L2_WEIGHT = [1, 1, 1]
assert len(WEAPON_L2_WEIGHT) == WEAPON_TOKEN_LENGTH - NUM_WEAPON_TYPES

NUM_WEAPON_TOKENS = 6

# the indices to keep when trimming a player token to be the same length as the weapon token
PLAYER_TRIM_INDICES = [0, 1, 4, 5, 6, 7] # 7 is a placeholder
assert len(PLAYER_TRIM_INDICES) == WEAPON_TOKEN_LENGTH, "trim indices must match weapon token length"
assert len(PLAYER_TRIM_INDICES) <= PLAYER_TOKEN_LENGTH, "trim indices must be compatible with player token length"

NUM_ATTACKS = 3
# shield(1), do nothing(1)
NUM_ACTIONS = NUM_ATTACKS + 1 + 1
# change in position (2), change in orientation as sin and cos (2)
NUM_MOVEMENT_VALS = 2 + 2
ACTION_DIM = NUM_ACTIONS + NUM_MOVEMENT_VALS

# 2 parameters for each beta dist and 3 beta dists for x, y, theta
# we'll use a mixture model with each set of 6 values parametrizing one of the movement dists in the mixture
NUM_PARAMETERS_PER_MOVEMENT_DIST = 6

# first value is health
STATE_SPLIT = [1, PLAYER_TOKEN_LENGTH, NUM_WEAPON_TOKENS * WEAPON_TOKEN_LENGTH]
STATE_DIM = 2 * sum(STATE_SPLIT)
HERO_HEALTH_INDEX = 0
ADVERSARY_HEALTH_INDEX = sum(STATE_SPLIT)

STRATEGY_DIM = 20

TIME_HORIZON = 20 # TODO: change after discussing with Jacob

EvolverLoss = namedtuple('EvolverLoss', ['player_token', 'weapon_token', 'opponent_health'])

ActionPolicy = namedtuple('ActionPolicy', ['logits', 'beta_parameters'])

SEARCH_MODE = 'search_mode'
BACKPROP_MODE = 'backprop_mode'

# minimum value for beta parameters since alpha and beta must be positive
BETA_EPSILON = 1e-3

# TODO: replace placeholder values
# movement limits
DX_LIMIT = 0.5
DY_LIMIT = 0.5
DTHETA_LIMIT = 0.5