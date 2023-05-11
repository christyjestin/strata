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
assert len(PLAYER_TRIM_INDICES) == WEAPON_TOKEN_LENGTH, "please update trim indices to match player/weapon token lengths"
assert len(PLAYER_TRIM_INDICES) <= PLAYER_TOKEN_LENGTH, "please update trim indices to match player/weapon token lengths"

NUM_ATTACKS = 3
# shield(1), do nothing(1)
NUM_ACTIONS = NUM_ATTACKS + 1 + 1
# change in position (2), change in orientation (2)
ACTION_DIM = 2 + 2 + NUM_ACTIONS

# first value is health
STATE_SPLIT = [1, PLAYER_TOKEN_LENGTH, NUM_WEAPON_TOKENS * WEAPON_TOKEN_LENGTH]
STATE_DIM = 2 * sum(STATE_SPLIT)

STRATEGY_DIM = 20

TIME_HORIZON = 20 # TODO: change after discussing with Jacob

EvolverLoss = namedtuple('EvolverLoss', ['player_token', 'weapon_token', 'opponent_health'])

SearchPolicy = namedtuple('SearchPolicy', ['logits', 'parameters'])

SEARCH_MODE = 'search_mode'
BACKPROP_MODE = 'backprop_mode'