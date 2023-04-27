# position (2), orientation as sin and cos (2), cooldown timers for shield and blast attack (2), one hot of mode (3)
NUM_MODES = 3
NUM_TIMERS = 2
PLAYER_TOKEN_LENGTH = 2 + 2 + NUM_TIMERS + NUM_MODES

# position (2), damage (1), one hot of type (3)
NUM_WEAPON_TYPES = 3
WEAPON_TOKEN_LENGTH = 3 + NUM_WEAPON_TYPES

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