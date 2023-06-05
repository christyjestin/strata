import numpy as np
import torch

from game_constants import *
from model_constants import *
from game_helpers import *

class Attack:
    def __init__(self, attack_type: AttackType, damage, duration, trajectory):
        self.attack_type = attack_type
        self.damage = damage
        self.duration = duration
        self.time_step = 0
        self.trajectory = trajectory
        self.position = self.trajectory[self.time_step]

    def one_step(self):
        self.time_step += 1
        # end attack when timestep has reached duration
        if self.time_step == self.duration:
            return DONE
        self.position = self.trajectory[self.time_step]

    def restart(self):
        self.time_step = 0
        self.position = self.trajectory[self.time_step]

class Player:
    def __init__(self, hitbox: Hitbox, starting_health, position, theta, stab_attack, blast_attack, sweep_attack, 
                 shield_cooldown_time, blast_cooldown_time, shield_duration):
        assert isinstance(position, np.ndarray) and position.shape == (2,), "position must be a numpy array of length 2"
        assert (0 <= position[0] < ARENA_LENGTH) and (0 <= position[1] < ARENA_WIDTH), \
                                                                    "position must be within the arena map"
        assert isinstance(theta, float) and 0 <= theta < 2 * np.pi, "theta must be in the range [0, 2pi)"

        self.hitbox = hitbox
        self.health = starting_health
        self.position = position
        self.theta = theta
        self.mode = Mode.NORMAL_MODE

        # cooldown timers
        self.shield_cooldown_timer = 0
        self.blast_cooldown_timer = 0
        self.shield_cooldown_time = shield_cooldown_time
        self.blast_cooldown_time = blast_cooldown_time

        # shield variables
        self.shield_time_left = 0
        self.shield_duration = shield_duration

        # attack variables
        self.attack: Attack = None
        self.attack_map = {
            AttackType.STAB_ATTACK: stab_attack,
            AttackType.BLAST_ATTACK: blast_attack,
            AttackType.SWEEP_ATTACK: sweep_attack
        }
        assert len(self.attack_map.keys()) == len(AttackType)

    def connect_opponent(self, opponent):
        self.opponent = opponent

    def one_step(self, action):
        # update timers
        if self.shield_cooldown_timer != 0:
            self.shield_cooldown_timer -= 1
        if self.blast_cooldown_timer != 0:
            self.blast_cooldown_timer -= 1

        # update modes
        if self.mode == Mode.SHIELD_MODE:
            self.shield_time_left -= 1
            # turn shield off and return to normal mode
            if self.shield_time_left == 0:
                self.shield_cooldown_timer = self.shield_cooldown_time
                self.mode == Mode.NORMAL_MODE
        elif self.mode == Mode.ATTACK_MODE:
            # end attack and return to normal mode if done - otherwise progress the attack
            if self.attack.one_step() == DONE:
                # set cooldown timer if the completed attack was a blast attack
                if self.attack.attack_type == AttackType.BLAST_ATTACK:
                    self.blast_cooldown_timer = self.blast_cooldown_time
                self.mode == Mode.NORMAL_MODE
                self.attack = None

        # process movement
        dx, dy, sin, cos = action[-NUM_MOVEMENT_VALS:]
        self.position += np.array([dx, dy])
        self.theta += np.arctan2(sin, cos)
        self.theta = self.theta % (2 * np.pi)

        # process action
        action_type = np.argmax(action[:NUM_ACTIONS])
        is_legal_action = True
        # must be in normal mode to start shielding or attacking
        if self.mode != Mode.NORMAL_MODE:
            if action_type != DO_NOTHING_INDEX:
                is_legal_action = False
        else:
            if action_type < NUM_ATTACKS:
                self.mode = Mode.ATTACK_MODE
                self.attack: Attack = self.attack_map[AttackType(action_type)]
                self.attack.restart()
            elif action_type == SHIELD_INDEX:
                self.mode == Mode.SHIELD_MODE
                self.shield_time_left = self.shield_duration

        self.opponent.calculate_damage() # calculate damage for opponent
        return LEGAL_ACTION if is_legal_action else ILLEGAL_ACTION

    def calculate_damage(self):
        pass

    def pack_player_state(self) -> torch.Tensor:
        health = torch.tensor([self.health])

        # construct player token, which is made up of position (2), orientation as sin and cos (2), 
        # cooldown timers for shield and blast attack (2), one hot of mode (3)
        orientation = np.array([np.cos(self.theta), np.sin(self.theta)])
        timers = np.array([self.shield_cooldown_timer, self.blast_cooldown_timer])
        mode = one_hot(num_classes = NUM_PLAYER_MODES, class_index = self.mode.value)
        player_token = np.concatenate([self.position, orientation, timers, mode])

        # construct weapon tokens
        if self.mode == Mode.ATTACK_MODE:
            # weapon token is position (2), damage (1), one hot of type (3)
            damage = np.array([self.attack.damage])
            attack_type = one_hot(num_classes = NUM_ATTACKS, class_index = self.attack.attack_type.value)
            weapon_info = np.tile(np.concatenate((damage, attack_type)), (NUM_WEAPON_TOKENS, 1))
            weapon_tokens = np.concatenate((self.attack.position, weapon_info), axis = 1).flatten()
        else:
            weapon_tokens = np.zeros(NUM_WEAPON_TOKENS * WEAPON_TOKEN_LENGTH)

        output = torch.cat((health, player_token, weapon_tokens))
        assert output.shape == (sum(STATE_SPLIT),)
        return output

class Game:
    def __init__(self, hero: Player, adversary: Player, round_length) -> None:
        self.hero = hero
        self.adversary = adversary
        self.hero.connect_opponent(adversary)
        self.adversary.connect_opponent(hero)
        self.time_left = round_length

    def tick(self):
        self.time_left -= 1
        # check for time running out
        if self.time_left == 0:
            health_diff = self.hero.health - self.adversary.health
            return GameState(stop = True, win = (health_diff >= 0), time_left = self.time_left)
        # check for win
        if self.adversary.health <= 0:
            return GameState(stop = True, win = True, time_left = self.time_left)
        # check for loss
        if self.hero.health <= 0:
            return GameState(stop = True, win = False, time_left = self.time_left)
        return GameState(stop = False, win = True, time_left = self.time_left)

    def pack_game_state(self, for_adversary: bool):
        a = self.hero.pack_player_state()
        b = self.adversary.pack_player_state()
        return torch.cat((b, a)) if for_adversary else torch.cat((a, b))
