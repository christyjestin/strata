import numpy as np
import torch
from typing import Tuple

from game_constants import *
from model_constants import *
from game_helpers import *

class Attack:
    def __init__(self, attack_type: AttackType, damage, duration, trajectory_maker, moves_with_player):
        assert duration > 0, "duration must be positive"
        self.attack_type = attack_type
        self.damage = damage
        self.duration = duration
        self.time_step = 0
        self.trajectory_maker = trajectory_maker
        self.moves_with_player = moves_with_player # e.g. blast attack doesn't move with player

    def one_step(self, player_position, player_angle):
        # need to update trajectory if the attack moves with the player
        if self.moves_with_player:
            self.trajectory = self.trajectory_maker(player_position, player_angle)
        self.position = self.trajectory[self.time_step]

        self.time_step += 1
        return DONE if (self.time_step == self.duration) else INCOMPLETE

    def restart(self, player_position, player_angle):
        self.time_step = 0
        self.trajectory = self.trajectory_maker(player_position, player_angle)

class Player:
    def __init__(self, hitbox: Hitbox, starting_health, position, theta, stab_attack, blast_attack, sweep_attack):
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

        # shield variables
        self.shield_time_left = 0

        # attack variables
        self.attack: Attack = None
        self.attack_map = {
            AttackType.STAB_ATTACK: stab_attack,
            AttackType.BLAST_ATTACK: blast_attack,
            AttackType.SWEEP_ATTACK: sweep_attack
        }
        assert len(self.attack_map.keys()) == len(AttackType)

    # type hint is a string literal because this is a forward reference within the same class
    def connect_opponent(self, opponent: 'Player'):
        self.opponent = opponent

    # n.b. the order of logic in one_step is really important: be very careful when making changes
    def one_step(self, action: np.ndarray) -> LEGAL_ACTION | ILLEGAL_ACTION:
        # process movement
        dx, dy, sin, cos = action[-NUM_MOVEMENT_VALS:]
        self.position += np.array([dx, dy])
        self.theta += np.arctan2(sin, cos)
        self.theta = self.theta % (2 * np.pi)

        # process action
        action_type = np.argmax(action[:NUM_ACTIONS])
        is_legal_action = True
        just_activated_shield = False
        # must be in normal mode to start shielding or attacking
        if self.mode == Mode.NORMAL_MODE:
            # logic for attempting to start an attack
            if action_type < NUM_ATTACKS:
                attack_type = AttackType(action_type)
                if attack_type == AttackType.BLAST_ATTACK and self.blast_cooldown_timer > 0:
                    is_legal_action = False
                else:
                    self.mode = Mode.ATTACK_MODE
                    self.attack: Attack = self.attack_map[attack_type]
                    self.attack.restart(self.position, self.theta)
            # logic for attempting to shield
            elif action_type == SHIELD_INDEX:
                if self.shield_cooldown_timer > 0:
                    is_legal_action = False
                else:
                    self.mode == Mode.SHIELD_MODE
                    self.shield_time_left = SHIELD_DURATION
                    just_activated_shield = True
        else:
            # tried doing something while not in normal mode
            if action_type != DO_NOTHING_INDEX:
                is_legal_action = False

        # update timers
        if self.shield_cooldown_timer != 0:
            self.shield_cooldown_timer -= 1
        if self.blast_cooldown_timer != 0:
            self.blast_cooldown_timer -= 1

        # progress attack and calculate damage
        if self.mode == Mode.ATTACK_MODE:
            attack_status = self.attack.one_step(self.position, self.theta)
            self.opponent.calculate_damage(self.attack.position, self.attack.damage)

            # end attack and return to normal mode if done
            if attack_status == DONE:
                # set cooldown timer if the completed attack was a blast attack
                if self.attack.attack_type == AttackType.BLAST_ATTACK:
                    self.blast_cooldown_timer = BLAST_COOLDOWN_TIME
                self.mode = Mode.NORMAL_MODE
                self.attack = None

        # progress shield (skip on first timestep)
        if self.mode == Mode.SHIELD_MODE and not just_activated_shield:
            self.shield_time_left -= 1
            # turn shield off and return to normal mode
            if self.shield_time_left == 0:
                self.shield_cooldown_timer = SHIELD_COOLDOWN_TIME
                self.mode == Mode.NORMAL_MODE

        return LEGAL_ACTION if is_legal_action else ILLEGAL_ACTION

    # calculate damage inflicted to this player and update health
    def calculate_damage(self, weapon_x: np.ndarray, damage):
        # no damage if you have your shield up
        if self.mode == Mode.SHIELD_MODE:
            return
        # inflict damage for every point in the weapon that is in contact with the player
        A, b = self.compute_hitbox_constraint()
        is_point_in_contact = np.all(A @ weapon_x.T <= b.reshape(-1, 1), axis = 0)
        self.health -= damage * np.sum(is_point_in_contact)

    # compute the hitbox of the player by generating matrix A and vector b 
    # such that a point x is in the hitbox iff Ax <= b
    def compute_hitbox_constraint(self) -> Tuple[np.ndarray, np.ndarray]:
        m_vert = np.tan(self.theta + np.pi / 2) # slope of original left and right (i.e. vertical) sides
        m_hoz = np.tan(self.theta) # slope of original top and bottom (i.e. horizontal) sides
        # A encodes either -mx + y \leq b (under a line) or mx - y \leq -b (above a line)
        A = np.column_stack((np.array([-m_vert, m_vert, -m_hoz, m_hoz]), np.array([1, -1, 1, -1])))

        width, height = self.hitbox
        # left, right, top, and bottom are the halfway points on the respective sides
        # we'll use them to compute the y-intercept for the lines that go through each of these sides
        half_width_vector = np.array([np.cos(self.theta), np.sin(self.theta)]) * width / 2
        left = self.position - half_width_vector
        right = self.position + half_width_vector
        half_height_vector = np.array([np.cos(self.theta + np.pi / 2), np.sin(self.theta + np.pi / 2)]) * height / 2
        top = self.position + half_height_vector
        bottom = self.position - half_height_vector

        # the left or right side could be on top depending on the angle of rotation; since the slopes are the same,
        # we simply compute the y-intercepts and say that we want to be below the line with the higher intercept and
        # above the line with the lower intercept; same deal with top and bottom points
        left_right_b = np.sort(np.array([-m_vert, 1]) @ np.column_stack([left, right]))[::-1]
        top_bottom_b = np.sort(np.array([-m_hoz, 1]) @ np.column_stack([top, bottom]))[::-1]
        b = np.concatenate((left_right_b, top_bottom_b)) * np.array([1, -1, 1, -1])

        assert A.shape == (4, 2) and b.shape == (4,)
        return A, b

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

    def health_diff(self):
        return self.hero.health - self.adversary.health

    def tick(self):
        self.time_left -= 1
        # check for time running out
        if self.time_left == 0:
            return GameState(game_over = True, won = (self.health_diff() >= 0), time_left = self.time_left)
        # check for won
        if self.adversary.health <= 0:
            return GameState(game_over = True, won = True, time_left = self.time_left)
        # check for loss
        if self.hero.health <= 0:
            return GameState(game_over = True, won = False, time_left = self.time_left)
        return GameState(game_over = False, won = True, time_left = self.time_left)

    def pack_game_state(self, for_adversary: bool):
        a = self.hero.pack_player_state()
        b = self.adversary.pack_player_state()
        return torch.cat((b, a)) if for_adversary else torch.cat((a, b))
