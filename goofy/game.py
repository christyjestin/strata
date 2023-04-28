from collections import namedtuple
from enum import Enum
import numpy as np

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
    #STUNNED = 4

class HeroAttacks(Enum):
    RANGESTAB = 1
    SHORTSWEEP = 2

class AdversaryAttacks(Enum):
    RANGESTAB = 1
    SHORTSWEEP = 2


# note trajectories may need to be shifted!
RANGESTAB_TRAJECTORY_HERO = np.array([[0,0], [0, 1], [0, 2], [0, 3], [0, 4]])

SHORTSWEEP_TRAJECTORY_HERO = np.array([[0,0], [0, 1], [0, 2], [0, 3], [0, 4]])  

RANGESTAB_BOX_HERO = {'width': 5, 'height': 5}
SHORTSWEEP_BOX_HERO = {'width': 5, 'height': 5}
                    
class Attack:

    def __init__(self, damagebox, damage, length, time, trajectory):
        self.damagebox_width = damagebox['width']
        self.damagebox_height = damagebox['height']
        self.damage = damage
        self.length = length
        self.time = 0
        self.trajectory = trajectory

        # Ensure this is doing the right thing
        self.position = trajectory[0]

    def one_step(self):

        # Attack ends
        if self.time == self.length:
            return True

        # Else Attack follows trajectory
        self.time+= 1
        self.position = self.trajectory[self.time]
        return False


class Game:
    arena_length = 1920
    arena_width = 960

    # im just gonna make comments for features to add

    def __init__(self, hero, adversary, round_length):
        self.hero = hero
        self.adversary = adversary
        self.round_length = round_length
        self.time_left = round_length

        # Gamestate vector is length of round, initially empty, total length = round_length (s) *TIMESTEPS_PER_SEC
        self.game_state_list = []


    def one_step(self):
        # maybe just let them have modes like shield raised and trust each agent to correctly do damage on the other     like a state?
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
    

    # check for overlap in damage boxes between hero and adversary, change their health accordingly
    # Currently ignoring defending
    # TAKE THE CONVENTION THAT  POSITION IS TOP LEFT OF HIT BOX!!!
    def calculate_damage(self):
        
        # damage_box_hero = self.hero.damagaebox
        # damage_box_adversary = self.adversary.damagebox

        # hero_pos = self.hero.position
        # adversary_pos = self.adversary.position

        # #NOTE: need to rotate damageboxes along with players/adversaries as they are in their frames

        # # # Hero takes damage
        # # if ():
        # #     self.hero.health -= damage_box_adversary['damage']


        # # Adversary takes damage
        # if ():
        #     self.adversary.health -= damage_box_hero['damage']
        pass



# ALL HITBOXES ARE ADDED TO PLAYER POSITION! so if players position 
# is [0,0] hitbox extends from [0,0] to [width, height]

class Player:

    mode = Mode.NORMAL
   
    def __init__(self, hitbox, max_movement_speed, max_rotation_speed, max_health, position, theta, arena_width, arena_length):
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

        # Damage box is, relative to player, where the player is doing damage and magnitude of damage
        self.Attack = None



    def one_step(self):

        # Player chooses movement and attack/defensive action
        self.position = self.position + 10*(np.random.rand(2) - 0.5)
        self.theta += 1

        if self.theta > 360:
            self.theta -= 360

        if self.theta < -360:
            self.theta += 360

        if self.position[0] < 0:
            self.position[0] = 0

        if self.position[1] < 0:
            self.position[1] = 0

        if self.position[0] > self.arena_width:
            self.position[0] = self.arena_width

        if self.position[1] > self.arena_length:
            self.position[1] = self.arena_width

        # # Step attack
        # if self.mode == Mode.NORMAL:




        

    # # Changes Damage box to the corresponding attack
    # def select_attack(self):
            



    # def adversary_one_step(self):
    #     pass




hero = Player(RANGESTAB_BOX_HERO, 1, 1, 10, np.array([50,50]), 0, 1920, 960)

adversary = Player(RANGESTAB_BOX_HERO, 1, 1, 10, np.array([50,50]), 0, 1920, 960)

game = Game(hero, adversary, 10)




game.one_step()

