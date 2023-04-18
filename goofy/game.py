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
    

    # check for overlap in damage boxes between hero and adversary, change their health accordingly
    def calculate_damage()



class Hero:

    mode = Mode.NORMAL
   
    def __init__(self, hitbox_width, hitbox_height, movement_speed, rotation_speed, max_health, position):
        self.hitbox_width = hitbox_width
        self.hitbox_height = hitbox_height
        self.max_health = max_health
        self.health = max_health
        self.position = position
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed

        self.Adversary = None

        # Damage box is, relative to player, where the player is doing damage and magnitude of damage
        self.damagebox = None



    def one_step()
        

    # Changes Damage box to the corresponding attack
    def select_attack()




class Adversary:
    def __init__(self, hitbox_width, hitbox_height, movement_speed, rotation_speed, max_health, position, attackloop):
        self.hitbox_width = hitbox_width
        self.hitbox_height = hitbox_height
        self.max_health = max_health
        self.health = max_health
        self.position = position
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed

        self.loop = attackloop
        self.Hero = None

        # Damage box is, relative to player, where the player is doing damage and magnitude of damage
        self.damagebox = None


    # Changes Damage box to the corresponding attack. 
    def select_attack()

    def one_step()
        # Chase Player until at a certain range

        # Use attack loop unless next move is ranged, then do ranged instead of chasing



        # If loop requires sequence, perform sequence 


