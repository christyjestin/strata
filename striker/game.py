import numpy as np
from constants import *

def random_bool():
    return np.random.rand() < 0.5

# helper function to rotate a point by th radians around z axis
def rotate_around_z(th):
    return np.array([[np.cos(th), np.sin(th), 0], 
                     [-np.sin(th), np.cos(th), 0], 
                     [0, 0, 1]])

# helper function to check that two distances are close
def isclose(d1, d2):
    return np.abs(d1 - d2) < EPSILON

class Striker:
    # height in inches, weight in pounds, speed in inches per timestep, power, and chin in abstract units
    def __init__(self, corner, dominant_hand, height = 70, weight = 180, speed = 4, power = 10, chin = 10):
        assert corner in [RED_CORNER, BLUE_CORNER]
        assert dominant_hand in [RIGHTY, LEFTY]
        self.corner = corner
        self.dominant_hand = dominant_hand
        self.stance = ORTHODOX if (dominant_hand == RIGHTY) else SOUTHPAW
        self.height = height
        # damage scales down with weight and chin
        self.weight = weight
        self.chin = chin
        self.speed = speed
        # damage scales up with power
        self.power = power
        self.health = 100

        # physical measurements
        self.upper_arm_length = height * UPPER_ARM_RATIO
        self.forearm_length = height * FOREARM_RATIO
        self.thigh_length = height * THIGH_RATIO
        self.shin_length = height * SHIN_RATIO
        self.head_length = height * HEAD_LENGTH_RATIO
        self.torso_length = height * TORSO_LENGTH_RATIO
        self.head_width = height * HEAD_WIDTH_RATIO
        self.torso_width = height * TORSO_WIDTH_RATIO
        self.head_thickness = height * HEAD_THICKNESS_RATIO
        self.torso_thickness = height * TORSO_THICKNESS_RATIO
        self.knee_down = self.height * KNEE_DOWN_RATIO
        self.knee_forward = self.height * KNEE_FORWARD_RATIO
        self.elbow_flare = self.height * ELBOW_FLARE_RATIO
        self.elbow_down = self.height * ELBOW_DOWN_RATIO
        self.elbow_forward = self.height * ELBOW_FORWARD_RATIO
        self.torso_box = UNIT_CUBE * np.array([[self.torso_width, self.torso_thickness, self.torso_length]])
        self.head_box = UNIT_CUBE * np.array([[self.head_width, self.head_thickness, self.head_length]])

        # moves occur over multiple timesteps so for example when you punch, you add future
        # hand and elbow positions to the trajectory dict, and in future timesteps, you simply
        # execute the positions
        self.trajectories = dict()

        # stance init
        y = (-1 if (self.corner == BLUE_CORNER) else 1) * Y_START_OFFSET_IN_INCHES
        self.torso_position = np.array([[0, y, Z_START_HEIGHT_MULTIPLIER * height]])
        # start at 0 or 180 depending on corner and then add or subtract 30 depending on stance
        base_angle = np.pi * int(self.corner == RED_CORNER)
        self.torso_angle = base_angle + (-1 if (self.stance == ORTHODOX) else 1) * STANCE_ANGLE

        # init arms and legs
        self.reset_legs()
        self.reset_arms()

    # expose opponent's Striker object; necessary for aiming + damage
    def connect_opponent(self, striker):
        self.opponent = striker

    # rotate function for left knee or foot
    def left_leg_rotate(self, arr):
        return arr @ rotate_around_z(self.frontfoot_angle if self.stance == ORTHODOX else self.backfoot_angle)

    # rotate function for right knee or foot
    def right_leg_rotate(self, arr):
        return arr @ rotate_around_z(self.backfoot_angle if self.stance == ORTHODOX else self.frontfoot_angle)

    # rotate function for both arms
    def arm_rotate(self, arr):
        return arr @ rotate_around_z(self.arm_angle)

    # rotate function for torso and head
    def center_rotate(self, arr):
        return arr @ rotate_around_z(self.torso_angle)

    # knee goes down and out, foot is straight under knee
    def reset_legs(self):
        knee = np.array([[self.knee_forward, 0, -self.knee_down]])
        foot = np.array([[self.knee_forward, 0, -(self.knee_down + self.shin_length)]])
        self.left_knee = self.left_hip + self.left_leg_rotate(knee)
        self.left_foot = self.left_hip + self.left_leg_rotate(foot)
        self.right_knee = self.right_hip + self.right_leg_rotate(knee)
        self.right_foot = self.right_hip + self.right_leg_rotate(foot)

    # elbows go out, away, and down, hands go out but stay close to body and about level with head
    # hand position looks slightly awkward because the arms point forward instead of being perpendicular
    # to stance
    def reset_arms(self):
        right_elbow = np.array([[self.elbow_forward, -self.elbow_flare, -self.elbow_down]])
        # left elbow flares left while right elbow flares right
        left_elbow = np.array([[1, -1, 1]]) * right_elbow
        hand_up = np.sqrt(self.forearm_length ** 2 - self.elbow_flare ** 2)
        hand = np.array([[self.elbow_forward, 0, hand_up - self.elbow_down]])

        self.left_elbow = self.left_shoulder + self.arm_rotate(left_elbow)
        self.left_hand = self.left_shoulder + self.arm_rotate(hand)
        self.right_elbow = self.right_shoulder + self.arm_rotate(right_elbow)
        self.right_hand = self.right_shoulder + self.arm_rotate(hand)

    @property # torso box
    def torso_vertices(self):
        return self.torso_position + self.center_rotate(self.torso_box)

    @property # head box
    def head_vertices(self):
        return self.head_position + self.center_rotate(self.head_box)

    @property # straight up from torso center
    def head_position(self):
        return self.torso_position + np.array([[0, 0, (self.torso_length + self.head_length) / 2]])

    @property # up and right from torso center
    def right_shoulder(self):
        return self.torso_position + 0.5 * self.center_rotate(np.array([[self.torso_width, 0, self.torso_length]]))

    @property # up and left from torso center
    def left_shoulder(self):
        return self.torso_position + 0.5 * self.center_rotate(np.array([[-self.torso_width, 0, self.torso_length]]))

    @property # down and right from torso center
    def right_hip(self):
        return self.torso_position + 0.5 * self.center_rotate(np.array([[self.torso_width, 0, -self.torso_length]]))

    @property # down and left from torso center
    def left_hip(self):
        return self.torso_position + 0.5 * self.center_rotate(np.array([[-self.torso_width, 0, -self.torso_length]]))

    @property # backfoot points sideways
    def backfoot_angle(self):
        return self.torso_angle + (STANCE_ANGLE if self.stance == ORTHODOX else (np.pi - STANCE_ANGLE))

    @property # frontfoot points forward
    def frontfoot_angle(self):
        return self.torso_angle + NINETY_DEGREES + (STANCE_ANGLE if self.stance == ORTHODOX else -STANCE_ANGLE)

    @property # both arms point forward
    def arm_angle(self):
        return self.frontfoot_angle

    # either jab or cross (a straight punch)
    def punch(self, hand, target):
        assert target in [HEAD, BODY]
        assert hand in [LEFT_HAND, RIGHT_HAND]
        xf = self.opponent.head_position if target == HEAD else self.opponent.torso_position
        x0 = self.right_hand if hand == RIGHT_HAND else self.left_hand
        d = np.linalg.norm(xf - x0)
        n_steps = int(np.ceil(d / self.speed))
        path = [x0 + (xf - x0) * (i / n_steps) for i in range(1, n_steps)]
        # go forward, hit target, go backward, return to starting position
        self.trajectories[hand] = [*path, xf, *path[::-1], x0]

    # ensure that relationship between joints is still close to limb length
    def check_limbs(self):
        assert isclose(np.linalg.norm(self.left_hip - self.left_knee), self.thigh_length)
        assert isclose(np.linalg.norm(self.right_hip - self.right_knee), self.thigh_length)
        assert isclose(np.linalg.norm(self.left_knee - self.left_foot), self.shin_length)
        assert isclose(np.linalg.norm(self.right_knee - self.right_foot), self.shin_length)
        assert isclose(np.linalg.norm(self.left_shoulder - self.left_elbow), self.upper_arm_length)
        assert isclose(np.linalg.norm(self.right_shoulder - self.right_elbow), self.upper_arm_length)
        assert isclose(np.linalg.norm(self.left_elbow - self.left_hand), self.forearm_length)
        assert isclose(np.linalg.norm(self.right_elbow - self.right_hand), self.forearm_length)

    def in_range(self, hand, target):
        assert target in [HEAD, BODY]
        assert hand in [LEFT_HAND, RIGHT_HAND]
        xf = self.opponent.head_position if target == HEAD else self.opponent.torso_position
        socket = self.right_shoulder if hand == RIGHT_HAND else self.left_shoulder
        d = np.linalg.norm(xf - socket)
        return d < self.upper_arm_length + self.forearm_length

    # execute moves you've committed to in the future
    def handle_trajectories(self):
        for key in self.trajectories.keys():
            if key not in SETTABLE_BODY_PARTS:
                raise KeyError(key)
            self.__dict__[key] = self.trajectories[key].pop(0)
        # remove completed trajectory
        self.trajectories = {k: v for k, v in self.trajectories.items() if v}

    def step_forward(self):
        self.torso_position[0][1] += 1 if self.corner == BLUE_CORNER else -1
        self.reset_legs()
        self.reset_arms()

    def one_step(self):
        if self.trajectories:
            self.handle_trajectories()
        else:
            hand = LEFT_HAND if self.stance == ORTHODOX else RIGHT_HAND
            target = HEAD
            if self.in_range(hand, target):
                self.punch(hand, target)
            else:
                self.step_forward()
        # self.check_limbs()

class Fight:
    def __init__(self, round_length, red_dominant_hand, blue_dominant_hand):
        self.a = Striker(RED_CORNER, red_dominant_hand)
        self.b = Striker(BLUE_CORNER, blue_dominant_hand)
        self.a.connect_opponent(self.b)
        self.b.connect_opponent(self.a)
        self.time = round_length

    def one_step(self):
        self.a.one_step()
        self.b.one_step()