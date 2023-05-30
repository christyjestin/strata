import numpy as np

# body proportions (based on a photo of Khabib Nurmagomedov)
UPPER_ARM_RATIO = 0.15
FOREARM_RATIO = 0.21
THIGH_RATIO = 0.255
SHIN_RATIO = 0.255
HEAD_LENGTH_RATIO = 0.16
TORSO_LENGTH_RATIO = 0.33
HEAD_WIDTH_RATIO = 0.1
TORSO_WIDTH_RATIO = 0.24
HEAD_THICKNESS_RATIO = 0.1
TORSO_THICKNESS_RATIO = 0.05

RING_RADIUS = 12 * 15 # 15 feet
RING_HEIGHT = 12 * 6 # 6 feet

RED_CORNER = 'red'
BLUE_CORNER = 'blue'

ORTHODOX = 'orthodox'
SOUTHPAW = 'southpaw'

RIGHTY =  'righty'
LEFTY = 'lefty'

HEAD = 'head'
BODY = 'body'

# 30 degree stance
STANCE_ANGLE = np.pi / 6

# starts this distance back from center of ring
Y_START_OFFSET_IN_INCHES = 32

# torso center starting height
Z_START_HEIGHT_MULTIPLIER = 0.66

# ratios for elbow and knee positioning
KNEE_DOWN_RATIO = Z_START_HEIGHT_MULTIPLIER - (TORSO_LENGTH_RATIO / 2) - SHIN_RATIO
KNEE_FORWARD_RATIO = np.sqrt(THIGH_RATIO ** 2 - KNEE_DOWN_RATIO ** 2)
ELBOW_FLARE_RATIO = 1 / 3 * UPPER_ARM_RATIO
ELBOW_DOWN_RATIO = 2 / 3 * UPPER_ARM_RATIO
ELBOW_FORWARD_RATIO = 2 / 3 * UPPER_ARM_RATIO

NINETY_DEGREES = np.pi / 2

# order of vertices matter for plotting
UNIT_CUBE = 0.5 * np.array([[1., 1., -1.], 
                            [-1., 1., -1.], 
                            [-1., -1., -1.], 
                            [1., -1., -1.], 
                            [1., 1., 1.], 
                            [-1., 1., 1.], 
                            [-1., -1., 1.], 
                            [1., -1., 1.]])

# upper bound on difference between the practical and true limb length
EPSILON = 1e-4

RIGHT_HAND = 'right_hand'
RIGHT_ELBOW = 'right_elbow'
RIGHT_FOOT = 'right_foot'
RIGHT_KNEE = 'right_knee'
LEFT_HAND = 'left_hand'
LEFT_ELBOW = 'left_elbow'
LEFT_FOOT = 'left_foot'
LEFT_KNEE = 'left_knee'

SETTABLE_BODY_PARTS = {
    RIGHT_HAND, 
    RIGHT_ELBOW, 
    RIGHT_FOOT, 
    RIGHT_KNEE, 
    LEFT_HAND, 
    LEFT_ELBOW, 
    LEFT_FOOT, 
    LEFT_KNEE, 
}