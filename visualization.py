import pygame
import math
from game import *


# initialize PyGame
pygame.init()

#Screen Dims
WIDTH = 1280
HEIGHT = 960

# Player dims
PLAYER_SIZE = 32

# Enemy dims
ADVERSARY_SIZE = 48

# Colors needed for visualization
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Build Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Strata Demo")

# player initial position
player_x = WIDTH // 2 - PLAYER_SIZE // 2
player_y = HEIGHT // 2 - PLAYER_SIZE // 2

# Adversary initial position
adversary_x = WIDTH // 3 - PLAYER_SIZE // 2
adversary_y = HEIGHT // 2 - PLAYER_SIZE // 2

# Character sprites
player_im = pygame.image.load("hero.png")
adversary_im = pygame.image.load("adversary.png")

# Prints the players to screen while preserving their sizing during rotation
def blitRotate(screen, image, pos, originPos, angle):

    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
    
    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    # rotate and blit the image
    screen.blit(rotated_image, rotated_image_rect)


# Prints all character's damage points to screen
def blit_damage_points(screen, points, color, point_radius = 2):
    for point in points:
        x, y = point
        pygame.draw.circle(screen, color, (x, y), point_radius)


font = pygame.font.Font(pygame.font.get_default_font(), 36)
w, h = player_im.get_size()

SPEED = 5
ROT_SPEED = 10

# Game Loop
running = True
while running:
    x_change = 0
    y_change = 0
    theta_dot = 0
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            y_change = -SPEED
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            y_change = SPEED
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            x_change = -SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            x_change = SPEED

        if keys[pygame.K_r]:
            theta_dot = ROT_SPEED
        if keys[pygame.K_q]:
            theta_dot = -ROT_SPEED

    
    game.one_step()
    game.hero.position = game.hero.position + np.array([x_change, y_change])
    game.hero.theta = game.hero.theta + theta_dot

    player_pos = list(game.hero.position)
    player_angle = game.hero.theta
    adversary_pos = list(game.adversary.position)
    adversary_angle = game.adversary.theta

    blitRotate(screen, player_im, player_pos, (w/2, h/2), player_angle)
    blitRotate(screen, adversary_im, adversary_pos, (w/2, h/2), adversary_angle)

    player_points = hero.damagepoints
    adversary_points = adversary.damagepoints

    blit_damage_points(screen, player_points, BLACK)
    blit_damage_points(screen, adversary_points, RED)

    goal = font.render('Autostab enabled, defeat adversary! (awsd to move, r,q to rotate)'.format(int(hero.health)), True, (0, 0, 0))
    screen.blit(goal, dest=(0,900))

    hero_health = font.render('Player Health: {}'.format(int(hero.health)), True, (0, 0, 0))
    screen.blit(hero_health, dest=(0,0))

    adversary_health = font.render('Adversary Health: {}'.format(int(adversary.health)), True, (0, 0, 0))
    screen.blit(adversary_health, dest=(800,0))

    pygame.display.update()
    pygame.time.delay(20)

# Quit the game
pygame.quit()




