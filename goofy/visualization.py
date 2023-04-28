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
pygame.display.set_caption("Goofy Game")

# player initial position
player_x = WIDTH // 2 - PLAYER_SIZE // 2
player_y = HEIGHT // 2 - PLAYER_SIZE // 2

# Adversary initial position
adversary_x = WIDTH // 3 - PLAYER_SIZE // 2
adversary_y = HEIGHT // 2 - PLAYER_SIZE // 2

player_im = pygame.image.load("hero.png")

adversary_im = pygame.image.load("adversary.png")

def blitRotate(surf, image, pos, originPos, angle):

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
    surf.blit(rotated_image, rotated_image_rect)

w, h = player_im.get_size()


hero = Player(RANGESTAB_BOX_HERO, 1, 1, 10, np.array([WIDTH/2,HEIGHT/2]), 0, WIDTH, HEIGHT)

adversary = Player(RANGESTAB_BOX_HERO, 1, 1, 10, np.array([WIDTH/4,HEIGHT/2]), 0, WIDTH, HEIGHT)

game = Game(hero, adversary, 10)

game.one_step()





# Game Loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    

    game.one_step()

    player_pos = list(game.hero.position)
    player_angle = game.hero.theta

    adversary_pos = list(game.adversary.position)
    adversary_angle = game.adversary.theta

    blitRotate(screen, player_im, player_pos, (w/2, h/2), player_angle)

    blitRotate(screen, adversary_im, adversary_pos, (w/2, h/2), adversary_angle)
    
    # Keep the box inside the screen boundaries
    player_x = max(0, min(WIDTH - PLAYER_SIZE, player_x))
    player_y = max(0, min(HEIGHT - PLAYER_SIZE, player_y))


    pygame.display.update()

# Quit the game
pygame.quit()




