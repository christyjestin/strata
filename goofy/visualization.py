import pygame
import math


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

player_body = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
player_body.fill(BLUE)

adversary_body = pygame.Surface((ADVERSARY_SIZE, ADVERSARY_SIZE))
adversary_body.fill(RED)


angle = 0


def blitplayer(body, bodysize, xpos, ypos, angle):


    # Rotate the square surface
    diagonal = math.sqrt(bodysize ** 2 + bodysize ** 2)
    rotated_square = pygame.Surface((diagonal, diagonal), pygame.SRCALPHA)
    rotated_square.fill((0, 0, 0, 0))

    rotated_surface = pygame.transform.rotate(body, angle)

    # Calculate the position to draw the rotated square
    pos_x = (rotated_square.get_width() - rotated_surface.get_width()) // 2
    pos_y = (rotated_square.get_height() - rotated_surface.get_height()) // 2

    rotated_square.blit(rotated_surface, (pos_x, pos_y))



    screen.blit(rotated_square, (xpos, ypos))










# Game Loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_y-=.1

    elif keys[pygame.K_DOWN]:
        player_y+=.1
    
    elif keys[pygame.K_RIGHT]:
        player_x+=.1
    
    elif keys[pygame.K_LEFT]:
        player_x-=.1




    screen.blit((player_body), (player_x, player_y))
    screen.blit(adversary_body, (adversary_x, adversary_y))
    
    # Keep the box inside the screen boundaries
    player_x = max(0, min(WIDTH - PLAYER_SIZE, player_x))
    player_y = max(0, min(HEIGHT - PLAYER_SIZE, player_y))


    pygame.display.update()

# Quit the game
pygame.quit()




