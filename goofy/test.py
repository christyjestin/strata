import pygame
import sys
import math
from pygame.locals import *

# Initialize pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))

# Create a square surface
square_size = 100
square_surface = pygame.Surface((square_size, square_size))
square_surface.fill((255, 0, 0))
square_surface.set_colorkey((0, 0, 0))

# Define a rotation angle
rotation_angle = 0

# Set up the game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # Get pressed keys
    keys = pygame.key.get_pressed()

    # Rotate the square using left and right arrow keys
    if keys[K_LEFT]:
        rotation_angle -= 1
    if keys[K_RIGHT]:
        rotation_angle += 1

    # Rotate the square surface
    diagonal = math.sqrt(square_size ** 2 + square_size ** 2)
    rotated_square = pygame.Surface((diagonal, diagonal), pygame.SRCALPHA)
    rotated_square.fill((0, 0, 0, 0))

    rotated_surface = pygame.transform.rotate(square_surface, rotation_angle)

    # Calculate the position to draw the rotated square
    pos_x = (rotated_square.get_width() - rotated_surface.get_width()) // 2
    pos_y = (rotated_square.get_height() - rotated_surface.get_height()) // 2

    rotated_square.blit(rotated_surface, (pos_x, pos_y))

    # Calculate the position to draw the rotated square on the screen
    screen_pos_x = 200
    screen_pos_y = 200

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the rotated surface on the screen
    screen.blit(rotated_square, (screen_pos_x, screen_pos_y))

    # Update the display
    pygame.display.update()
    pygame.time.delay(10)