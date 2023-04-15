import pygame


# initialize PyGame
pygame.init()

#Screen Dims
WIDTH = 640
HEIGHT = 480

# Player dims
PLAYER_SIZE = 32

# Colors needed for visualization
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Build Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Goofy Game")

# Box initial position
box_x = WIDTH // 2 - PLAYER_SIZE // 2
box_y = HEIGHT // 2 - PLAYER_SIZE // 2


# Game Loop
running = True
while running:
    screen.fill(WHITE)

    # Draw the box
    pygame.draw.rect(screen, BLUE, (box_x, box_y, PLAYER_SIZE, PLAYER_SIZE))

    # Update the display
    pygame.display.flip()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        box_y-=.1

    elif keys[pygame.K_DOWN]:
        box_y+=.1
    
    elif keys[pygame.K_RIGHT]:
        box_x+=.1
    
    elif keys[pygame.K_LEFT]:
        box_x-=.1
    

        # # Keyboard movement
        # if event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_UP:
        #         box_y -= 10
        #     elif event.key == pygame.K_DOWN:
        #         box_y += 10
        #     elif event.key == pygame.K_LEFT:
        #         box_x -= 10
        #     elif event.key == pygame.K_RIGHT:
        #         box_x += 10

    # Keep the box inside the screen boundaries
    box_x = max(0, min(WIDTH - PLAYER_SIZE, box_x))
    box_y = max(0, min(HEIGHT - PLAYER_SIZE, box_y))

# Quit the game
pygame.quit()




