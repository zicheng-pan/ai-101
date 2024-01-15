# 游戏面板
import pygame

from FlappyBird.Bird import Bird


class GamePad:
    # 这个会随着时间调整而主键变快
    scroll_speed = -3
    pipe_list = []

    # Colors
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)

    def __init__(self, screen_width, screen_height):
        # Initialize Pygame
        pygame.init()
        # Game variables
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Set up the screen
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.game_active = True
        pygame.display.set_caption("Flappy Bird")
        self.bird = Bird(0, self.screen_height // 2, 20, 10)
        # Clock to control the frame rate
        self.clock = pygame.time.Clock()
        # 重力
        self.gravity = 0.5

    def play(self):
        # Game loop
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game_active:
                        self.bird.jump()

            self.screen.fill((0, 0, 0))  # Clear screen
            if self.game_active:
                # Bird movement
                self.bird.free_fall(self.gravity)
                self.bird.draw_self(self)
                #
                # # Move and draw pipes
                # pipe_list = move_pipes(pipe_list)
                #
                # draw_pipes(pipe_list)
                #
                # # Check for collisions
                # game_active = check_collision(pipe_list)

            # Update the display
            pygame.display.update()

            # Frame rate
            self.clock.tick(60)  # 60 frames per second
