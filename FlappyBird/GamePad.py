# 游戏面板
import pygame

from FlappyBird.Bird import Bird
from FlappyBird.PipePair import PipePair


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
        # Clock to control the frame rate
        self.clock = pygame.time.Clock()
        # 重力
        self.gravity = 0.5
        pygame.display.set_caption("Flappy Bird")

        self.game_active = True
        self.bird = Bird(self, 24, self.screen_height // 2, 20, 10)
        self.pipe_pair = PipePair(self, GamePad.WHITE)
        self.reward = 0.1
        self.pass_pipe_count = 0

    def check_collision(self):

        if not (0 + self.bird.bird_radius <= self.bird.bird_y <= self.screen_height - self.bird.bird_radius):
            return True

        if self.pipe_pair.get_pipes_position_x() - self.bird.bird_radius <= self.bird.bird_x <= self.pipe_pair.get_pipes_position_x() + PipePair.pipe_width + self.bird.bird_radius:
            if self.pipe_pair.get_pipes_position_y()[0] <= self.bird.bird_y <= self.pipe_pair.get_pipes_position_y()[1]:
                return False
            else:
                return True
        return False

    def show_game_over_screen(self):
        # 显示分数和重启提示
        font = pygame.font.SysFont(None, 36)
        text = font.render("Game Over! Press Enter to Restart", True, self.WHITE)
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text, text_rect)

        text = font.render("get scores: {}".format(self.pipe_pair.get_pipes_count()), True, self.WHITE)
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 72))
        self.screen.blit(text, text_rect)

    def reset_game(self):
        self.game_active = True
        self.bird = Bird(self, 24, self.screen_height // 2, 20, 10)
        self.pipe_pair = PipePair(self, GamePad.WHITE)
        self.reward = 0.1
        self.pass_pipe_count = 0
        return self.get_state()

    # 解释下强化学习的各种名词
    def get_state(self):
        # [      bird_y,         bird_velocity,          pipe_distance,
        # top_pipe_bottom_y, bottom_pipe_top_y]

        return [self.bird.bird_y, self.bird.bird_movement, self.pipe_pair.get_pipes_position_x(),
                self.pipe_pair.get_pipes_position_y()[0], self.pipe_pair.get_pipes_position_y()[1]]

    def add_reward(self, reward=1):
        self.reward += reward

    def sub_reward(self):
        self.reward -= 1000

    def get_reward(self):
        return self.reward

    def step(self):
        # Bird movement
        self.bird.free_fall(self.gravity)
        self.pipe_pair.move_pipes()
        # 碰撞检测
        is_collision = self.check_collision()
        if is_collision:
            self.sub_reward()
        elif self.pass_pipe_count < self.pipe_pair.get_pipes_count():
            self.add_reward(self.pipe_pair.get_pipes_count() - self.pass_pipe_count)
            self.pass_pipe_count = self.pipe_pair.get_pipes_count()

        return self.get_state(), self.get_reward(), is_collision

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
                    if event.key == pygame.K_RETURN and not self.game_active:
                        self.reset_game()

            self.screen.fill((0, 0, 0))  # Clear screen
            if self.game_active:
                # Bird movement
                new_state, reward, done = self.step()
                # print(new_state, reward, done)
                self.pipe_pair.draw_self()
                self.bird.draw_self()

                # 碰撞检测
                if done:
                    self.game_active = False

            else:
                self.show_game_over_screen()

            # Update the display
            pygame.display.flip()

            # Frame rate
            self.clock.tick(60)  # 60 frames per second



