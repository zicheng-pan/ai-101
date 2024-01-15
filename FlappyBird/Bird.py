# 绘制小鸟类
import pygame


class Bird:
    # 构造函数
    def __init__(self, pad, bird_x, bird_y, bird_radius, bird_movement=0, color=(0, 0, 255)):
        self.pad = pad
        self.bird_x = bird_x
        self.bird_y = bird_y  # screen_height // 2
        self.bird_radius = bird_radius
        self.color = color  # BLUE
        self.bird_movement = bird_movement  # 初始化小鸟降落的速度,初始速度是0

    def draw_self(self):
        pygame.draw.circle(self.pad.screen, self.color, (self.bird_x, self.bird_y), self.bird_radius)

    def jump(self):
        self.bird_movement = -10
        self.bird_y += self.bird_movement

    def free_fall(self, gravity):
        # 这里设置一个向上的惯性
        self.bird_movement += gravity
        self.bird_y += self.bird_movement
