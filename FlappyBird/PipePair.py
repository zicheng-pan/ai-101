import random

import pygame


class PipePair:
    pipe_heights = [400, 500, 300]
    pipe_width = 70
    pipe_gap = 200

    def __init__(self, pad, color):
        self.pad = pad
        self.color = color
        self.pipe_pair = self.create_pipe()
        self.count = 0

    def draw_self(self):
        # TODO 记录下方法的作用，先展示下pygame的api
        # Rect：这是一个 Pygame Rect 对象或者是一个包含四个元素的元组 (left, top, width, height)
        pygame.draw.rect(self.pad.screen, self.color,
                         (self.pipe_pair[0][0], self.pipe_pair[0][1], PipePair.pipe_width, self.pad.screen_height))
        pygame.draw.rect(self.pad.screen, self.color,
                         (self.pipe_pair[1][0], self.pipe_pair[1][1], PipePair.pipe_width, self.pad.screen_height))

    # 创建管道位置
    def create_pipe(self):
        # 这里choice随机选择一个下管道所在的位置，然后计算通过gap和屏幕长度计算上管道所在的绘画起始位置
        random_pipe_pos = random.choice(PipePair.pipe_heights)
        bottom_pipe = [self.pad.screen_width, random_pipe_pos]
        top_pipe = [self.pad.screen_width, random_pipe_pos - self.pad.screen_height - PipePair.pipe_gap]
        return [top_pipe, bottom_pipe]

    def move_pipes(self):
        self.pipe_pair[0][0] += self.pad.scroll_speed
        self.pipe_pair[1][0] += self.pad.scroll_speed
        if self.pipe_pair[0][0] < 0:
            self.pipe_pair = self.create_pipe()
            self.count += 1

    def get_pipes_position_x(self):
        return self.pipe_pair[0][0]

    def get_pipes_position_y(self):
        return [self.pipe_pair[0][1] + self.pad.screen_height, self.pipe_pair[1][1]]

    def get_pipes_count(self):
        return self.count
