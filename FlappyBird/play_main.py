# 这里找到使用requirements的方式来找到管理
import pygame
import random




# 绘制管道 pipes是list类型
def draw_pipes(pipes):
    for pipe_pair in pipes:
        pygame.draw.rect(screen, WHITE, (pipe_pair[0][0], pipe_pair[0][1], pipe_width, screen_height))
        pygame.draw.rect(screen, WHITE, (pipe_pair[1][0], pipe_pair[1][1], pipe_width, screen_height))


# 创建管道位置
def create_pipe():
    random_pipe_pos = random.choice(pipe_heights)
    bottom_pipe = [screen_width, random_pipe_pos]  # Using list instead of tuple
    top_pipe = [screen_width, random_pipe_pos - pipe_gap - screen_height]  # Using list
    return [bottom_pipe, top_pipe]  # Returning a list of lists


def move_pipes(pipes):
    for pipe_pair in pipes:
        pipe_pair[0][0] += scroll_speed  # Update x-position of bottom pipe
        pipe_pair[1][0] += scroll_speed  # Update x-position of top pipe
        if pipe_pair[0][0] < 0:
            pipes.pop()
            pipes.append(create_pipe())
    return pipes


def check_collision(pipes):
    for pipe_pair in pipes:
        if bird_x + bird_radius > pipe_pair[0][0] and bird_x - bird_radius < pipe_pair[0][0] + pipe_width:
            if bird_y + bird_radius > pipe_pair[0][1] or bird_y - bird_radius < pipe_pair[1][1] + screen_height:
                return False
    return True


# Create initial pipes
for _ in range(1):
    pipe_list.append(create_pipe())

# Game loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird_movement = -10

    # Game logic
    screen.fill((0, 0, 0))  # Clear screen
    if game_active:
        # Bird movement
        bird_movement += gravity
        bird_y += bird_movement
        draw_bird(bird_x, bird_y)

        # Move and draw pipes
        pipe_list = move_pipes(pipe_list)

        draw_pipes(pipe_list)

        # Check for collisions
        game_active = check_collision(pipe_list)

    # Update the display
    pygame.display.update()

    # Frame rate
    clock.tick(60)  # 60 frames per second