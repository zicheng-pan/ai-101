# 强化学习需要一个套模拟，评估的机制
import threading

from FlappyBird.GamePad import GamePad


class FlappyBirdEnv():

    def __init__(self):
        # 定义动作空间
        self.action_space = [0, 1]  # 两个动作：0:跳跃或1:不跳跃

        # Agent可以观察到的空间是R
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_states,), dtype=np.float32)  # 状态空间
        self.pad = GamePad(400, 600)

    def step(self, action):
        # 应用动作并计算新的状态
        # 这里需要集成游戏逻辑，比如小鸟的移动、管道的移动等
        # 返回新的状态、奖励、游戏是否结束的标志以及额外信息
        # new_state = [bird_y, bird_velocity, pipe_distance, top_pipe_bottom_y, bottom_pipe_top_y]
        # return new_state, reward, done
        return self.pad.step(action)

    def reset(self):
        # 重置环境到初始状态
        # 初始化小鸟位置、管道位置等
        return self.pad.reset_game()

    def render(self):
        # # 可视化环境（如果需要的话）
        # self.pad.screen.fill((0, 0, 0))  # Clear screen
        # self.pad.pipe_pair.draw_self()
        # self.pad.bird.draw_self()
        # if self.pad.game_active:
        #     self.pad.show_game_over_screen()
        self.pad.play(True)

    def close(self):
        # 清理资源（如果有的话）
        pass
