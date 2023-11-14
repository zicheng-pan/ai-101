import random
import time

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
import chex
from PIL import Image
from torch.nn.functional import relu
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm_notebook

# 声明Env
env = gym.make("LunarLander-v2", render_mode='rgb_array')

# 画图方法
frames = []


def draw():
    rgb = env.render()
    frames.append(rgb)
    plt.imshow(rgb)
    plt.show()
    time.sleep(1)


def rgb2y(array: np.ndarray) -> np.ndarray:
    """Converts RGB image array into grayscale."""
    chex.assert_rank(array, 3)
    output = np.tensordot(array, [0.299, 0.587, 1 - (0.299 + 0.587)], (-1, 0))
    return output.astype(np.uint8)


def resize(array, image_shape):
    image = Image.fromarray(array).resize(
        image_shape, Image.Resampling.BILINEAR
    )
    return np.array(image, dtype=np.uint8)


def getStateFromEnv(picarray, image_shape=(84, 84)):
    resized = resize(picarray, image_shape)
    turntogray = rgb2y(resized)
    return turntogray


class ExperienceBuffer:
    def __init__(self, size=0):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.next_actions = []
        self.size = 0

    def clear(self):
        self.__init__()

    def append(self, s, n_s, r, a, n_a):
        self.states.append(s)
        self.next_states.append(n_s)
        self.rewards.append(r)
        self.actions.append(a)
        self.next_actions.append(n_a)
        self.size += 1

    def batch(self, batch_size=128):
        # return 'bath_size' experiences
        indices = np.random.choice(self.size, size=batch_size, replace=True)

        return (
            np.array(self.states)[indices],
            np.array(self.next_states)[indices],
            np.array(self.rewards)[indices],
            np.array(self.actions)[indices],
            np.array(self.next_actions)[indices],
        )


class DQN(nn.Module):

    def __init__(self, action_size):
        self.action_size = action_size
        super().__init__()
        self.conv1 = nn.Sequential(  # 输入大小 (1, 84, 84)
            nn.Conv2d(
                in_channels=1,  # 灰度图   说明输入是几个channel的
                out_channels=16,  # 要得到几多少个特征图，也是卷积核的特殊
                kernel_size=5,  # 卷积核大小 越小越好 一般3*3 或者5*5
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为 (16, 84, 84)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (16, 42, 42)
            # pooling对原来的数据进行压缩，一般默认都是变成原来的一半1/2，正常的池化就是变成原来的一般，除非特殊的池化操作
        )
        # 这里设置第二个层
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 42, 42)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 42, 42)
            nn.ReLU(),  # relu层
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (32, 21, 21)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 1, 5, 1, 2),  # 输出 (batch_size, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出 (batch_size, 12, 12)
        )

        self.out = nn.Linear(1 * 10 * 10, action_size)  # 全连接层得到的结果 ， 4个动作

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view( -1)  # 这样就将特征展开，组成了【batch，特征数】
        output = self.out(x)  # 这样就可以计算全连接操作，一个特征图不能做全连接操作
        return output


def policy(model, state, _eval=False):
    eps = 0.1
    if not _eval and random.random() < eps:
        # exploration
        return random.randint(0, model.action_size - 1)
    else:
        # exploitation
        q_values = dqn_model(torch.tensor(state, dtype=torch.float).to(torch.device("cuda")).unsqueeze(0))
        action = torch.multinomial(F.softmax(q_values), num_samples=1)
        return int(action[0])


# 经验回放

experience_buffer = ExperienceBuffer()

eval_iter = 100
eval_num = 100

dqn_model = DQN(action_size=4)
target_model = DQN(action_size=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn_model.to(device)
target_model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
adam = Adam(lr=1e-3, params=dqn_model.parameters())  # 定义优化器，普通的随机梯度下降算法
loss_fn = nn.MSELoss()

num_epochs = 4


# collect
def collect():
    for e in tqdm_notebook(range(num_epochs)):
        # trajectory = []
        state, info = env.reset()
        state_grade_array = getStateFromEnv(env.render())
        # print(info["action_mask"])
        # a = env.action_space.sample(info["action_mask"])
        action = policy(dqn_model, state_grade_array)

        sum_reward = 0

        while True:
            state_next, reward, terminated, truncated, info_next = env.step(action)
            next_state_grade_array = getStateFromEnv(env.render())
            sum_reward += reward

            action_next = policy(dqn_model, next_state_grade_array)

            experience_buffer.append(
                state_grade_array, next_state_grade_array, reward, action, action_next
            )
            if terminated or truncated:
                break

            state_grade_array = next_state_grade_array
            state = state_next
            info = info_next
            action = action_next


collect()

## learning
losses = []
target_fix_period = 5
gamma = 0.99

epoch = 100


def train():
    for e in range(epoch):
        batch_size = 2
        for i in range(experience_buffer.size // batch_size):
            print(i, end=',')
            s, s_n, r, a, a_n = experience_buffer.batch(batch_size)

            target = torch.tensor(r, dtype=torch.float).to(torch.device("cuda")) + gamma * target_model(torch.tensor(s_n, dtype=torch.float).to(torch.device("cuda")).unsqueeze(0))[0][
                torch.arange(batch_size), a_n]
            y = dqn_model(s)[0][torch.arange(batch_size), a]

            loss = loss_fn(y, target)
            loss.backward()
            adam.zero_grad()
            adam.step()
            losses.append(loss.detach().numpy())

            if i % 10 == 0:
                print(f'i == {i}, loss = {loss} ')

            if i % target_fix_period == 0:
                ## copy the paramters from 'dqn_model' to 'target_model'
                target_model.load_state_dict(dqn_model.state_dict())


train()

env.reset()
env.action_space.sample()
draw()

resized = resize(frames[0], (84, 84))
turntogray = rgb2y(resized)
plt.imshow(turntogray)
plt.show()
print(turntogray.shape)
