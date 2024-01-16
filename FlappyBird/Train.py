import random

import numpy as np

# 学习率
from FlappyBird.FlappyBirdEnv import FlappyBirdEnv

alpha = 0.9
# 折扣因子
gamma = 0.9

# 鸟高度*鸟速度*管道距离*管道高度（3种）
total_count = 300 * 500 * 10 * 3
Q_table = np.zeros((total_count, 5))

# 训练次数
episode = 1000

# 声明Env
env = FlappyBirdEnv()

# https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.337.search-card.all.click&vd_source=f0d4ad56c3e3d129d244615840c601ee
def policy(state):
    eps = 0.05
    if random.random() < eps:
        return env.action_space.sample()
    else:
        # 获取可以执行的action
        array1 = np.where(info["action_mask"] == 1)[0]
        array2 = np.array(Q_table[state][array1], dtype=int)
        # 获取对应位置的Q_table中的最大值的索引
        array3 = np.where(array2 == np.max(array2))[0]
        # 如果多个同等价值的action，那么随机筛选一个动作
        return array1[np.random.choice(array3)]
