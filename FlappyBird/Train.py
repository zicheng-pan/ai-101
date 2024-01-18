import random
import pickle

# 学习率
from FlappyBird.FlappyBirdEnv_local import FlappyBirdEnv

alpha = 0.9
# 折扣因子
gamma = 0.9

# 鸟高度*鸟速度*管道距离*管道高度（3种）
total_count = 300 * 500 * 10 * 3

# 从文件中加载序列化的对象
with open('version_1_Q_table.pkl', 'rb') as file:
    Q_table = pickle.load(file)

# 训练次数
# 大家也同我一起见证了大力出奇迹的效果
episode = 100000000

# 声明Env
env = FlappyBirdEnv()

#
# # https://www.bilibili.com/video/BV1sd4y167NS/?spm_id_from=333.337.search-card.all.click&vd_source=f0d4ad56c3e3d129d244615840c601ee
def policy(state, is_train):
    eps = 0.05
    if is_train and random.random() < eps:
        return random.choice(env.action_space)
    else:
        # 获取可以执行的action
        if state not in Q_table:
            Q_table[state] = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
            return random.choice(env.action_space)
        else:
            if Q_table[state][0] > Q_table[state][1]:
                return 0
            else:
                return 1


# 进行Q_table的训练
for i in range(episode):
    # trajectory = []
    state = env.reset()
    # print(info["action_mask"])
    # a = env.action_space.sample(info["action_mask"])
    action = policy(state, True)
    count = 0
    while True:
        state_next, reward, done = env.step(action)

        if state_next not in Q_table:
            Q_table[state_next] = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
        # plt.imshow(env.render())
        # plt.show()
        action_next = policy(state_next, True)

        Q_old = Q_table[state][action]
        Q_new = gamma * Q_table[state_next][action_next] + reward

        # 防止错误路径依赖
        # Q_new = gamma * Q_table[state_next][np.argmax(Q_table[state_next][np.where(info_next["action_mask"] == 1)[0]])] + reward
        Q_table[state][action] = alpha * (Q_new - Q_old) + Q_old

        if done :
            print("第{}次,执行了{}".format(i, count))
            count = 0
            break
        count += 1
        state = state_next
        action = action_next

state = env.reset()
print(state)

# 将对象序列化并保存到文件
# with open('version_1_table.pkl', 'wb') as file:
#     pickle.dump(Q_table, file)
new_state = state
while True:
    action = policy(new_state, False)
    print("action:" + str(action))
    new_state, reward, done = env.step(action)
    print(new_state, reward, done)
    if done:
        print("finish")
        break
