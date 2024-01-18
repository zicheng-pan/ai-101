import pickle
import random


class Policy():

    def __init__(self, Q_table_file):
        # 从文件中加载序列化的对象
        with open(Q_table_file, 'rb') as file:
            self.Q_table = pickle.load(file)

    def dopolicy(self, state):
        eps = 0.05
        # 获取可以执行的action
        if state not in self.Q_table:
            self.Q_table[state] = [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]
            return random.choice([0, 1])
        else:
            if self.Q_table[state][0] > self.Q_table[state][1]:
                return 0
            else:
                return 1
