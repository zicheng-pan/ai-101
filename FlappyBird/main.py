import random
from time import sleep

from FlappyBird.FlappyBirdEnv import FlappyBirdEnv
from FlappyBird.GamePad import GamePad

# 首先使用大语言模型，然后验证效果和检查api代码，验证可行性，进行修改
# if __name__ == '__main__':
#     pad = GamePad(400, 600)
#     pad.play(False)

if __name__ == '__main__':
    flappyBird = FlappyBirdEnv()
    state = flappyBird.reset()
    print(state)
    while True:
        action = random.choice([0, 1])
        print("action:" + str(action))
        new_state, reward, done = flappyBird.step(action)
        print(new_state, reward, done)
        if done:
            print("finish")
            break
