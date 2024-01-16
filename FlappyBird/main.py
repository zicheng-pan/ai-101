from time import sleep

from FlappyBird.FlappyBirdEnv import FlappyBirdEnv
from FlappyBird.GamePad import GamePad

# 首先使用大语言模型，然后验证效果和检查api代码，验证可行性，进行修改
if __name__ == '__main__':
    pad = GamePad(400, 600)
    pad.play(False)

# if __name__ == '__main__':
#     flappyBird = FlappyBirdEnv()
#     flappyBird.reset()
#     flappyBird.render()
#     flappyBird.step(1)
#     flappyBird.step(1)
