import random
import threading
from time import sleep

from FlappyBird.FlappyBirdEnv_local import FlappyBirdEnv
from FlappyBird.GamePad import GamePad
import socket

# 首先使用大语言模型，然后验证效果和检查api代码，验证可行性，进行修改
# if __name__ == '__main__':
#     pad = GamePad(400, 600)
#     pad.play(False)


if __name__ == '__main__':

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9999))
    # thread = threading.Thread(target=operation, args=(flappyBird,))
    # thread.start()
    frames = []
    i = 0
    while True:

        # rgb = flappyBird.render("FlappyBird0" + str(i))
        i = i + 1
        print(1)
        # frames.append(rgb)
        action = random.choice([0, 1])
        print("action:" + str(action))

        client.send(str(action).encode('utf-8'))
        sleep(1)
        # if not flappyBird.pad.game_active:
        #     break
        # response = client.recv(1024)
        # print(f"Received: {response.decode('utf-8')}")
        # new_state, reward, done = flappyBird.step(action)
        # print(new_state, reward, done)
        # if done:
        #     print("finish")
        #     break
