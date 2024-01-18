import json
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
from FlappyBird.TrainPolicy import Policy

if __name__ == '__main__':

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9999))
    client.send(str("reset").encode('utf-8'))
    msg = client.recv(1024).decode('utf-8')
    state = tuple(json.loads(msg))

    policy = Policy("version_Q_table.pkl")


    # thread = threading.Thread(target=operation, args=(flappyBird,))
    # thread.start()
    frames = []
    i = 0

    while True:
        # rgb = flappyBird.render("FlappyBird0" + str(i))
        i = i + 1
        print(1)
        # frames.append(rgb)

        print("action:" + str(policy.dopolicy(state)))

        client.send(str(policy.dopolicy(state)).encode('utf-8'))
        msg = client.recv(1024).decode('utf-8')
        state, r, is_collision = json.loads(msg)
        state = tuple(state)
        print(state)
        sleep(0.2)

        if is_collision:
            break

    client.send("close".encode('utf-8'))
    client.close()