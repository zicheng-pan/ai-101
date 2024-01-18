import pygame
import socket
import threading

def handle_client(client_socket):
    while True:
        msg = client_socket.recv(1024).decode('utf-8')
        if msg:
            if msg == 'SPACE':
                simulate_key_press(pygame.K_SPACE)

def simulate_key_press(key):
    fake_event = pygame.event.Event(pygame.KEYDOWN, key=key)
    pygame.event.post(fake_event)


# 设置 socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 9999))
server.listen(1)

print("Waiting for a connection...")
client_sock, addr = server.accept()
print(f"Connected to {addr}")

# 创建并启动一个新线程来处理客户端
thread = threading.Thread(target=handle_client, args=(client_sock,))
thread.start()

# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("空格键被按下")

    pygame.display.flip()

pygame.quit()
client_sock.close()
server.close()
