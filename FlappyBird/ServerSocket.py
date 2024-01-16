import socket
import threading

def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        # 处理接收到的数据
        print(f"Received: {data.decode('utf-8')}")
        # 可以在这里发送回复
        client_socket.send("Ack!".encode('utf-8'))
    client_socket.close()

def start_server(host='localhost', port=9999):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Listening on {host}:{port}...")

    while True:
        client_sock, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_sock,))
        client_thread.start()

if __name__ == "__main__":
    start_server()
