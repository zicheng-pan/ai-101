import socket

def start_client(host='localhost', port=9999):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    while True:
        message = input("Enter message: ")
        client.send(message.encode('utf-8'))
        response = client.recv(1024)
        print(f"Received: {response.decode('utf-8')}")

if __name__ == "__main__":
    start_client()
