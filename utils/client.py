import socket
import sys
import json
import redis



def 




HOST, PORT = "damsl2.cs.jhu.edu", 80

# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    data = sys.argv[1]
    # Connect to server and send data
    sock.connect((HOST, PORT))

    sock.sendall(bytes(json.dumps(data) + "\n", "utf-8"))

    # Receive data from the server and shut down
    received = str(sock.recv(1024), "utf-8")
finally:
    sock.close()

print("Sent:     {}".format(data))
print("Received: {}".format(received))