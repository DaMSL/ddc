import socket
import sys
import json
import redis

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)



HOST, PORT = "damsl2.cs.jhu.edu", 80
REDIS_HOST, REDIS_PORT = 'localhost', 6379

# Create a socket (SOCK_STREAM means a TCP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

catalog = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT)

try:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    logging.info("Connected to %s:%d", HOST, PORT)
    entrylist = [e.decode() for e in catalog.lrange('globalconvergelist', 0, -1)]
    logging.info("Sending %d entries...", len(entrylist))
    for i, entry in enumerate(entrylist):
      sock.send(bytes(entry + "\n", "utf-8"))
      logging.debug("Sent #%d", i)
      received = str(sock.recv(1024), "utf-8")
      logging.debug("  REsponse: %s", received)

    # Receive data from the server and shut down
finally:
    sock.close()

print("Sent:     {}".format(data))
print("Received: {}".format(received))