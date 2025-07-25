# This script is used to test the broadcast speed of the interntet switch

# RECEIVING MACHINE

import socket
import json

UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(4096)
    try:
        msg = json.loads(data.decode())
        print(f"Received from {addr}: {msg}")
        if "timestamp" in msg:
            ack = {"timestamp": msg["timestamp"]}
            sock.sendto(json.dumps(ack).encode(), addr)
            print(f"Sent ACK to {addr}")
    except Exception as e:
        print(f"Error: {e}")