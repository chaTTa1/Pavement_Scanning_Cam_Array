# This script is used to test the broadcast speed of the interntet switch

# BROADCASTING MACHINE

import socket
import time
import json

UDP_IP = "192.168.1.255"  # Broadcast or use receiver's IP directly
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.settimeout(2.0)

message = {
    "test": "ping",
    "timestamp": time.time()
}

start_time = time.perf_counter()
sock.sendto(json.dumps(message).encode(), (UDP_IP, UDP_PORT))

try:
    data, addr = sock.recvfrom(1024)
    ack = json.loads(data.decode())
    if "timestamp" in ack:
        rtt = (time.time() - ack["timestamp"]) * 1000  # ms
        print(f"[TIMER] RTT to {addr}: {rtt:.2f} ms")
    else:
        print("[WARN] ACK received but no timestamp found.")
except socket.timeout:
    print("[WARN] No ACK received")
end_time = time.perf_counter()
print(f"Total elapsed: {(end_time - start_time)*1000:.2f} ms")