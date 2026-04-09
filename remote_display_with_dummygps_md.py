# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:50:44 2025

@author: nicho
"""

import socket
import cv2
import numpy as np
import pygame
import subprocess
import threading
import queue
import os
import time
from PIL import Image
import io
import json
import struct
# =====================
# Configuration
# =====================
LISTEN_IP = "0.0.0.0"
PORT = 5000
WIDTH = 1920
HEIGHT = 1080
ls_q = queue.Queue(10000)
ms_q = queue.Queue(10000)
rs_q = queue.Queue(10000)
frame_lock = threading.Lock()
latest_left = None
latest_mid = None
latest_right = None
CAMERA_CONFIGS = {
    "192.168.1.12": {
        "label": "left",
        "ssh_user": "ryan4",
        "queue": ls_q,
        "display_attr": "latest_left",
    },
    "192.168.1.11": {
        "label": "mid",
        "ssh_user": "ryan5",
        "queue": ms_q,
        "display_attr": "latest_mid",
    },
    "192.168.1.13": {
        "label": "right",
        "ssh_user": "ryan6",
        "queue": rs_q,
        "display_attr": "latest_right",
    },
}
def create_tcp_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind((LISTEN_IP, port))
    s.listen(1)
    s.settimeout(1.0)
    return s


def accept_client(server_sock, label):
    while not stop_event.is_set():
        try:
            conn, addr = server_sock.accept()
            conn.settimeout(1.0)
            print(f"{label} connected: {addr}")
            return conn, addr
        except socket.timeout:
            continue


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        except socket.timeout:
            continue
    return buf

# =====================
# UDP Socket
# =====================

stop_event = threading.Event()


def put_latest(q, item, label):
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            print(f"{label} queue full; dropped oldest frame")
            return True
        except queue.Full:
            print(f"{label} queue still full; dropped newest frame")
            return False

def init():
    for host, config in CAMERA_CONFIGS.items():
        remote_cmd = (
            f"cd /home/{config['ssh_user']}/Alex/Blackfly && "
            f"nohup python3 blkfly_md.py > /tmp/{config['label']}_blkfly_md.log 2>&1 &"
        )
        try:
            subprocess.Popen(["ssh", f"{config['ssh_user']}@{host}", remote_cmd])
            print(f"Started remote sender on {host}")
        except OSError as e:
            print(f"Failed to start remote sender on {host}: {e}")


def receiver(server_sock, label):
    global latest_left, latest_mid, latest_right

    while not stop_event.is_set():
        print(f"{label}: waiting for connection...")

        conn, addr = accept_client(server_sock, label)
        if conn is None:
            continue

        sender_ip, _ = addr
        config = CAMERA_CONFIGS.get(sender_ip)

        if config is None:
            print(f"{label}: unknown ip {sender_ip}, closing connection")
            conn.close()
            continue

        print(f"{label}: streaming from {sender_ip}")

        try:
            while not stop_event.is_set():
                # ---- Read frame length ----
                header = recv_exact(conn, 4)
                if header is None:
                    print(f"{label}: connection closed by client")
                    break

                frame_len = struct.unpack("!I", header)[0]
                print(f"{label}: got header frame_len={frame_len}")

                # ---- Read frame payload ----
                payload = recv_exact(conn, frame_len)
                if payload is None:
                    print(f"{label}: incomplete frame / disconnect")
                    break
                print(f"{label}: got payload bytes={len(payload)}")
                
                # ---- Decode ----
                npdata = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    print(f"{label}: decode failed")
                    continue

                # ---- Queue + display ----
                put_latest(config["queue"], payload, config["label"])

                with frame_lock:
                    if config["display_attr"] == "latest_left":
                        latest_left = frame
                    elif config["display_attr"] == "latest_mid":
                        latest_mid = frame
                    else:
                        latest_right = frame

        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"{label}: connection error: {e}")

        except Exception as e:
            print(f"{label}: unexpected error: {e}")

        finally:
            try:
                conn.close()
            except:
                pass

            with frame_lock:
                if label == "left":
                    latest_left = None
                elif label == "mid":
                    latest_mid = None
                else:
                    latest_right = None
            print(f"{label}: disconnected, restarting accept loop...")
            time.sleep(.2)  # prevent tight reconnect loop
        
'''def displayer():
    global latest_left, latest_mid, latest_right
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Stream")
        
    clock = pygame.time.Clock()
    running = True
    while not stop_event.is_set():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                with frame_lock:
                    left_img = latest_left
                    mid_img = latest_mid
                    right_img = latest_right
                if left_img is None or mid_img is None or right_img is None:
                    continue
                
                   
            
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
            mid_img = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
            left_text = 'left camera'
            mid_text = 'mid camera'
            right_text = 'right camera'
            text_color = (255,255,255)
            bg_color = (0,0,0)
            font = pygame.font.SysFont(None, 36)
            screen.fill(bg_color)
            l_text_surface = font.render(left_text, True, text_color)
            m_text_surface = font.render(mid_text, True, text_color)
            r_text_surface = font.render(right_text, True, text_color)
            left_surface = pygame.surfarray.make_surface(np.swapaxes(left_img, 0, 1))
            mid_surface = pygame.surfarray.make_surface(np.swapaxes(mid_img, 0, 1))
            right_surface = pygame.surfarray.make_surface(np.swapaxes(right_img, 0, 1))
            left_rect = l_text_surface.get_rect(center=(360,552))
            mid_rect = m_text_surface.get_rect(center = (1080, 552))
            right_rect = r_text_surface.get_rect(center = (1800, 552))
            screen.blit(l_text_surface, left_rect)
            screen.blit(m_text_surface, mid_rect)
            screen.blit(r_text_surface, right_rect)
            screen.blit(left_surface, (0,0))
            screen.blit(mid_surface, (640,0))
            screen.blit(right_surface, (1280,0))
            pygame.display.flip()
            clock.tick(30)
    print('displayer exiting cleanly')'''

def gps_dummy_sender():
    # Configuration matches blkfly2.py gps_listener
    TARGET_PORT = 5005
    BROADCAST_IP = '192.168.1.255' # Broadcast to the camera subnet
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    print(f"Starting GPS dummy broadcast on port {TARGET_PORT}")
    
    # Dummy starting values
    lat = 39.7589  
    lon = -84.1916
    alt = 225.0
    
    try:
        while not stop_event.is_set():
            gps_payload = {
                "left":  {"lat": lat, "lon": lon, "alt": alt},
                "mid":   {"lat": lat, "lon": lon, "alt": alt},
                "right": {"lat": lat, "lon": lon, "alt": alt}
            }
            
            try:
                message = json.dumps(gps_payload).encode('utf-8')
                sock.sendto(message, (BROADCAST_IP, TARGET_PORT))
                lat += 0.00001
                lon += 0.00001
            except Exception as e:
                print(f"GPS Sender Error: {e}")
                
            time.sleep(1)
    finally:
        sock.close()
        print("GPS sender exiting cleanly")
    
def saver():
    counters = {"left": 1, "mid": 1, "right": 1}
    paths = {
        "left": r"C:\SFM_IMAGES\left_cam",
        "mid": r"C:\SFM_IMAGES\mid_cam",
        "right": r"C:\SFM_IMAGES\right_cam",
    }
    source_queues = {"left": ls_q, "mid": ms_q, "right": rs_q}

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    while not stop_event.is_set():
        made_progress = False

        for label, q in source_queues.items():
            try:
                img_bytes = q.get(timeout=0.05)
            except queue.Empty:
                continue

            made_progress = True
            try:
                i = counters[label]
                file_path = os.path.join(paths[label], f"frame_{i}.jpg")
                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                counters[label] += 1
                print(f"saved {label} frame {i}")
            except OSError as e:
                print(f"Save error for {label} frame {counters[label]}: {e}")

        if not made_progress:
            time.sleep(0.05)

    print("saver exiting cleanly")

        
# Wait for the "READY" signal from remote


# =====================
# Pygame Init
# =====================
'''pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Live Stream")

clock = pygame.time.Clock()
frames = {}
camera_positions = {
    "192.168.1.12": (0, 0),
    "192.168.1.11": (640, 0),
    "192.168.1.13": (1280, 0)
}

# =====================
# Receive Loop
# =====================
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    data, addr = sock.recvfrom(65536)  # Max UDP size
    sender_ip, _ = addr
    # Decode JPEG
    npdata = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)

    if frame is None:
        continue

    # Convert to pygame surface
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    surface = pygame.surfarray.make_surface(
    np.swapaxes(frame_rgb, 0, 1))
    frames[sender_ip] = surface
    text = "left camera"
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0) 
    mid_text = "mid camera" 
    right_text = 'right camera'        # Black background

# Render the text surface
    font = pygame.font.SysFont(None, 36)
    left_surface = font.render(text, True, text_color)
    mid_surface = font.render(mid_text, True, text_color)
    right_surface = font.render(right_text, True, text_color)

# Get the rectangle of the text and center it
    left_rect = left_surface.get_rect(center=(360,552))
    mid_rect = mid_surface.get_rect(center = (1080, 552))
    right_rect = right_surface.get_rect(center = (1800, 552))
    
    screen.fill(bg_color)
    

    for ip, surface in frames.items():
        pos = camera_positions[ip]
        screen.blit(surface, pos)
        screen.blit(left_surface, left_rect)
        screen.blit(mid_surface, mid_rect)
        screen.blit(right_surface, right_rect)
    pygame.display.flip()

    clock.tick(30)  # Display cap (not stream cap)

pygame.quit()'''

def de_init():
    for host, config in CAMERA_CONFIGS.items():
        try:
            result = subprocess.run(
                ["ssh", f"{config['ssh_user']}@{host}", "pgrep -f blkfly_md.py"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as e:
            print(f"Failed to query remote PIDs on {host}: {e}")
            continue

        pids = result.stdout.strip().split()
        if not pids:
            continue

        for _ in range(2):
            try:
                subprocess.run(
                    ["ssh", f"{config['ssh_user']}@{host}", "kill -2 " + " ".join(pids)],
                    timeout=10,
                )
            except Exception as e:
                print(f"Failed to stop remote script on {host}: {e}")
            time.sleep(1)
        print(f"Killed remote script PIDs on {host}: {', '.join(pids)}")
    
    
def main():
    global l_sock, m_sock, r_sock
    l_sock = create_tcp_server(5001)
    m_sock = create_tcp_server(5002)
    r_sock = create_tcp_server(5000)

    l_rec_thread = threading.Thread(target=receiver, args=(l_sock, "left"), daemon=True)
    m_rec_thread = threading.Thread(target=receiver, args=(m_sock, "mid"), daemon=True)
    r_rec_thread = threading.Thread(target=receiver, args=(r_sock, "right"), daemon=True)
    save_thread = threading.Thread(target=saver, daemon = True)
    #disp_thread = threading.Thread(target = displayer, daemon = True)
    gps_thread = threading.Thread(target=gps_dummy_sender, daemon=True)
    l_rec_thread.start()
    m_rec_thread.start()
    r_rec_thread.start()
    init()
    save_thread.start()
    #disp_thread.start()
    gps_thread.start()
    global latest_left, latest_mid, latest_right
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Stream")
        
    clock = pygame.time.Clock()
    running = True
    try:
        while running and not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    stop_event.set()
            with frame_lock:
                left_img = latest_left
                mid_img = latest_mid
                right_img = latest_right
            if left_img is None and mid_img is None and right_img is None:
                print('all cameras failing to send')
                clock.tick(30)
                continue
            
            left_text = 'left camera'
            mid_text = 'mid camera'
            right_text = 'right camera'
            text_color = (255,255,255)
            bg_color = (0,0,0)
            font = pygame.font.SysFont(None, 36)
            screen.fill(bg_color)
            if left_img is not None:
                left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
                left_surface = pygame.surfarray.make_surface(np.swapaxes(left_img, 0, 1))
                screen.blit(left_surface, (0,0))
            if mid_img is not None:
                mid_img = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2RGB)
                mid_surface = pygame.surfarray.make_surface(np.swapaxes(mid_img, 0, 1))
                screen.blit(mid_surface, (640,0))
            if right_img is not None:
                right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
                right_surface = pygame.surfarray.make_surface(np.swapaxes(right_img, 0, 1))
                screen.blit(right_surface, (1280,0))
            l_text_surface = font.render(left_text, True, text_color)
            m_text_surface = font.render(mid_text, True, text_color)
            r_text_surface = font.render(right_text, True, text_color)
            left_rect = l_text_surface.get_rect(center=(360,552))
            mid_rect = m_text_surface.get_rect(center = (1080, 552))
            right_rect = r_text_surface.get_rect(center = (1800, 552))
            screen.blit(l_text_surface, left_rect)
            screen.blit(m_text_surface, mid_rect)
            screen.blit(r_text_surface, right_rect)
            pygame.display.flip()
            clock.tick(144)
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as e:
        stop_event.set()
        print(f"Display loop error: {e}")
    finally:
        stop_event.set()
        pygame.quit()
        l_sock.close()
        m_sock.close()
        r_sock.close()
        time.sleep(1)
        de_init()
        l_rec_thread.join(timeout=5)
        m_rec_thread.join(timeout=5)
        r_rec_thread.join(timeout=5)
        save_thread.join(timeout=5)
        gps_thread.join(timeout=5)
        print('main exiting')
        

if __name__ == "__main__":
    main()


