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
STAT_PORT1 = 6000
STAT_PORT2 = 6001
STAT_PORT3 = 6002
WIDTH = 1920
HEIGHT = 1080
raw_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}

decoded_q = {
    "left": queue.Queue(100),
    "mid": queue.Queue(100),
    "right": queue.Queue(100),
}
save_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}
frame_lock = threading.Lock()
stats_lock = threading.Lock()
latest_left = None
latest_mid = None
latest_right = None
m_capt = None
m_enc = None
m_stream = None
m_save = None
m_time = None
m_exif = None
m_send = None
l_capt = None
l_enc = None
l_stream = None
l_save = None
l_time = None
l_exif = None
l_send = None
r_capt = None
r_enc = None
r_stream = None
r_save = None
r_time = None
r_exif = None
r_send = None
CAMERA_CONFIGS = {
    "192.168.1.12": {"label": "left", "ssh_user": "ryan4"},
    "192.168.1.11": {"label": "mid",  "ssh_user": "ryan5"},
    "192.168.1.13": {"label": "right","ssh_user": "ryan6"},
}

def stat_thread(STAT_PORT, label):
    global m_capt, m_save, m_send, m_enc, m_time, m_exif, m_stream, l_capt, l_enc, l_send, l_stream, l_save, l_exif, l_time, r_capt, r_enc, r_send, r_stream, r_time, r_exif, r_save
    stat_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stat_server.bind((LISTEN_IP, STAT_PORT))
    stat_server.listen(5)
    conn, addr = stat_server.accept()
    print('connected:', addr)
    buffer = ""
    while not stop_event.is_set():
        data = conn.recv(4096)

        if not data:
            break
        buffer += data.decode()

        while "\n" in buffer:
            stats, buffer = buffer.split("\n", 1)

            parts = [p.strip() for p in stats.split(",")]

            if len(parts) != 8:
                print(f"{label}: malformed stats: {stats}")
                continue

            fps_label = parts[0]
            capture = str(parts[1].split("=")[1])
            encode = str(parts[2].split("=")[1])
            send = str(parts[3].split("=")[1])
            save_q = str(parts[4].split("=")[1])
            stream_q = str(parts[5].split("=")[1])
            exif_q = str(parts[6].split("=")[1])
            time_q = str(parts[7].split("=")[1])
            with stats_lock:
                if label =="mid":
                    m_capt = capture
                    m_enc = encode
                    m_stream = stream_q
                    m_save = save_q
                    m_time = time_q
                    m_exif = exif_q
                    m_send = send
                elif label =="left":
                    l_capt = capture
                    l_enc = encode
                    l_stream = stream_q
                    l_save = save_q
                    l_time = time_q
                    l_exif = exif_q
                    l_send = send
                else:
                    r_capt = capture
                    r_enc = encode
                    r_stream = stream_q
                    r_save = save_q
                    r_time = time_q
                    r_exif = exif_q
                    r_send = send
        #print(data.decode(), end="")
    
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


def recv_exact(sock, n, timeout = 2.0):
    buf = b''
    sock.settimeout(timeout)
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
            f"nohup python3 blkfly_md435.py > /tmp/{config['label']}_blkfly_md435.log 2>&1 &"
        )
        try:
            subprocess.Popen(["ssh", f"{config['ssh_user']}@{host}", remote_cmd])
            print(f"Started remote sender on {host}")
        except OSError as e:
            print(f"Failed to start remote sender on {host}: {e}")


def receiver(server_sock, label):
    global latest_left, latest_mid, latest_right
    frame_count = 0
    last_time = time.time()
    while not stop_event.is_set():
        print(f"{label}: waiting for connection...")
        conn, addr = accept_client(server_sock, label)

        try:
            while not stop_event.is_set():
                header = recv_exact(conn, 4)
                if header is None:
                    break

                frame_len = struct.unpack("!I", header)[0]

                payload = recv_exact(conn, frame_len)
                if payload is None:
                    break

                # ONLY STORE RAW PACKET
                q = raw_q[label]
                if q.full():
                    try:
                        q.get_nowait()
                    except:
                        pass
                q.put_nowait(payload)
                frame_count += 1
                now = time.time()
                elapsed = now-last_time
                if elapsed >= 3:
                    fps = frame_count/elapsed
                    print(f"{label} camera recieving {fps} fps")
                    frame_count = 0
                    last_time = now
                with frame_lock:
                    if label == "left":
                        latest_left = payload 
                    elif label == "mid":
                        latest_mid = payload
                    else:
                        latest_right = payload

        except Exception as e:
            print(f"{label} receiver error: {e}")
        finally:
            conn.close()

def decoder(label):
    global latest_left, latest_mid, latest_right

    while not stop_event.is_set():
        try:
            payload = raw_q[label].get(timeout=0.1)
        except queue.Empty:
            continue

        npdata = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            continue

        # update display buffer


        # forward to saver (optional)
        if not save_q[label].full():
            save_q[label].put(payload)
        
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
        "left": "road_test_2/left_cam",
        "mid": "road_test_2/mid_cam",
        "right": "road_test_2/right_cam",
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    while not stop_event.is_set():
        made_progress = False

        for label in ["left", "mid", "right"]:
            try:
                payload = save_q[label].get(timeout=0.05)
            except queue.Empty:
                continue

            made_progress = True
            i = counters[label]
            file_path = os.path.join(paths[label], f"frame_{i}.jpg")

            try:
                with open(file_path, "wb") as f:
                    f.write(payload)
                counters[label] += 1
            except Exception as e:
                print(f"saver error {label}: {e}")

        if not made_progress:
            time.sleep(0.02)

        
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
        if event.type == pyg+ame.QUIT:
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
                ["ssh", f"{config['ssh_user']}@{host}", "pgrep -f blkfly_md435.py"],
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
    global l_sock, m_sock, r_sock, STAT_PORT1, STAT_PORT2, STAT_PORT3
    l_sock = create_tcp_server(5001)
    m_sock = create_tcp_server(5002)
    r_sock = create_tcp_server(5000)

    l_rec_thread = threading.Thread(target=receiver, args=(l_sock, "left"), daemon=True)
    m_rec_thread = threading.Thread(target=receiver, args=(m_sock, "mid"), daemon=True)
    r_rec_thread = threading.Thread(target=receiver, args=(r_sock, "right"), daemon=True)

    l_dec_thread = threading.Thread(target=decoder, args=("left",), daemon=True)
    m_dec_thread = threading.Thread(target=decoder, args=("mid",), daemon=True)
    r_dec_thread = threading.Thread(target=decoder, args=("right",), daemon=True)

    save_thread = threading.Thread(target=saver, daemon = True)
    #disp_thread = threading.Thread(target = displayer, daemon = True)
    gps_thread = threading.Thread(target=gps_dummy_sender, daemon=True)
    m_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT1, "mid"), daemon=True)
    l_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT2, "left"), daemon=True)
    r_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT3, "right"), daemon=True)
    l_rec_thread.start()
    m_rec_thread.start()
    r_rec_thread.start()

    l_dec_thread.start()
    m_dec_thread.start()
    r_dec_thread.start()
    m_stats_thread.start()
    l_stats_thread.start()
    r_stats_thread.start()


    init()
    save_thread.start()
    #disp_thread.start()
    gps_thread.start()
    global latest_left, latest_mid, latest_right, l_capt, l_send, l_enc, l_stream, l_save, l_exif, l_time, m_capt, m_send, m_enc, m_stream, m_save, m_exif, m_time, r_capt, r_enc, r_send, r_stream, r_save, r_exif, r_time
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
                left_img = None
                mid_img = None
                right_img = None
                left_bytes = latest_left
                mid_bytes = latest_mid
                right_bytes = latest_right
            if left_bytes is None and mid_bytes is None and right_bytes is None:
                print('all cameras failing to send')
                clock.tick(30)
                continue
            if left_bytes is not None:
                npdata = np.frombuffer(left_bytes, dtype=np.uint8)
                left_img = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
            else:
                #print('left camera failing')
                pass
            if mid_bytes is not None:
                npdata = np.frombuffer(mid_bytes, dtype=np.uint8)
                mid_img = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
            else:
                #print('mid camera failing')
                pass
            if right_bytes is not None:
                npdata = np.frombuffer(right_bytes, dtype=np.uint8)
                right_img = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
            else:
                #print('right camera failing')
                pass
            
            left_text = 'left camera'
            l_capture = f"capture: {l_capt} FPS"
            l_encode = f"encode: {l_enc} FPS"
            l_sent = f"send: {l_send} FPS"
            l_streamq = f"stream_q: {l_stream} frames"
            l_saveq = f"save_q: {l_save} frames"
            l_timeq = f"time_q: {l_time} timestamps"
            l_exifq = f"exif_q: {l_exif} arrays"
            mid_text = 'mid camera'
            m_capture = f"capture: {m_capt} FPS"
            m_encode = f"encode: {m_enc} FPS"
            m_sent = f"send: {m_send} FPS"
            m_streamq = f"stream_q: {m_stream} frames"
            m_saveq = f"save_q: {m_save} frames"
            m_timeq = f"time_q: {m_time} timestamps"
            m_exifq = f"exif_q: {m_exif} arrays"
            right_text = 'right camera'
            r_capture = f"capture: {r_capt} FPS"
            r_encode = f"encode: {r_enc} FPS"
            r_sent = f"send: {r_send} FPS"
            r_streamq = f"stream_q: {r_stream} frames"
            r_saveq = f"save_q: {r_save} frames"
            r_timeq = f"time_q: {r_time} timestamps"
            r_exifq = f"exif_q: {r_exif} arrays"
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
            lcs = font.render(l_capture, True, text_color)
            les = font.render(l_encode, True, text_color)
            lss = font.render(l_sent, True, text_color)
            lsts = font.render(l_streamq, True, text_color)
            lsas = font.render(l_saveq, True, text_color)
            lts = font.render(l_timeq, True, text_color)
            lexs = font.render(l_exifq, True, text_color)
            m_text_surface = font.render(mid_text, True, text_color)
            mcs = font.render(m_capture, True, text_color)
            mes = font.render(m_encode, True, text_color)
            mss = font.render(m_sent, True, text_color)
            msts = font.render(m_streamq, True, text_color)
            msas = font.render(m_saveq, True, text_color)
            mts = font.render(m_timeq, True, text_color)
            mexs = font.render(m_exifq, True, text_color)
            r_text_surface = font.render(right_text, True, text_color)
            rcs = font.render(r_capture, True, text_color)
            res = font.render(r_encode, True, text_color)
            rss = font.render(r_sent, True, text_color)
            rsts = font.render(r_streamq, True, text_color)
            rsas = font.render(r_saveq, True, text_color)
            rts = font.render(r_timeq, True, text_color)
            rexs = font.render(r_exifq, True, text_color)
            left_rect = l_text_surface.get_rect(center=(360,552))
            lcr = lcs.get_rect(center=(360,600))
            ler = les.get_rect(center=(360,650))
            lsr = lss.get_rect(center=(360,700))
            lstr = lsts.get_rect(center=(360,750))
            lsar = lsas.get_rect(center=(360,800))
            ltr = lts.get_rect(center=(360,850))
            lexr = lexs.get_rect(center=(360,900))
            mid_rect = m_text_surface.get_rect(center = (1080, 552))
            mcr = mcs.get_rect(center=(1080,600))
            mer = mes.get_rect(center=(1080,650))
            msr = mss.get_rect(center=(1080,700))
            mstr = msts.get_rect(center=(1080,750))
            msar = msas.get_rect(center=(1080,800))
            mtr = mts.get_rect(center=(1080,850))
            mexr = mexs.get_rect(center=(1080,900))
            right_rect = r_text_surface.get_rect(center = (1800, 552))
            rcr = rcs.get_rect(center=(1800,600))
            rer = res.get_rect(center=(1800,650))
            rsr = rss.get_rect(center=(1800,700))
            rstr = rsts.get_rect(center=(1800,750))
            rsar = rsas.get_rect(center=(1800,800))
            rtr = rts.get_rect(center=(1800,850))
            rexr = rexs.get_rect(center=(1800,900))
            screen.blit(l_text_surface, left_rect)
            screen.blit(lcs,lcr)
            screen.blit(les,ler)
            screen.blit(lss,lsr)
            screen.blit(lsts, lstr)
            screen.blit(lsas,lsar)
            screen.blit(lts, ltr)
            screen.blit(lexs,lexr)
            screen.blit(m_text_surface, mid_rect)
            screen.blit(mcs,mcr)
            screen.blit(mes,mer)
            screen.blit(mss,msr)
            screen.blit(msts,mstr)
            screen.blit(msas,msar)
            screen.blit(mts,mtr)
            screen.blit(mexs,mexr)
            screen.blit(r_text_surface, right_rect)
            screen.blit(rcs,rcr)
            screen.blit(res,rer)
            screen.blit(rss,rsr)
            screen.blit(rsts,rstr)
            screen.blit(rsas, rsar)
            screen.blit(rts,rtr)
            screen.blit(rexs,rexr)
            pygame.display.flip()
            clock.tick(30)
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
        m_stats_thread.join(timeout=5)
        l_stats_thread.join(timeout=5)
        r_stats_thread.join(timeout=5)
        print('main exiting')
        

if __name__ == "__main__":
    main()
