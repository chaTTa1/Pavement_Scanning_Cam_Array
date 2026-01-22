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
# =====================
# Configuration
# =====================
LISTEN_IP = "0.0.0.0"
PORT = 5000
WIDTH = 1920
HEIGHT = 1080
left_q = queue.Queue(1)
mid_q = queue.Queue(1)
right_q = queue.Queue(1)
ls_q = queue.Queue(10000)
ms_q = queue.Queue(10000)
rs_q = queue.Queue(10000)
frame_lock = threading.Lock()
latest_left = None
latest_mid = None
latest_right = None

# =====================
# UDP Socket
# =====================

stop_event = threading.Event()

def init():
    ssh_cmd = [
        "ssh",
        "ryan4@192.168.1.12",
        "cd /home/ryan4/Alex/Blackfly && nohup python3 blkfly2.py > /tmp/12_17.log 2>&1 &"
    ]
    ssh_cmdr5 = [
        "ssh",
        "ryan5@192.168.1.11",
        "cd /home/ryan5/Alex/Blackfly && nohup python3 blkfly2.py > /tmp/12_17.log 2>&1 &"
        ]
    
    ssh_cmd6 = [
        "ssh",
        "ryan6@192.168.1.13",
        "cd /home/ryan6/Alex/Blackfly && nohup python3 blkfly2.py > /tmp/12_17.log 2>&1 &"]

    # Run the SSH command and capture output

    subprocess.Popen(ssh_cmd)
    subprocess.Popen(ssh_cmdr5)
    subprocess.Popen(ssh_cmd6)


def reciever():
    global latest_left, latest_mid, latest_right
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65536)
            print('revieved from ', addr)
        except socket.timeout:
            print('no data recieved')
            continue
        except OSError:
            print('OSerror')
            break
        sender_ip, _ = addr
        npdata = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_GRAYSCALE)
        if sender_ip == "192.168.1.12":
            ls_q.put(frame, timeout = 5)
            with frame_lock:
                latest_left = frame
        elif sender_ip == "192.168.1.11":
            ms_q.put(frame, timeout = 5)
            with frame_lock:
                latest_mid = frame
        elif sender_ip == "192.168.1.13":
            rs_q.put(frame, timeout = 5)
            with frame_lock:
                latest_right = frame
        else:
            print("unknown ip address")
    print ('reciever exiting cleanly')
        
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
def saver():
    i = 1
    while not stop_event.is_set():
        left_file_path = r"C:\SFM_IMAGES\left_cam"
        mid_file_path = r"C:\SFM_IMAGES\mid_cam"
        right_file_path = r"C:\SFM_IMAGES\right_cam"
        os.makedirs(left_file_path, exist_ok=True)
        os.makedirs(mid_file_path, exist_ok=True)
        os.makedirs(right_file_path, exist_ok=True)
        l_name = 'frame'
        m_name = 'frame'
        r_name = 'frame'
        ext = '.jpg'
        while not stop_event.is_set():
            try:
                l_img = ls_q.get(timeout =1)
                m_img = ms_q.get(timeout = 1)
                r_img = rs_q.get(timeout = 1)
            except queue.Empty:
                continue
            l_file_path = os.path.join(left_file_path, f"{l_name}_{i}{ext}")
            m_file_path = os.path.join(mid_file_path, f"{m_name}_{i}{ext}")
            r_file_path = os.path.join(right_file_path, f"{r_name}_{i}{ext}")
            cv2.imwrite(l_file_path, l_img)
            cv2.imwrite(m_file_path, m_img)
            cv2.imwrite(r_file_path, r_img)
            i = i+1
        time.sleep(.5)
    print('saver exiting cleanly')

        
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
    result = subprocess.run(
        ["ssh", "ryan4@192.168.1.12", "pgrep -f blkfly2.py"],
        capture_output=True, text=True
    )
    
    pids = result.stdout.strip().split()
    if pids:
        subprocess.run([
            "ssh",
            "ryan4@192.168.1.12",
            "kill -2 " + " ".join(pids)
        ])
        time.sleep(3)
        subprocess.run([
            "ssh",
            "ryan4@192.168.1.12",
            "kill -2 " + " ".join(pids)
        ])
        print(f"Killed remote script PIDs: {', '.join(pids)}")
    result5 = subprocess.run(
        ["ssh", "ryan5@192.168.1.11", "pgrep -f blkfly2.py"],
        capture_output=True, text = True
        )
    pids = result5.stdout.strip().split()
    if pids:
        subprocess.run([
            "ssh",
            "ryan5@192.168.1.11",
            "kill -2 " + " ".join(pids)
            ])
        time.sleep(3)
        subprocess.run([
            "ssh",
            "ryan5@192.168.1.11",
            "kill -2 " + " ".join(pids)
            ])
        print(f"Killed remote script PIDs: {', '.join(pids)}")
    result6 = subprocess.run([
        "ssh",
        "ryan6@192.168.1.13",
        "pgrep -f blkfly2.py"],
        capture_output=True,text = True
        )
    pids = result6.stdout.strip().split()
    if pids:
        subprocess.run([
            "ssh",
            "ryan6@192.168.1.13",
            "kill -2 " + " ".join(pids)
            ])
        time.sleep(3)
        subprocess.run([
            "ssh",
            "ryan6@192.168.1.13",
            "kill -2 " + " ".join(pids)
            ])
        print(f"Killed remote script PIDs: {', '.join(pids)}")
    
    
def main():
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LISTEN_IP, PORT))
    sock.settimeout(.5)
    init()
    rec_thread = threading.Thread(target=reciever, daemon = True)
    save_thread = threading.Thread(target=saver, daemon = True)
    #disp_thread = threading.Thread(target = displayer, daemon = True)
    rec_thread.start()
    save_thread.start()
    #disp_thread.start()
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
            if left_img is None or mid_img is None or right_img is None:
                print('no images to display')
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
            clock.tick(144)
    except KeyboardInterrupt:
        stop_event.set()
        
    finally:
        pygame.quit()
        sock.close()
        time.sleep(1)
        de_init()
        rec_thread.join()
        #disp_thread.join()
        save_thread.join()
        print('main exiting')
        

if __name__ == "__main__":
    main()


