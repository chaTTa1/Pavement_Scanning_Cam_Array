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
# Window size to display images on
WIDTH = 1920
HEIGHT = 1080

# Contains the raw bytes recieved from the jetsons
raw_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}

# Contains the decoded frames to display on screen
decoded_q = {
    "left": queue.Queue(100),
    "mid": queue.Queue(100),
    "right": queue.Queue(100),
}

# Contains the decoded frames to be saved to disk
save_q = {
    "left": queue.Queue(500),
    "mid": queue.Queue(500),
    "right": queue.Queue(500),
}


frame_lock = threading.Lock()
stats_lock = threading.Lock()

# Latest_left/mid/right are the updated images from each camera to display
latest_left = None
latest_mid = None
latest_right = None

# each camera stat m_capt = middle camera's image capture rate
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
l_rec = None
m_rec = None
r_rec = None

CAMERA_CONFIGS = {
    "192.168.1.12": {"label": "left", "ssh_user": "ryan4"},
    "192.168.1.11": {"label": "mid",  "ssh_user": "ryan5"},
    "192.168.1.13": {"label": "right","ssh_user": "ryan6"},
}


# ===============
# Camera Stats
# ===============
def stat_thread(STAT_PORT, label):
    global m_capt, m_save, m_send, m_enc, m_time, m_exif, m_stream, l_capt, l_enc, l_send, l_stream, l_save, l_exif, l_time, r_capt, r_enc, r_send, r_stream, r_time, r_exif, r_save
    
    # creates a socket to receive information from the jetson
    stat_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stat_server.bind((LISTEN_IP, STAT_PORT))
    stat_server.listen(5)
    conn, addr = stat_server.accept()
    print('connected:', addr)
    buffer = ""
    
    # recieves the camera statistics from the jetson
    while not stop_event.is_set():
        data = conn.recv(4096)

        if not data:
            break
        buffer += data.decode()
        
        # separtates the fields of the message on the comma ","
        while "\n" in buffer:
            stats, buffer = buffer.split("\n", 1)

            parts = [p.strip() for p in stats.split(",")]
            
            # checks to ensure that the proper number of fields have been recieved before moving on
            if len(parts) != 8:
                print(f"{label}: malformed stats: {stats}")
                continue
            
            # locates each number by splitting the field on the equals sign "=" and taking the 1 index [1]
            #fps_label = parts[0]
            capture = str(parts[1].split("=")[1])
            encode = str(parts[2].split("=")[1])
            send = str(parts[3].split("=")[1])
            save_q = str(parts[4].split("=")[1])
            stream_q = str(parts[5].split("=")[1])
            #exif_q = str(parts[6].split("=")[1])
            #time_q = str(parts[7].split("=")[1])
            
            # puts the recieved camera stats into their corresponding global variables
            with stats_lock:
                if label =="mid":
                    m_capt = capture
                    m_enc = encode
                    m_stream = stream_q
                    m_save = save_q
                    #m_time = time_q
                    #m_exif = exif_q
                    m_send = send
                elif label =="left":
                    l_capt = capture
                    l_enc = encode
                    l_stream = stream_q
                    l_save = save_q
                    #l_time = time_q
                    #l_exif = exif_q
                    l_send = send
                else:
                    r_capt = capture
                    r_enc = encode
                    r_stream = stream_q
                    r_save = save_q
                    #r_time = time_q
                    #r_exif = exif_q
                    r_send = send
        #print(data.decode(), end="")
 
# ====================
# TCP server creation
# ====================
def create_tcp_server(port):
    # Creates a TCP server using the inputed port and using "0.0.0.0" for the IP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.bind((LISTEN_IP, port))
    s.listen(1)
    s.settimeout(1.0)
    return s

# =========================
# accepting the connection
# =========================
def accept_client(server_sock, label):
    # attempts to establish the socket connection for the inputed server_sock
    while not stop_event.is_set():
        try:
            conn, addr = server_sock.accept()
            conn.settimeout(1.0)
            print(f"{label} connected: {addr}")
            return conn, addr
        except socket.timeout:
            continue

# ==============
# receive exact
# ==============
def recv_exact(sock, n, timeout = 2.0):
    # creates the buffer and socket timeout
    buf = b''
    sock.settimeout(timeout)
    # while the buffer is shorter than the expected package chunks that are received will be added to the buffer
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

# ======================
# inserting into Queues
# ======================
def put_latest(q, item, label):
    # attempts to insert a given item into a given Queue
    # if the Queue is full it will pull out the oldest frame and drop it
    # if the Queue is still full after dropping the oldest frame it will also drop the newest frame
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

# ==============
# initilization
# ==============
def init():
    # goes through all camera configurations and will SSH each camera's computer
    # the path in remote_cmd can be modified to target different variations of the blackfly script
    # blkfly_md can run at 200 fps
    # blkfly_md435 can run at 435 fps
    # this requires that an SSH key exists on the target computer that is generated by the host machine
    # without this a password is required which cannot be done with this code
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

# ==============
# receiver
# ==============
def receiver(server_sock, label):
    global latest_left, latest_mid, latest_right, l_rec, m_rec, r_rec
    frame_count = 0
    last_time = time.time()
    
    # when the code runs it will attempt to connect to the camera computers
    while not stop_event.is_set():
        print(f"{label}: waiting for connection...")
        conn, addr = accept_client(server_sock, label)
        
        # after connecting it will begin receiving images from the cameras
        # first a 4 byte header will come to give the length of the incoming frame
        # once received the header will be decoded into the frame length and used to inform recv_exact of how long the frame is
        # once the frame is fully received the raw bytes will be put into a queue to be processed elsewhere
        # the frame count is also kept along with the elapsed time to help calculate the frame rate received
        # and the bytes will be put into their respective latest_left/mid/right global variables for displaying
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
                if elapsed >= 1:
                    with stats_lock:
                        if label == 'left':
                            l_rec = frame_count/elapsed
                        elif label == 'mid':
                            m_rec = frame_count/elapsed
                        else:
                            r_rec = frame_count/elapsed
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


# ===============
# decoder
# ===============
def decoder(label):
    global latest_left, latest_mid, latest_right
    # when the code runs this attempts to pull items from the raw Queue corresponding to the inputed label
    # it then uses np.frombuffer and cv2.imdecode to convert the raw bytes into a grayscale .jpg file
    # the .jpg file is then sent to the corresponding save Queue to be written to the disk
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

# ==================
# dummy gps sender 
# ==================
def gps_dummy_sender():
    # optional function to send dummy gps coordinates to the cameras to test live embedding of the coordinates into the images
    # requires the gps receiver to be turned on in more recent versions of the blkfly script (double check this before using)
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
                lat += 0.00001 # just continuously adds to the lat and lon coordinate to ensure predictable variability
                lon += 0.00001
            except Exception as e:
                print(f"GPS Sender Error: {e}")
                
            time.sleep(1)
    finally:
        sock.close()
        print("GPS sender exiting cleanly")
    
# =================
# saver
# =================
def saver():
    # the counters can keep track of how many frames have been saved, this can be done either independently
    # (when the right camera saves only the right counter goes up) or this can be done dependently 
    # (when the right camera save all counters go up)
    # each camera saves the images to a different folder
    counters = {"left": 1, "mid": 1, "right": 1}

    paths = {
        "left": "road_test_2/left_cam",
        "mid": "road_test_2/mid_cam",
        "right": "road_test_2/right_cam",
    }
    # if the directories don't exist on your machine os.makedirs() will create the required directories
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    # once the directories exist this function will pull the decoded frames from each save_q
    # then pull the count from the corresponding label's counter and create a file path using the count as the frame ID
    # the function then writes the image into the file path and updates the counters (currently working dependent on one another)
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
                counters['left'] += 1
                counters['mid'] += 1
                counters['right'] += 1
            except Exception as e:
                print(f"saver error {label}: {e}")

        if not made_progress:
            time.sleep(0.02)

        
# =======================
# de-initilization
# =======================
def de_init():
    # will SSH into each camera's computer and pass the "pgrep -f" command
    # this identifies all PIDs where the desired script is running
    # then using the PIDs it will SSH into the computer again and pass the "kill -2" command twice to those PIDs
    # this acts like sending a CTRL+C twice
    # the first "kill -2" causes it to exit the aquisition loops and the second tells it to exit the main function and terminate
    
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
    
    # creates the tcp servers for each camera using different ports
    l_sock = create_tcp_server(5001)
    m_sock = create_tcp_server(5002)
    r_sock = create_tcp_server(5000)
    
    # creates a receiving thread for each camera using the created TCP servers
    l_rec_thread = threading.Thread(target=receiver, args=(l_sock, "left"), daemon=True)
    m_rec_thread = threading.Thread(target=receiver, args=(m_sock, "mid"), daemon=True)
    r_rec_thread = threading.Thread(target=receiver, args=(r_sock, "right"), daemon=True)
    
    # creates a decode thread for each camera
    l_dec_thread = threading.Thread(target=decoder, args=("left",), daemon=True)
    m_dec_thread = threading.Thread(target=decoder, args=("mid",), daemon=True)
    r_dec_thread = threading.Thread(target=decoder, args=("right",), daemon=True)
    
    # creates a save thread to handle all of three cameras at once
    save_thread = threading.Thread(target=saver, daemon = True)
    
    # creates the dummy gps thread
    gps_thread = threading.Thread(target=gps_dummy_sender, daemon=True)
    
    #creates a stats thread for each camera
    m_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT2, "mid"), daemon=True)
    l_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT1
                                                                , "left"), daemon=True)
    r_stats_thread = threading.Thread(target=stat_thread, args=(STAT_PORT3, "right"), daemon=True)
    
    # starts the receiving threads
    l_rec_thread.start()
    m_rec_thread.start()
    r_rec_thread.start()
    
    # starts the decoding threads
    l_dec_thread.start()
    m_dec_thread.start()
    r_dec_thread.start()
    
    # starts the stats threads
    m_stats_thread.start()
    l_stats_thread.start()
    r_stats_thread.start()

    # runs the initilization function to start the cameras
    init()
    
    # starts the save thread 
    save_thread.start()
    
    # starts the dummy gps thread
    gps_thread.start()
    global latest_left, latest_mid, latest_right, l_capt, l_send, l_enc, l_stream, l_save, l_exif, l_time, m_capt, m_send, m_enc, m_stream, m_save, m_exif, m_time, r_capt, r_enc, r_send, r_stream, r_save, r_exif, r_time, l_rec, m_rec, r_rec
    
    # initilizes pygame and sets the screen size
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Stream")
        
    clock = pygame.time.Clock()
    running = True
    
    # creates a window to display the live camera views and statistics while the script runs
    try:
        while running and not stop_event.is_set():
            # if the pygame window is shut down the script will terminate
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    stop_event.set()
            
            # using frame lock the left/mid/right bytes are taken directly from the latest_left/mid/right
            # left/mid/right_img is set to None so that if a camera stops sending the image will go black instead of freezing
            with frame_lock:
                left_img = None
                mid_img = None
                right_img = None
                left_bytes = latest_left
                mid_bytes = latest_mid
                right_bytes = latest_right
                
            # if left/mid/right_bytes are all None then the loop runs again after a short waiting time
            if left_bytes is None and mid_bytes is None and right_bytes is None:
                print('all cameras failing to send')
                clock.tick(2)
                continue
            
            # if any of left/mid/right_bytes is not None then the bytes are decoded into a grayscale .jpg similar to the save thread
            # optional print available to tell which camera is failing
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
            
            # initilizes the text to be displayed on screen"
            left_text = 'left camera'
            l_capture = f"capture: {l_capt} FPS"
            l_encode = f"encode: {l_enc} FPS"
            l_sent = f"send: {l_send} FPS"
            l_receive = f"receiced: {l_rec} FPS"
            l_streamq = f"stream_q: {l_stream} frames"
            l_saveq = f"save_q: {l_save} frames"
            #l_timeq = f"time_q: {l_time} timestamps"
            #l_exifq = f"exif_q: {l_exif} arrays"
            mid_text = 'mid camera'
            m_capture = f"capture: {m_capt} FPS"
            m_encode = f"encode: {m_enc} FPS"
            m_sent = f"send: {m_send} FPS"
            m_receive = f"received: {m_rec} FPS"
            m_streamq = f"stream_q: {m_stream} frames"
            m_saveq = f"save_q: {m_save} frames"
            #m_timeq = f"time_q: {m_time} timestamps"
            #m_exifq = f"exif_q: {m_exif} arrays"
            right_text = 'right camera'
            r_capture = f"capture: {r_capt} FPS"
            r_encode = f"encode: {r_enc} FPS"
            r_sent = f"send: {r_send} FPS"
            r_receive = f"received: {r_rec} FPS"
            r_streamq = f"stream_q: {r_stream} frames"
            r_saveq = f"save_q: {r_save} frames"
            #r_timeq = f"time_q: {r_time} timestamps"
            #r_exifq = f"exif_q: {r_exif} arrays"
            
            # sets the text color to white, background color to black, font to 24 pt, and fills the screen with the background color
            text_color = (255,255,255)
            bg_color = (0,0,0)
            font = pygame.font.SysFont(None, 24)
            screen.fill(bg_color)
            
            # each image that is available will have a surface created for it and screen.blit will paste the image
            # to the screen at specified coordinates. These coordinates correspond to the top left of the image
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
                
            # creates surfaces for each line of text to be displayed
            l_text_surface = font.render(left_text, True, text_color)
            lcs = font.render(l_capture, True, text_color)
            les = font.render(l_encode, True, text_color)
            lss = font.render(l_sent, True, text_color)
            lrs = font.render(l_receive, True, text_color)
            lsts = font.render(l_streamq, True, text_color)
            lsas = font.render(l_saveq, True, text_color)
            #lts = font.render(l_timeq, True, text_color)
            #lexs = font.render(l_exifq, True, text_color)
            m_text_surface = font.render(mid_text, True, text_color)
            mcs = font.render(m_capture, True, text_color)
            mes = font.render(m_encode, True, text_color)
            mss = font.render(m_sent, True, text_color)
            mrs = font.render(m_receive, True, text_color)
            msts = font.render(m_streamq, True, text_color)
            msas = font.render(m_saveq, True, text_color)
            #mts = font.render(m_timeq, True, text_color)
            #mexs = font.render(m_exifq, True, text_color)
            r_text_surface = font.render(right_text, True, text_color)
            rcs = font.render(r_capture, True, text_color)
            res = font.render(r_encode, True, text_color)
            rss = font.render(r_sent, True, text_color)
            rrs = font.render(r_receive, True, text_color)
            rsts = font.render(r_streamq, True, text_color)
            rsas = font.render(r_saveq, True, text_color)
            #rts = font.render(r_timeq, True, text_color)
            #rexs = font.render(r_exifq, True, text_color)
            
            # creates the rectangles for each text surface to be displayed within giving the coordinates of the center of each
            left_rect = l_text_surface.get_rect(center=(360,552))
            lcr = lcs.get_rect(center=(360,600))
            ler = les.get_rect(center=(360,650))
            lsr = lss.get_rect(center=(360,700))
            lrr = lrs.get_rect(center=(360,750))
            lstr = lsts.get_rect(center=(360,800))
            lsar = lsas.get_rect(center=(360,850))
            #ltr = lts.get_rect(center=(360,900))
            #lexr = lexs.get_rect(center=(360,950))
            mid_rect = m_text_surface.get_rect(center = (1080, 552))
            mcr = mcs.get_rect(center=(1080,600))
            mer = mes.get_rect(center=(1080,650))
            msr = mss.get_rect(center=(1080,700))
            mrr = mrs.get_rect(center=(1080,750))
            mstr = msts.get_rect(center=(1080,800))
            msar = msas.get_rect(center=(1080,850))
            #mtr = mts.get_rect(center=(1080,900))
            #mexr = mexs.get_rect(center=(1080,950))
            right_rect = r_text_surface.get_rect(center = (1800, 552))
            rcr = rcs.get_rect(center=(1800,600))
            rer = res.get_rect(center=(1800,650))
            rsr = rss.get_rect(center=(1800,700))
            rrr = rrs.get_rect(center=(1800,750))
            rstr = rsts.get_rect(center=(1800,800))
            rsar = rsas.get_rect(center=(1800,850))
            #rtr = rts.get_rect(center=(1800,900))
            #rexr = rexs.get_rect(center=(1800,950))
            
            # screen.blit will paste each text surface to the screen on their rectangle
            screen.blit(l_text_surface, left_rect)
            screen.blit(lcs,lcr)
            screen.blit(les,ler)
            screen.blit(lss,lsr)
            screen.blit(lrs, lrr)
            screen.blit(lsts, lstr)
            screen.blit(lsas,lsar)
            #screen.blit(lts, ltr)
            #screen.blit(lexs,lexr)
            screen.blit(m_text_surface, mid_rect)
            screen.blit(mcs,mcr)
            screen.blit(mes,mer)
            screen.blit(mss,msr)
            screen.blit(mrs, mrr)
            screen.blit(msts,mstr)
            screen.blit(msas,msar)
            #screen.blit(mts,mtr)
            #screen.blit(mexs,mexr)
            screen.blit(r_text_surface, right_rect)
            screen.blit(rcs,rcr)
            screen.blit(res,rer)
            screen.blit(rss,rsr)
            screen.blit(rrs, rrr)
            screen.blit(rsts,rstr)
            screen.blit(rsas, rsar)
            #screen.blit(rts,rtr)
            #screen.blit(rexs,rexr)
            pygame.display.flip()
            clock.tick(30)
            
    # if a CTRL+C command is sent the script will begin shutting down
    except KeyboardInterrupt:
        stop_event.set()
        
    # also if the display loop crashes the script will exit
    except Exception as e:
        stop_event.set()
        print(f"Display loop error: {e}")
    finally:
        # sets stop_event to cleanly exit the threads, closes the pygame window, closes the sockets,
        # uses de_init to shut down remote scripts, ensures that threads are exited then terminates
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
