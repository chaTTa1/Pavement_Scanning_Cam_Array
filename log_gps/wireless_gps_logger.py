# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:32:06 2026

@author: nicho
"""
import socket
import threading
import time
import csv
from datetime import datetime
import keyboard
import os

#got it to work when connecting the wifi module to hotspot and then the laptop to the XBEE hotspot
#must make sure all settings are correct on the mosaic-H as well (check nmea outputs, streams, ntrip settings (NTR1 must be on))

#HOST = "192.168.4.1"   # NTRIP master IP on xbee hotspot
#HOST_1 = '192.168.0.3'        #NTRIP master IP on TP-Link_33D8
#HOST_2 = '192.168.0.4'
HOST_1 = '172.20.10.5'       #NTIRP master IP on Alex's Hotspot
HOST_2 = '172.20.10.6'
HOST_3 = '172.20.10.7'
PORT = 5000
stop_event = threading.Event()
logging_enabled = False
XB_1 = []
XB_2 = []
XB_3 = []

def keyboard_thread():
    global logging_enabled

    last_q = False

    print("Press 'q' to start/stop logging.")
    print("Press Ctrl+C to quit.")

    while not stop_event.is_set():
        if keyboard.is_pressed('q'):
            if not last_q:
                logging_enabled = not logging_enabled

                if logging_enabled:
                    print("\n========== LOGGING STARTED ==========\n")
                else:
                    print("\n========== LOGGING STOPPED ==========\n")

                last_q = True
        else:
            last_q = False

        time.sleep(0.05)

def nmea_to_decimal(coord, direction):
    if not coord:
        return None

    degrees = float(coord[:2])
    minutes = float(coord[2:])
    decimal = degrees + minutes / 60.0

    if direction in ['S', 'W']:
        decimal *= -1

    return decimal
def parse_gst(sentence):
    parts = sentence.split(",")
    if len(parts) < 8:
        return None
    Lat_error = parts[6]
    Lon_error = parts[7]
    Alt_error = parts[8].split("*")[0]
    return {
        "Lat_error": Lat_error,
        "Lon_error": Lon_error,
        "Alt_error": Alt_error}

def parse_gga(sentence):
    parts = sentence.split(",")

    if len(parts) < 10:
        return None

    utc_time = parts[1]
    lat = parts[2]
    lat_dir = parts[3]
    lon = parts[4]
    lon_dir = parts[5]
    fix_quality = parts[6]
    satellites = parts[7]
    altitude = parts[9]

    #if fix_quality == "0":
        #return None  # no fix

    return {
        "utc_time": utc_time,
        "latitude": nmea_to_decimal(lat, lat_dir),
        "longitude": nmea_to_decimal(lon, lon_dir),
        "altitude_m": float(altitude) if altitude else None,
        "fix_quality": int(fix_quality),
        "satellites": int(satellites) if satellites else 0
    }

def XBEE_1():
    global XB_1
    filename = "GPS_1.csv"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST_1, PORT))
        s.settimeout(1)
        s.sendall(b"Hello from laptop")
        buffer = ''
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "pc_timestamp",
                "utc_time",
                "latitude",
                "longitude",
                "altitude_m",
                "fix_quality",
                "satellites",
                "Lat_error",
                "Lon_error",
                "Alt_error"
            ])
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            latest_gga = {}
            latest_gst = {}
            count = 0
            while not stop_event.is_set():
                try:
                    data = s.recv(1024)
                    if not data:
                        break
                    buffer += data.decode("utf-8", errors="ignore")
                    count += 1
                    if count == 5:
                        print('recieved from XBEE_1: ',data.decode('utf-8'))
                        count = 0

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if line.startswith("$") and "GGA" in line:
                            parsed = parse_gga(line)
                        
                            if parsed:
                                parsed["pc_timestamp"] = datetime.utcnow().isoformat()
                                latest_gga = parsed
                        

                        
                        if line.startswith("$") and "GST" in line:
                            parsed2 = parse_gst(line)
                        
                            if parsed2:
                                latest_gst = parsed2
                        if logging_enabled:
                            row = {**latest_gga, **latest_gst}
                            writer.writerow(row)
                            csvfile.flush()


                except socket.timeout:
                    continue
                except Exception as e:
                    print("GPS_1 error:", e)
                    break
    s.close()
def XBEE_2():
    global XB_2
    filename = 'GPS_2.csv'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as b:
        b.connect((HOST_2, PORT))
        b.settimeout(1)
        b.sendall(b"Hello from laptop")
        buffer  = ''
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "pc_timestamp",
                "utc_time",
                "latitude",
                "longitude",
                "altitude_m",
                "fix_quality",
                "satellites",
                "Lat_error",
                "Lon_error",
                "Alt_error"
            ])
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            latest_gga = {}
            latest_gst = {}
            count = 0

            while not stop_event.is_set():
                try:
                    data = b.recv(1024)
                    if not data:
                        break

                    buffer += data.decode("utf-8", errors="ignore")
                    #print('recieved from XBEE_2: ',data.decode('utf-8'))
                    count += 1
                    if count == 5:
                        print('recieved from XBEE_2: ',data.decode('utf-8'))
                        count = 0
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if line.startswith("$") and "GGA" in line:
                            parsed = parse_gga(line)
                        
                            if parsed:
                                parsed["pc_timestamp"] = datetime.utcnow().isoformat()
                                latest_gga = parsed
                        

                        
                        if line.startswith("$") and "GST" in line:
                            parsed2 = parse_gst(line)
                        
                            if parsed2:
                                latest_gst = parsed2
                        if logging_enabled:
                            row = {**latest_gga, **latest_gst}
                            writer.writerow(row)
                            csvfile.flush()


                except socket.timeout:
                    continue
                except Exception as e:
                    print("GPS_2 error:", e)
                    break
    b.close()
    
    
    
def XBEE_3():
    global XB_3
    filename = "GPS_3.csv"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST_3, PORT))
        s.settimeout(1)
        s.sendall(b"Hello from laptop")
        buffer = ''
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "pc_timestamp",
                "utc_time",
                "latitude",
                "longitude",
                "altitude_m",
                "fix_quality",
                "satellites",
                "Lat_error",
                "Lon_error",
                "Alt_error"
            ])
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writeheader()
            latest_gga = {}
            latest_gst = {}
            count = 0
            while not stop_event.is_set():
                try:
                    data = s.recv(1024)
                    if not data:
                        break
                    buffer += data.decode("utf-8", errors="ignore")
                    count += 1
                    if count == 5:
                        print('recieved from XBEE_3: ',data.decode('utf-8'))
                        count = 0

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if line.startswith("$") and "GGA" in line:
                            parsed = parse_gga(line)
                        
                            if parsed:
                                parsed["pc_timestamp"] = datetime.utcnow().isoformat()
                                latest_gga = parsed
                        

                        
                        if line.startswith("$") and "GST" in line:
                            parsed2 = parse_gst(line)
                        
                            if parsed2:
                                latest_gst = parsed2
                        if logging_enabled:
                            row = {**latest_gga, **latest_gst}
                            writer.writerow(row)
                            csvfile.flush()


                except socket.timeout:
                    continue
                except Exception as e:
                    print("GPS_3 error:", e)
                    break
        
        
def main():
    global XB_1, XB_2
    GPS_1 = threading.Thread(target = XBEE_1, daemon = True)
    keyboard_listener = threading.Thread(target=keyboard_thread, daemon=True)
    GPS_2 = threading.Thread(target = XBEE_2, daemon = True)
    GPS_3 = threading.Thread(target = XBEE_3, daemon = True)
    GPS_1.start()
    keyboard_listener.start()
    GPS_2.start()
    GPS_3.start
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping threads")
        stop_event.set()
    finally:
        GPS_1.join()
        keyboard_listener.join()
        GPS_2.join()

        
        
if __name__ == "__main__":
    main()

