# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:00:33 2024

@author: Desktop
"""

# coding=utf-8
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
from tkinter import Image
import PySpin
import sys
import time
import struct
#import piexif
#import usb.core
#import lcpy
import numpy as np
from datetime import datetime, timezone
from time import perf_counter_ns
import piexif
from PIL import Image as Image1
from exif import Image
import threading
import socket
import json
from fractions import Fraction
import queue
import signal
import io
import Jetson.GPIO as GPIO

# --- Camera identifier: set this for each camera script ---
CAMERA_ID = "mid"  # Change to "mid" or "right" for each camera

PORT_MAP = {
    "left": 5001,
    "mid": 5002,
    "right": 5000
    }

latest_gps = { "lat": None, "lon": None, "alt": None }
SAVE_QUEUE_SIZE = 10000
TIME_QUEUE_SIZE = 10000
STREAM_QUEUE_SIZE = 1000
STREAM_EVERY_N = 1  # 200 FPS acquisition -> 20 FPS stream
Exif_Queue_size = 1000
save_q = queue.Queue(maxsize=SAVE_QUEUE_SIZE)
stream_q = queue.Queue(maxsize=STREAM_QUEUE_SIZE)
exif_q = queue.Queue(maxsize=Exif_Queue_size)
time_q = queue.Queue(maxsize=TIME_QUEUE_SIZE)

# Event to signal threads to stop
stop_event = threading.Event()
TARGET_IP = '192.168.1.1'
TARGET_PORT = PORT_MAP[CAMERA_ID]   # same port as before, now used for TCP image streaming
stop_event = threading.Event()
event_time = 0.0


def drop_oldest_and_put(q, item, label):
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
            print(f"{label} queue full; dropped oldest item")
            return True
        except queue.Full:
            print(f"{label} queue still full; dropped newest item")
            return False


def verify_piexif_inputs(exif_bytes, image_bytes):
    """
    Checks if the data is in the proper format for piexif.insert.
    Returns: (bool, str) - (Success status, Error message)
    """
    if not isinstance(exif_bytes, bytes):
        return False, f"exif_bytes must be bytes, got {type(exif_bytes)}"
    if not isinstance(image_bytes, bytes):
        return False, f"image_bytes must be bytes, got {type(image_bytes)}"

    if not exif_bytes.startswith(b'Exif'):
        return False, "exif_bytes missing 'Exif' header"

    if not image_bytes.startswith(b'\xff\xd8'):
        return False, "image_bytes is not a valid JPEG (missing SOI marker)"

    return True, "Valid"


def connect_stream_socket():
    """
    Create and connect a TCP socket for image streaming.
    Uses the same TARGET_IP and TARGET_PORT previously used for UDP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((TARGET_IP, TARGET_PORT))
    print(f"[broadcast_thread] Connected to TCP stream target {TARGET_IP}:{TARGET_PORT}")
    return sock


def send_framed_payload(sock, payload):
    """
    Send one TCP-framed message:
      4-byte big-endian payload length
      followed by payload bytes
    """
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def broadcast_thread(stream_q, stop_event):
    """
    TCP image streaming thread.
    Sends one full image payload per frame, prefixed with a 4-byte length.
    """
    sock = None

    try:
        sock = connect_stream_socket()

        while not stop_event.is_set() or not stream_q.empty():
            try:
                stream_item = stream_q.get(timeout=0.5)
                print(f"[broadcast] got item type={type(stream_item)}")
            except queue.Empty:
                continue

            jpeg_bytes = stream_item
            exif_byte = None
            frame_id = None

            if isinstance(stream_item, tuple):
                if len(stream_item) == 3:
                    frame_id, jpeg_bytes, exif_byte = stream_item
                elif len(stream_item) == 2:
                    jpeg_bytes, exif_byte = stream_item

            final_packet = jpeg_bytes

            if exif_byte is not None:
                is_valid, msg = verify_piexif_inputs(exif_byte, jpeg_bytes)
                if is_valid:
                    try:
                        output = io.BytesIO()
                        piexif.insert(exif_byte, jpeg_bytes, output)
                        final_packet = output.getvalue()
                    except Exception as e:
                        print(f'Failed to insert EXIF: {e}')
                        final_packet = jpeg_bytes
                else:
                    print(f'skipping EXIF insertion: {msg}')

            try:
                print(f"[broadcast] sending frame {frame_id} bytes={len(final_packet)} to {TARGET_IP}:{TARGET_PORT}")
                send_framed_payload(sock, final_packet)
                print(f"[broadcast] sent frame {frame_id}")
            except OSError as e:
                print(f"TCP send failed: {e}")
                try:
                    if sock is not None:
                        sock.close()
                except Exception:
                    pass
                sock = None

                # retry connection
                while sock is None and not stop_event.is_set():
                    try:
                        print("[broadcast_thread] Attempting TCP reconnect...")
                        sock = connect_stream_socket()
                    except OSError as reconnect_err:
                        print(f"[broadcast_thread] Reconnect failed: {reconnect_err}")
                        time.sleep(1)

                # try resending the current frame once after reconnect
                if sock is not None:
                    try:
                        send_framed_payload(sock, final_packet)
                    except OSError as resend_err:
                        print(f"[broadcast_thread] Resend failed after reconnect: {resend_err}")
            finally:
                stream_q.task_done()

    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def gps_listener(udp_port=5005):
    print(f"[gps_listener] Thread started on port {udp_port}")
    global latest_gps
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(('', udp_port))
    while not stop_event.is_set():
        data, _ = sock.recvfrom(4096)
        try:
            gps_data = json.loads(data.decode())
            if CAMERA_ID in gps_data:
                latest_gps = gps_data[CAMERA_ID]
        except Exception as e:
            print("GPS packet error:", e)


def decimal_to_dms_precise(decimal):
    degrees = abs(int(decimal))
    minutes_float = abs((decimal - degrees) * 60)
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return (degrees, minutes, round(seconds, 6))


def get_tow_from_utc():
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    gps_seconds = (now - gps_epoch).total_seconds()
    tow = gps_seconds % (7 * 86400)  # Time of Week (TOW) in seconds
    return round(tow, 4)


def exif_bytes(make, model, comment, focal):
    """
    Build EXIF bytes compatible with piexif.insert().
    GPS data removed.
    """
    from fractions import Fraction

    def to_rational(value):
        f = Fraction(value).limit_denominator()
        return (f.numerator, f.denominator)

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: make,
            piexif.ImageIFD.Model: model,
        },
        "Exif": {
            piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + comment.encode(),
            piexif.ExifIFD.FocalLength: to_rational(float(focal)),
        },
        "GPS": {},
        "1st": {},
        "thumbnail": None
    }

    return piexif.dump(exif_dict)


def embed_tow_with_exif_module(image_path, tow_value, latest_gps):
    with open(image_path, 'rb') as img_file:
        img = Image(img_file)

    img.user_comment = f"T_from_pulse={tow_value:.4f}"
    img.make = "Flir"
    img.model = "Blackfly S"
    img.focal_length = "6"

    if latest_gps and latest_gps["lat"] is not None and latest_gps["lon"] is not None:
        lat_deg, lat_min, lat_sec = decimal_to_dms_precise(abs(latest_gps["lat"]))
        lon_deg, lon_min, lon_sec = decimal_to_dms_precise(abs(latest_gps["lon"]))

        img.gps_latitude = (lat_deg, lat_min, lat_sec)
        img.gps_latitude_ref = 'N'
        img.gps_longitude = (lon_deg, lon_min, lon_sec)
        img.gps_longitude_ref = 'W'
        img.gps_altitude = latest_gps["alt"]
        img.gps_altitude_ref = 0

    with open(image_path, 'wb') as new_file:
        new_file.write(img.get_file())

    return True


def capture_image(cam, frame_id, timeout=.5, save_path=None, return_array=True):
    """
    Once the camera engine has been activated, this function is used to extract
    one image from the buffering memory and save it into a numpy array.
    """
    image_result = cam.GetNextImage(int(timeout * 1000))

    try:
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus(), end="\r")
            return False, None, None

        if return_array:
            frame = image_result.GetNDArray()
            ts = time.perf_counter()
            packet = (frame_id, ts, frame)
            image_array = frame
        else:
            packet = None
            image_array = None

        return True, image_array, packet
    finally:
        image_result.Release()


def cam_configuration(nodemap,
                      s_node_map,
                      frameRate=200,
                      pgrExposureCompensation=0,
                      exposureTime=3000,
                      gain=0,
                      blackLevel=0,
                      bufferCount=30,
                      verbose=True):
    print('enter configuration')
    """
    Configure the camera.
    """
    print('\n=================== Config camera ==============================================\n')
    result = True

    ptrAcquisitionFrameRateEnable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsAvailable(ptrAcquisitionFrameRateEnable) and PySpin.IsWritable(ptrAcquisitionFrameRateEnable):
        ptrAcquisitionFrameRateEnable.SetValue(True)
        print('Set AcquisitionFrameRateEnable to True')
    else:
        print('Failed to enable AcquisitionFrameRateEnable')
        return False

    result &= setAcquisitionMode(nodemap, AcquisitionModeName='Continuous')

    if frameRate is not None:
        result &= setFrameRate(nodemap, frameRate=frameRate)

    if isinstance(exposureTime, str) and exposureTime.lower() == "auto":
        result &= enableExposureAuto(nodemap)
    elif exposureTime is not None:
        result &= setExposureTime(nodemap, exposureTime=exposureTime)

    if gain is not None:
        result &= setGain(nodemap, gain=gain)
    if blackLevel is not None:
        result &= setBlackLevel(nodemap, blackLevel=blackLevel)
    if bufferCount is not None:
        result &= setBufferCount(s_node_map, bufferCount=bufferCount)

    if verbose:
        print('\n=================== Camera status after configuration ==========================\n')
        print_camera_config(nodemap, s_node_map)
    print('exit configuration')
    return result


def image_proc_thread(save_q, time_q, savedir, stop_event):
    count = 0
    while not stop_event.is_set() or not save_q.empty() or not time_q.empty():
        try:
            payload = save_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], tuple):
                packet, frame_time = payload
            else:
                packet = payload
                frame_time = time_q.get(timeout=0.5)

            frame_id, ts, frame = packet
            frame_time = str(frame_time)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"[image_proc] encode ok={ok} frame_id={frame_id}")
            if not ok:
                print(f"JPEG encode failed for frame {frame_id}")
                continue

            final_packet = encoded.tobytes()
            print(f"[image_proc] encoded bytes={len(final_packet)} frame_id={frame_id}")
            exif_byte = exif_bytes('Flir', 'BlackFly', frame_time, 6)
            drop_oldest_and_put(stream_q, (frame_id, final_packet, exif_byte), "stream")
            print(f"[image_proc] pushed frame {frame_id} to stream_q")
            count += 1
        except queue.Empty:
            print("time queue empty while processing image; dropping frame")
        except Exception as e:
            print(f"image_proc_thread error: {e}")
        finally:
            save_q.task_done()


def acquire_images(cam,
                   acquisition_index,
                   num_images,
                   savedir,
                   triggerType,
                   save_q,
                   stream_q,
                   stop_event,
                   cam_list,
                   system,
                   frameRate=200,
                   exposureTime=300,
                   gain=0,
                   blackLevel=0,
                   bufferCount=300,
                   timeout=10,
                   verbose=True):
    print('acquiring images')
    """
    Acquire and save images from a device.
    """
    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()
    global latest_gps

    if verbose:
        print_device_info(nodemap_tldevice)
        print_camera_config(nodemap, s_node_map)
        print_trigger_config(nodemap, s_node_map)

    result = True

    result &= cam_configuration(nodemap=nodemap,
                                s_node_map=s_node_map,
                                frameRate=frameRate,
                                exposureTime=exposureTime,
                                gain=gain,
                                blackLevel=blackLevel,
                                bufferCount=bufferCount,
                                verbose=verbose)

    print('*** IMAGE ACQUISITION ***\n')

    result &= trigger_configuration(nodemap=nodemap,
                                    s_node_map=s_node_map,
                                    triggerType="off",
                                    verbose=verbose)

    if not cam.IsStreaming():
        cam.BeginAcquisition()

    count = 0
    try:
        while True:
            ret, frame, packet = capture_image(cam, count)
            tow = get_tow_from_utc()
            offset = tow - event_time
            if not ret or frame is None or packet is None:
                print("Capture failed")
                continue
            drop_oldest_and_put(save_q, (packet, offset), "save")
            print(f"[acquire] pushed frame {count} to save_q")

            count += 1

            #img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)

            #if cv2.waitKey(1) == ord('q'):
             #   break
    except queue.Full:
        print("processing queues are full; dropping frame")
    except PySpin.SpinnakerException as ex:
        print(f"Spinnaker error during acquisition: {ex}")
        result = False
    except Exception as ex:
        print(f"Unexpected acquisition error: {ex}")
        result = False
    except KeyboardInterrupt:
        pass

    result &= trigger_configuration(nodemap=nodemap,
                                    s_node_map=s_node_map,
                                    triggerType=triggerType,
                                    verbose=verbose)
    activate_trigger(nodemap)

    if triggerType == "software":
        print("=================Trigger is setting to software================")
        count = 0
        while count < num_images:
            try:
                start = perf_counter_ns()
                cam.TriggerSoftware.Execute()
                ret, image_array = capture_image(cam)
                end = perf_counter_ns()
                t = (end - start) / 1e9
                print('time spent: %2.3f s' % t)

            except PySpin.SpinnakerException as ex:
                print("Error %s" % ex)
                ret = False
                image_array = None
                pass

            if ret:
                filename = 'Acquisition-%04d.jpg' % count
                save_path = os.path.join(savedir, filename)
                print('Image saved at %s' % save_path)
                count += 1
            else:
                print('Capture failed')
                result = False

    if triggerType == "hardware":
        count = 0
        mtx = []
        start = perf_counter_ns()
        while count < num_images:
            try:
                ret, image_array = capture_image(cam)
                mtx.append(image_array)
                end_time = perf_counter_ns()
                print(end_time - start - 10000)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                ret = False
                image_array = None
                pass
            if ret:
                print("extract successfully")
                filename = 'Acquisition-%02d-%03d.jpg' % (acquisition_index, count)
                save_path = os.path.join(savedir, filename)
                print('Image saved at %s' % save_path)
                count += 1
                start = perf_counter_ns()
                print('waiting clock is reset')
            else:
                end = perf_counter_ns()
                waiting_time = (end - start) / 1e9
                print('Capture failed. Time spent %2.3f s before %2.3f s timeout' % (waiting_time, timeout))
                if waiting_time > timeout:
                    print('timeout is reached, stop capturing image ...')
                    break
        if count == 0:
            result = False

    cam.EndAcquisition()
    setExposureMode(nodemap, "Timed")
    deactivate_trigger(nodemap)

    return result


def print_device_info(nodemap_tldevice):
    print('\n*** DEVICE INFORMATION ***\n')
    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
        display_name_node_device_information = node_device_information.GetDisplayName()
        print(display_name_node_device_information)
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
            print('\n')
        else:
            print('Device control information not available.')
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return result


def run_single_camera(cam,
                      savedir,
                      acquisition_index,
                      num_images,
                      triggerType,
                      cam_list,
                      system,
                      frameRate=200,
                      exposureTime=6000,
                      gain=0,
                      bufferCount=30,
                      timeout=10):
    result = True
    try:
        cam.Init()
        result &= acquire_images(cam=cam,
                                 acquisition_index=acquisition_index,
                                 num_images=num_images,
                                 savedir=savedir,
                                 triggerType=triggerType,
                                 save_q=save_q,
                                 stream_q=stream_q,
                                 stop_event=stop_event,
                                 cam_list=cam_list,
                                 system=system,
                                 frameRate=frameRate,
                                 exposureTime=exposureTime,
                                 gain=gain,
                                 bufferCount=bufferCount,
                                 timeout=timeout)
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    except Exception as ex:
        print(f"Unhandled camera error: {ex}")
        result = False
    finally:
        try:
            if cam.IsStreaming():
                cam.EndAcquisition()
        except Exception:
            pass
        try:
            cam.DeInit()
        except Exception:
            pass
    return result


def sysScan():
    result = True

    system = PySpin.System.GetInstance()

    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    if not cam_list:
        result = False
        print('No camera is detected...')
    else:
        print('Number of cameras detected: %d' % num_cameras)

    return result, system, cam_list, num_cameras


def clearDir(targetDir):
    if len(os.listdir(targetDir)) != 0:
        for f in os.listdir(targetDir):
            os.remove(os.path.join(targetDir, f))
        print('Directory is cleared!')
    else:
        print('The target directory is empty! No image file needs to be removed')


def get_IEnumeration_node_current_entry_name(nodemap, nodename, verbose=True):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(nodename))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        print(f"Node {nodename} is not available or not readable.")
        return None
    node_int_val = node.GetIntValue()
    node_entry = node.GetEntry(node_int_val)
    node_entry_name = node_entry.GetSymbolic()
    if verbose:
        node_description = node.GetDescription()
        node_entries = node.GetEntries()
        print(f'{nodename}: {node_entry_name}')
        print(node_description)
        print('All entries are listed below:')
        for i, entry in enumerate(node_entries):
            entry_name = PySpin.CEnumEntryPtr(entry).GetSymbolic()
            print(f'{i}: {entry_name}')
        print('\n')
    return node_entry_name


def get_IInteger_node_current_val(nodemap, nodename, verbose=True):
    node = PySpin.CIntegerPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    if verbose:
        node_val_max = node.GetMax()
        node_val_min = node.GetMin()
        node_description = node.GetDescription()
        print('%s: %d' % (nodename, node_val))
        print(node_description)
        print('Max = %d' % node_val_max)
        print('Min = %d' % node_val_min)
        print('\n')
    return node_val


def get_IFloat_node_current_val(nodemap, nodename, verbose=True):
    node = PySpin.CFloatPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    if verbose:
        node_val_max = node.GetMax()
        node_val_min = node.GetMin()
        node_unit = node.GetUnit()
        print('%s: %f' % (nodename, node_val))
        print('Max = %f' % node_val_max)
        print('Min = %f' % node_val_min)
        print('Unit: ', node_unit)
        print('\n')
    return node_val


def get_IString_node_current_str(nodemap, nodename, verbose=True):
    node = PySpin.CStringPtr(nodemap.GetNode(nodename))
    node_str = node.GetValue()
    if verbose:
        node_description = node.GetDescription()
        print('%s: %s' % (nodename, node_str))
        print(node_description, '\n')
    return node_str


def get_IBoolean_node_current_val(nodemap, nodename, verbose=True):
    node = PySpin.CBooleanPtr(nodemap.GetNode(nodename))
    node_val = node.GetValue()
    if verbose:
        node_description = node.GetDescription()
        print('%s: %s' % (nodename, node_val))
        print(node_description, '\n')
    return node_val


def enableFrameRateSetting(nodemap):
    if not PySpin.IsAvailable(nodemap.GetNode("AcquisitionFrameRateEnable")):
        print('AcquisitionFrameRateEnable node not available')
        return False

    ptrAcquisitionFrameRateEnable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if not PySpin.IsAvailable(ptrAcquisitionFrameRateEnable) or not PySpin.IsWritable(ptrAcquisitionFrameRateEnable):
        print('Unable to access AcquisitionFrameRateEnable node')
        return False

    ptrAcquisitionFrameRateEnable.SetValue(True)
    print('Set AcquisitionFrameRateEnable to True')

    if PySpin.IsAvailable(nodemap.GetNode("AcquisitionFrameRateAuto")):
        ptrAcquisitionFrameRateAuto = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionFrameRateAuto"))
        if PySpin.IsAvailable(ptrAcquisitionFrameRateAuto) and PySpin.IsWritable(ptrAcquisitionFrameRateAuto):
            entry_off = ptrAcquisitionFrameRateAuto.GetEntryByName("Off")
            if PySpin.IsAvailable(entry_off) and PySpin.IsReadable(entry_off):
                ptrAcquisitionFrameRateAuto.SetIntValue(entry_off.GetValue())
                print('Set AcquisitionFrameRateAuto to Off')

    return True


def setFrameRate(nodemap, frameRate):
    if not enableFrameRateSetting(nodemap):
        return False
    ptrAcquisitionFramerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if (not PySpin.IsAvailable(ptrAcquisitionFramerate)) or (not PySpin.IsWritable(ptrAcquisitionFramerate)):
        print('Unable to retrieve AcquisitionFrameRate. Aborting...')
        return False
    ptrAcquisitionFramerate.SetValue(frameRate)
    print('AcquisitionFrameRate set to %3.3f Hz' % frameRate)
    return True


def enableExposureAuto(nodemap):
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)):
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    ExposureAuto_on = ptrExposureAuto.GetEntryByName("Continuous")
    if (not PySpin.IsAvailable(ExposureAuto_on)) or (not PySpin.IsReadable(ExposureAuto_on)):
        print('Unable to set ExposureAuto mode to Continuous. Aborting...')
        return False
    ptrExposureAuto.SetIntValue(ExposureAuto_on.GetValue())
    print('ExposureAuto mode is set to "Continuous"')
    return True


def disableExposureAuto(nodemap):
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)):
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    ExposureAuto_off = ptrExposureAuto.GetEntryByName("Off")
    if (not PySpin.IsAvailable(ExposureAuto_off)) or (not PySpin.IsReadable(ExposureAuto_off)):
        print('Unable to set ExposureAuto mode to Off. Aborting...')
        return False
    ptrExposureAuto.SetIntValue(ExposureAuto_off.GetValue())
    print('ExposureAuto mode is set to "off"')
    return True


def disableExposureCompensationAuto(nodemap):
    ptrExposureCompensationAuto = PySpin.CEnumerationPtr(nodemap.GetNode("pgrExposureCompensationAuto"))
    if (not PySpin.IsAvailable(ptrExposureCompensationAuto)) or (not PySpin.IsWritable(ptrExposureCompensationAuto)):
        print('Unable to retrieve ExposureCompensationAuto. Aborting...')
        return False
    ExposureCompensationAuto_off = ptrExposureCompensationAuto.GetEntryByName("Off")
    if (not PySpin.IsAvailable(ExposureCompensationAuto_off)) or (not PySpin.IsReadable(ExposureCompensationAuto_off)):
        print('Unable to set ExposureCompensationAuto mode to Off. Aborting...')
        return False
    ptrExposureCompensationAuto.SetIntValue(ExposureCompensationAuto_off.GetValue())
    print('ExposureCompensationAuto mode is set to "off"')
    return True


def setExposureMode(nodemap, exposureModeToSet):
    ptrExposureMode = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    if (not PySpin.IsAvailable(ptrExposureMode)) or (not PySpin.IsWritable(ptrExposureMode)):
        print('Unable to retrieve ExposureMode. Aborting...')
        return False
    ExposureMode_selected = ptrExposureMode.GetEntryByName(exposureModeToSet)
    if (not PySpin.IsAvailable(ExposureMode_selected)) or (not PySpin.IsReadable(ExposureMode_selected)):
        print('Unable to set ExposureMode to %s. Aborting...' % exposureModeToSet)
        return False
    ptrExposureMode.SetIntValue(ExposureMode_selected.GetValue())
    print('ExposureMode is set to %s' % exposureModeToSet)
    return True


def setTriggerMode(nodemap, TriggerModeToSet):
    ptrTriggerMode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
    if (not PySpin.IsAvailable(ptrTriggerMode)) or (not PySpin.IsWritable(ptrTriggerMode)):
        print('Unable to retrieve TriggerMode. Aborting...')
        return False
    TriggerMode_selected = ptrTriggerMode.GetEntryByName(TriggerModeToSet)
    if (not PySpin.IsAvailable(TriggerMode_selected)) or (not PySpin.IsReadable(TriggerMode_selected)):
        print('Unable to set TriggerMode to %s. Aborting...' % TriggerModeToSet)
        return False
    ptrTriggerMode.SetIntValue(TriggerMode_selected.GetValue())
    print('TriggerMode is set to %s...' % TriggerModeToSet)
    return True


def setTriggerActivation(nodemap, TriggerActivationToSet):
    ptrTriggerActivation = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerActivation"))
    if (not PySpin.IsAvailable(ptrTriggerActivation)) or (not PySpin.IsWritable(ptrTriggerActivation)):
        print('Unable to retrieve TriggerActivation. Aborting...')
        return False
    TriggerActivation_selected = ptrTriggerActivation.GetEntryByName(TriggerActivationToSet)
    if (not PySpin.IsAvailable(TriggerActivation_selected)) or (not PySpin.IsReadable(TriggerActivation_selected)):
        print('Unable to set TriggerActivation to %s. Aborting...' % TriggerActivationToSet)
        return False
    ptrTriggerActivation.SetIntValue(TriggerActivation_selected.GetValue())
    print('TriggerActivation is set to %s...' % TriggerActivationToSet)
    return True


def setTriggerOverlap(nodemap, TriggerOverlapToSet):
    ptrTriggerOverlap = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerOverlap"))
    if (not PySpin.IsAvailable(ptrTriggerOverlap)) or (not PySpin.IsWritable(ptrTriggerOverlap)):
        print('Unable to retrieve TriggerOverlap. Aborting...')
        return False
    TriggerOverlap_selected = ptrTriggerOverlap.GetEntryByName(TriggerOverlapToSet)
    if (not PySpin.IsAvailable(TriggerOverlap_selected)) or (not PySpin.IsReadable(TriggerOverlap_selected)):
        print('Unable to set TriggerOverlap to %s. Aborting...' % TriggerOverlapToSet)
        return False
    ptrTriggerOverlap.SetIntValue(TriggerOverlap_selected.GetValue())
    print('TriggerOverlap is set to %s..' % TriggerOverlapToSet)
    return True


def setTriggerSelector(nodemap, TriggerSelectorToSet):
    ptrTriggerSelector = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSelector"))
    if (not PySpin.IsAvailable(ptrTriggerSelector)) or (not PySpin.IsWritable(ptrTriggerSelector)):
        print('Unable to retrieve TriggerSelector. Aborting...')
        return False
    TriggerSelector_selected = ptrTriggerSelector.GetEntryByName(TriggerSelectorToSet)
    if (not PySpin.IsAvailable(TriggerSelector_selected)) or (not PySpin.IsReadable(TriggerSelector_selected)):
        print('Unable to set TriggerSelector to %s. Aborting...' % TriggerSelectorToSet)
        return False
    ptrTriggerSelector.SetIntValue(TriggerSelector_selected.GetValue())
    print('TriggerSelector is set to %s...' % TriggerSelectorToSet)
    return True


def setTriggerSource(nodemap, TriggerSourceToSet):
    ptrTriggerSource = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
    if (not PySpin.IsAvailable(ptrTriggerSource)) or (not PySpin.IsWritable(ptrTriggerSource)):
        print('Unable to retrieve TriggerSource. Aborting...')
        return False
    TriggerSource_selected = ptrTriggerSource.GetEntryByName(TriggerSourceToSet)
    if (not PySpin.IsAvailable(TriggerSource_selected)) or (not PySpin.IsReadable(TriggerSource_selected)):
        print('Unable to set TriggerSource to %s. Aborting...' % TriggerSourceToSet)
        return False
    ptrTriggerSource.SetIntValue(TriggerSource_selected.GetValue())
    print('TriggerSource is set to %s...' % TriggerSourceToSet)
    return True


def setExposureTime(nodemap, exposureTime=None):
    if not setExposureMode(nodemap, "Timed"):
        return False
    if not disableExposureAuto(nodemap):
        return False
    ptrExposureTime = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if (not PySpin.IsAvailable(ptrExposureTime)) or (not PySpin.IsWritable(ptrExposureTime)):
        print('Unable to retrieve Exposure Time. Aborting...')
        return False
    exposureTimeMax = ptrExposureTime.GetMax()
    if exposureTime is None:
        exposureTime = exposureTimeMax
    else:
        if exposureTime > exposureTimeMax:
            exposureTime = exposureTimeMax
    ptrExposureTime.SetValue(exposureTime)
    print('Exposure Time set to %5.2f us' % exposureTime)
    return True


def setAcquisitionMode(nodemap, AcquisitionModeName):
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if (not PySpin.IsAvailable(node_acquisition_mode)) or (not PySpin.IsWritable(node_acquisition_mode)):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False
    node_acquisition_mode_selected = node_acquisition_mode.GetEntryByName(AcquisitionModeName)
    if (not PySpin.IsAvailable(node_acquisition_mode_selected)) or (not PySpin.IsReadable(node_acquisition_mode_selected)):
        print('Unable to set acquisition mode to %s. Aborting...' % node_acquisition_mode_selected)
        return False
    node_acquisition_mode.SetIntValue(node_acquisition_mode_selected.GetValue())
    print('Acquisition mode set to %s' % AcquisitionModeName)
    return True


def setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName):
    handlingMode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
    if (not PySpin.IsAvailable(handlingMode)) or (not PySpin.IsWritable(handlingMode)):
        print('Unable to set Buffer Handling mode (node retrieval). Aborting...')
        return False
    handlingModeSelected = handlingMode.GetEntryByName(StreamBufferHandlingModeName)
    if (not PySpin.IsAvailable(handlingModeSelected)) or (not PySpin.IsReadable(handlingModeSelected)):
        print('Unable to set Buffer Handling mode (Value retrieval). Aborting...')
        return False
    handlingMode.SetIntValue(handlingModeSelected.GetValue())
    print('Buffer Handling Mode set to %s...' % StreamBufferHandlingModeName)
    return True


def setBufferCount(s_node_map, bufferCount):
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
    if (not PySpin.IsAvailable(buffer_count)) or (not PySpin.IsWritable(buffer_count)):
        print('Unable to set Buffer Count (Integer node retrieval). Aborting...')
        return False
    buffer_count.SetValue(bufferCount)
    print('Buffer count now set to: %d' % buffer_count.GetValue())
    return True


def disableGainAuto(nodemap):
    gainAuto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    if (not PySpin.IsAvailable(gainAuto)) or (not PySpin.IsWritable(gainAuto)):
        print('Unable to retrieve GainAuto. Aborting...')
        return False
    gainAutoOff = gainAuto.GetEntryByName('Off')
    if (not PySpin.IsAvailable(gainAutoOff)) or (not PySpin.IsReadable(gainAutoOff)):
        print('Unable to set GainAuto to off (Value retrieval). Aborting...')
        return False
    gainAuto.SetIntValue(gainAutoOff.GetValue())
    print('Set GainAuto to off')
    return True


def setGain(nodemap, gain):
    if not disableGainAuto(nodemap):
        return False
    gainValue = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if (not PySpin.IsAvailable(gainValue)) or (not PySpin.IsWritable(gainValue)):
        print('Unable to retrieve Gain. Aborting...')
        return False
    gainValue.SetValue(gain)
    print('Set Gain to %2.3f' % gain)
    return True


def setBlackLevel(nodemap, blackLevel):
    blackLevelValue = PySpin.CFloatPtr(nodemap.GetNode("BlackLevel"))
    if (not PySpin.IsAvailable(blackLevelValue)) or (not PySpin.IsWritable(blackLevelValue)):
        print('Unable to retrieve BlackLevel. Aborting...')
        return False
    blackLevelValue.SetValue(blackLevel)
    print('Set BlackLevel to %2.3f' % blackLevel)
    return True


def setExposureCompensation(nodemap, pgrExposureCompensation):
    ExposureCompensationValue = PySpin.CFloatPtr(nodemap.GetNode("pgrExposureCompensation"))
    if (not PySpin.IsAvailable(ExposureCompensationValue)) or (not PySpin.IsWritable(ExposureCompensationValue)):
        print('Unable to retrieve pgrExposureCompensation. Aborting...')
        return False
    ExposureCompensationValue.SetValue(pgrExposureCompensation)
    print('Set pgrExposureCompensation to %2.3f' % pgrExposureCompensation)
    return True


def trigger_configuration(nodemap, s_node_map, triggerType, verbose=True):
    print('\n*** CONFIGURING TRIGGER ***\n')
    if triggerType == 'software':
        print('Software trigger is chosen...')
    elif triggerType == 'hardware':
        print('Hardware trigger is chosen...')
    elif triggerType == 'off':
        print('Disable trigger mode for live view...')

    try:
        result = True
        result &= setTriggerMode(nodemap, "Off")

        if triggerType == 'off':
            result &= setExposureMode(nodemap, "Timed")
            result &= setTriggerSelector(nodemap, "FrameStart")
            result &= setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='NewestOnly')
            print("----------------------------------------------------------trigger type OFF return: ", result)

        if triggerType == 'software':
            result &= setTriggerSource(nodemap, "Software")
            result &= setExposureMode(nodemap, "Timed")
            result &= setTriggerSelector(nodemap, "FrameStart")
            result &= setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='OldestFirst')
            print("----------------------------------------------------------trigger type software return: ", result)

        if triggerType == 'hardware':
            result &= setTriggerSource(nodemap, "Line0")
            result &= setExposureMode(nodemap, "TriggerWidth")
            result &= setTriggerSelector(nodemap, "FrameStart")
            result &= setTriggerActivation(nodemap, "FallingEdge")
            result &= setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName='NewestOnly')
            print("----------------------------------------------------------trigger type hardware return: ", result)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    if verbose:
        print('\n=================== Trigger status after configuration ==========================\n')
        print_trigger_config(nodemap, s_node_map, triggerType)

    return result


def activate_trigger(nodemap):
    result = setTriggerMode(nodemap, "On")
    return result


def deactivate_trigger(nodemap):
    result = setTriggerMode(nodemap, "Off")
    return result


def print_camera_config(nodemap, s_node_map):
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode')
    get_IBoolean_node_current_val(nodemap, 'AcquisitionFrameRateEnable')
    get_IFloat_node_current_val(nodemap, 'AcquisitionFrameRate')
    get_IFloat_node_current_val(nodemap, 'AutoExposureEVCompensation')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureAuto')
    get_IEnumeration_node_current_entry_name(nodemap, 'GainAuto')
    get_IFloat_node_current_val(nodemap, 'Gain')
    get_IFloat_node_current_val(nodemap, 'BlackLevel')
    get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferCountMode')
    get_IInteger_node_current_val(s_node_map, 'StreamBufferCountManual')


def print_trigger_config(nodemap, s_node_map, triggerType="software"):
    if triggerType == 'software':
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource')
        get_IEnumeration_node_current_entry_name(nodemap, 'ExposureMode')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
        get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
        get_IFloat_node_current_val(nodemap, 'TriggerDelay')

    if triggerType == 'hardware':
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSource')
        exp_mode = get_IEnumeration_node_current_entry_name(nodemap, 'ExposureMode')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerActivation')
        get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
        get_IFloat_node_current_val(nodemap, 'TriggerDelay')


def main():
    global system, x, event_time
    savedir = r"/home/ryan5/Alex/Blackfky/road_test_2"
    PPS_PIN = 7
    x = 1
    pps = 0

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PPS_PIN, GPIO.IN)

    print(f"--- LISTENING ON PIN {PPS_PIN} ---")
    print("Setup:")
    print(f"1. ESP32 GPIO 18 -> Buffer Input")
    print(f"2. Buffer Output  -> Jetson Pin {PPS_PIN}")
    print("3. VCC = 3.3V (Do not use 5V!)")
    print("----------------------------------\n")

    try:
        while pps < 1:
            edge = GPIO.wait_for_edge(PPS_PIN, GPIO.RISING, timeout=1000)
            if edge is None:
                continue

            event_time = get_tow_from_utc()
            print(f"[{event_time}] 🟢 PPS Trigger Received!")

            time.sleep(0.1)
            pps = 1

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        GPIO.cleanup()

    print("[main] Launching gps_listener thread...")
    print("[main] Threads before GPS start:", [t.name for t in threading.enumerate()])
    #gps_thread = threading.Thread(target=gps_listener, name="GPS-Listener", daemon=True)
    stream_thread = threading.Thread(target=broadcast_thread, args=(stream_q, stop_event), daemon=True)
    process_thread = threading.Thread(target=image_proc_thread, args=(save_q, time_q, savedir, stop_event), daemon=True)
    process_thread.start()
    #gps_thread.start()
    stream_thread.start()
    time.sleep(5)
    print("[main] Threads after GPS start: ", [t.name for t in threading.enumerate()])
    print(latest_gps)

    acquisition_index = 0
    num_images = 15
    triggerType = "Off"
    result, system, cam_list, num_camerasmeras = sysScan()

    if result:
        os.makedirs(savedir, exist_ok=True)
        clearDir(savedir)
        for i, cam in enumerate(cam_list):
            print('Running example for camera %d...' % i)
            result &= run_single_camera(cam=cam,
                                        savedir=savedir,
                                        acquisition_index=acquisition_index,
                                        num_images=num_images,
                                        triggerType=triggerType,
                                        cam_list=cam_list,
                                        system=system,
                                        frameRate=200,
                                        exposureTime="auto",
                                        gain=None,
                                        bufferCount=15,
                                        timeout=1000)
            print('Camera %d example complete...' % i)
            x = 2

        if cam_list:
            pass
        else:
            print('Camera list is empty! No camera is detected, please check camera connection.')
    else:
        pass

    print("----------------HERE---------------------")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()
        while not save_q.empty():
            save_q.get_nowait()
        print('emptied save q')
        while not stream_q.empty():
            stream_q.get_nowait()
        print('emptied stream q')
        print('joined gps thread')
        print('joined stream thread')
        print('joined process thread')

        cam.DeInit()

        del cam
        cam_list.Clear()
        print('shutdown camera')

        system.ReleaseInstance()
        print("Shutdown complete.")
        cv2.destroyAllWindows()

    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)


#%%

# acquisition_index = 0
# num_images = 15
# triggerType = "software"
# result, system, cam_list, num_camerasmeras = sysScan()

# if result:
#     savedir = r"D:\images\test"
#     clearDir(savedir)
#     for i, cam in enumerate(cam_list):
#         print('Running example for camera %d...' % i)

# cam.Init()

# nodemap = cam.GetNodeMap()
# nodemap_tldevice = cam.GetTLDeviceNodeMap()
# s_node_map = cam.GetTLStreamNodeMap()

# frameRate=10
# exposureTime=8333
# gain=0
# blackLevel=0
# bufferCount=15
# timeout=10
# verbose=True
