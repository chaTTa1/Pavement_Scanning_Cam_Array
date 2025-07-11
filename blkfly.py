# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:00:33 2024

@author: Desktop
"""


# coding=utf-8
import os
from tkinter import Image
import PySpin
import sys
import cv2
import piexif
import usb.core
import lcpy
import numpy as np
from datetime import datetime, timezone
from time import perf_counter_ns
import piexif
from PIL import Image
from exif import Image


def get_tow_from_utc():
    # gps epoch: 1980-01-06 00:00:00 UTC
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    gps_seconds = (now - gps_epoch).total_seconds()
    tow = gps_seconds % (7 * 86400) # Time of Week (TOW) in seconds
    return round(tow, 4)

def embed_tow_with_exif_module(image_path, tow_value):
    with open(image_path, 'rb') as img_file:
        img = Image(img_file)


    img.user_comment = f"TOW={tow_value:.4f}"
    img.make = "Flir"
    img.model = "Blackfly S"
    img.focal_length = "6"


    with open(image_path, 'wb') as new_file:
        new_file.write(img.get_file())

    
    return True

def capture_image(cam, timeout=10000, save_path=None, return_array=True):
    """
    Once the camera engine has been activated, this function is used to Extract 
    one image from the buffering memory and save it into a numpy array.
    Note that the camera mode must be set, e.g., "Continuous", and the 
    cam.BeginAcquisition() must be called before calling this function.
    
    Parameters
    ----------
    cam:camera object from PySpin
        camera object to be used
    timeout:int
        timeout for capturing a frame in milliseconds
    save_path:str
        save_path for saving jpeg file
    return_array:bool
        whether numpy array is saved
    Returns
    -------
    result:bool
        True--success
        False--failed
    image_averaged:uint8
        output image (numpy ndarray).

    """
    image_result = cam.GetNextImage(timeout)

    # Ensure image completion
    if image_result.IsIncomplete():
        print('Image incomplete with image status %d ...' % image_result.GetImageStatus(), end="\r")
        return False, None
    else:
        if save_path is not None:
            image_result.Save(save_path)
        if return_array:
            image_array = image_result.GetNDArray()
        else:
            image_array = None
        # Release image
        image_result.Release()
        return True, image_array


def cam_configuration(nodemap,
                      s_node_map,
                      frameRate=200,
                      pgrExposureCompensation=0,
                      exposureTime=3000,
                      gain=0,
                      blackLevel=0,
                      bufferCount=30,
                      verbose=True):
    """
    Configurate the camera. Note that the camera must be initialized before calling
    this function, i.e., cam.Init() must be called before calling this function.

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap
    s_node_map:INodemap
        camera stream nodemap
    frameRate:float
        Framerate, if given None, it will not be configured.
    pgrExposureCompensation:float
        Exposure compensation, if given None, it will not be configured.
    exposureTime:float
        Exposure time in microseconds, if given None, it will not be configured.
    gain:float
        Gain, if given None, it will not be configured.
    blackLevel:float
        black level, if given None, it will not be configured.
    bufferCount:int
        Buffer count, if given None, it will not be configured.
    verbose:bool
        If information should be printed out
    Returns
    -------
    result:bool
        result
    """

    print('\n=================== Config camera ==============================================\n')
    result = True
    ## find AcquisitionMode
    AcquisitionMode = get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode', verbose=False)
    if not (AcquisitionMode == 'Continuous'):
        result &= setAcquisitionMode(nodemap, AcquisitionModeName='Continuous')
    # TODO:
    ## find frame rate
    # if frameRate is not None:
    #     result &= setFrameRate(nodemap, frameRate=frameRate)
    # ExposureCompensationAuto = get_IEnumeration_node_current_entry_name(nodemap, 'pgrExposureCompensationAuto', verbose=False)
    # if not (ExposureCompensationAuto == 'Off'):
    #     result &= disableExposureCompensationAuto(nodemap)
    # if pgrExposureCompensation is not None:
    #     result &= setExposureCompensation(nodemap, pgrExposureCompensation=pgrExposureCompensation)
    ## find exposure mode
    if exposureTime is not None:
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
    return result


def acquire_images(cam,
                   acquisition_index,
                   num_images,
                   savedir,
                   triggerType,
                   frameRate=200,
                   exposureTime=3000,
                   gain=0,
                   blackLevel=0,
                   bufferCount=30,
                   timeout=10,
                   verbose=True):
    """
    This function acquires and saves images from a device. Note that camera 
    must be initialized and configured before calling this function, i.e.,
    cam.Init() and cam_configuration(cam,triggerType, ...,) must be called 
    before calling this function.

    :param cam: Camera to acquire images from.
    :param acquisition_index: the index number of the current acquisition.
    :param num_images: the total number of images to be taken.
    :param savedir: directory to save images.
    :param triggerType: Must be one of {"software", "hardware", "off"}.
                        If triggerType is "off", camera is configured for live view.
    :param frameRate: frame rate.
    :param pgrExposureCompensation: Exposure compensation
    :param exposureTime: exposure time in microseconds.
    :param gain: gain
    :param blackLevel: black level
    :param bufferCount: buffer count in RAM
    :param timeout: the maximum waiting time in seconds before termination.
    :param verbose: verbose print camera config status
    :type cam: CameraPtr
    :type acquisition_index: int
    :type num_images: int
    :type savedir: str
    :type triggerType: str
    :type frameRate: float
    :type pgrExposureCompensation: float
    :type exposureTime: int
    :type gain: float
    :type blackLevel: float
    :type bufferCount: int
    :type timeout: float
    :type verbose: bool
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    nodemap = cam.GetNodeMap()
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    s_node_map = cam.GetTLStreamNodeMap()
    
    if verbose:
        print_device_info(nodemap_tldevice)
        print_camera_config(nodemap, s_node_map)
        print_trigger_config(nodemap, s_node_map)    

    result = True
    # live view
    # config camera
    result &= cam_configuration(nodemap=nodemap,
                                s_node_map=s_node_map,
                                frameRate=frameRate,
                                exposureTime=exposureTime,
                                gain=gain,
                                blackLevel=blackLevel,
                                bufferCount=bufferCount,
                                verbose=verbose)

    print('*** IMAGE ACQUISITION ***\n')
    # config trigger for preview
    result &= trigger_configuration(nodemap=nodemap,
                                    s_node_map=s_node_map,
                                    triggerType="off",
                                    verbose=verbose)  # 'off' for preview

    cam.BeginAcquisition()
    count = 0
    try:
        while True:
            ret, frame = capture_image(cam)
            if not ret:
                print("Capture failed")
                continue

            # Save every frame (or use `if count % 5 == 0` to save every 5th frame)
            filename = f'frame_{count:06d}.jpg'
            save_path = os.path.join(savedir, filename)
            cv2.imwrite(save_path, frame)
            #write TOW to exif data
            tow = get_tow_from_utc()
            embed_tow_with_exif_module(save_path, tow)

            count += 1

            # Display preview
            img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.imshow("Press 'q' to stop", img_show)
            
            # Exit on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        cam.EndAcquisition()
        cv2.destroyAllWindows()
        print(f"Saved {count} images.")

    # Retrieve, convert, and save image
    # config trigger for image acquisition
    result &= trigger_configuration(nodemap=nodemap,
                                    s_node_map=s_node_map,
                                    triggerType=triggerType,
                                    verbose=verbose)
    activate_trigger(nodemap)
    cam.BeginAcquisition()

      

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
                print("Error %s"%ex)
                ret=False
                image_array = None
                pass
                
            if ret:
                filename = 'Acquisition-%04d.jpg' % count
                save_path = os.path.join(savedir, filename)
                cv2.imwrite(save_path, image_array)
                print('Image saved at %s' % save_path)
                count+=1
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
                cv2.imwrite(save_path, image_array)
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
    """
    This function prints the device information of the camera from the transport
    layer.

    :param nodemap_tldevice: Transport layer device nodemap.
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

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
                      frameRate=200,
                      exposureTime=3000,
                      gain=0,
                      bufferCount=30,
                      timeout=10):
    """
    Initialize and configurate a camera and take images. This is a wrapper
    function.

    :param cam: Camera to acquire images from.
    :param savedir: directory to save images.
    :param acquisition_index: the index number of the current acquisition.
    :param num_images: the total number of images to be taken.    
    :param triggerType: trigger type, must be one of {"software", "hardware"}
    :param frameRate: framerate.
    :param exposureTime: exposure time in microseconds.
    :param gain: gain
    :param bufferCount: buffer count number on RAM
    :param timeout: the waiting time in seconds before termination
    :type cam: CameraPtr
    :type savedir: str
    :type acquisition_index: int
    :type num_images: int    
    :type triggerType: str
    :type frameRate: float
    :type exposureTime: int
    :type gain: float
    :type bufferCount: int
    :type timeout: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Initialize camera
        cam.Init()

        # Acquire images        
        result &= acquire_images(cam=cam,
                                 acquisition_index=acquisition_index,
                                 num_images=num_images,
                                 savedir=savedir,
                                 triggerType=triggerType,
                                 frameRate=frameRate,
                                 exposureTime=exposureTime,
                                 gain=gain,
                                 bufferCount=bufferCount,
                                 timeout=timeout)
        # Deinitialize camera        
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result


def sysScan():
    """
    Scan the system and find all available cameras

    Returns
    -------
    result : bool
        Operation result, True or False.
    system : system object
        Camera system object
    cam_list : list
        Camera list.
    num_cameras : int
        Number of cameras.

    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    # Get the total number of cameras
    num_cameras = cam_list.GetSize()

    if not cam_list:
        result = False
        print('No camera is detected...')
    else:
        print('Number of cameras detected: %d' % num_cameras)

    return result, system, cam_list, num_cameras


def clearDir(targetDir):
    """
    Clear the directory
    
    Parameters
    ----------
    targetDir:str
        targetDir to be cleared.

    Returns
    -------
    None.

    """
    if len(os.listdir(targetDir)) != 0:
        for f in os.listdir(targetDir):
            os.remove(os.path.join(targetDir, f))
        print('Directory is cleared!')
    else:
        print('The target directory is empty! No image file needs to be removed')


# def get_IEnumeration_node_current_entry_name(nodemap, nodename, verbose=True):
#     node = PySpin.CEnumerationPtr(nodemap.GetNode(nodename))
#     node_int_val = node.GetIntValue()
#     node_entry = node.GetEntry(node_int_val)
#     node_entry_name = node_entry.GetSymbolic()
#     if verbose:
#         node_description = node.GetDescription()
#         node_entries = node.GetEntries()  # node_entries is a list of INode instances
#         print('%s: %s' % (nodename, node_entry_name))
#         print(node_description)
#         print('All entries are listed below:')
#         for i, entry in enumerate(node_entries):
#             entry_name = PySpin.CEnumEntryPtr(entry).GetSymbolic()
#             print('%d: %s' % (i, entry_name))
#         print('\n')
#     return node_entry_name

def get_IEnumeration_node_current_entry_name(nodemap, nodename, verbose=True):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(nodename))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        print(f"Node {nodename} is not available or not readable.")
        return None  # Or handle it as needed
    node_int_val = node.GetIntValue()
    node_entry = node.GetEntry(node_int_val)
    node_entry_name = node_entry.GetSymbolic()
    if verbose:
        node_description = node.GetDescription()
        node_entries = node.GetEntries()  # List of INode instances
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


# def disableOnBoardColorProcess(nodemap):
#     ptrOnBoardColorProcessEnabled = PySpin.CBooleanPtr(nodemap.GetNode("OnBoardColorProcessEnabled"))
#     if (not PySpin.IsAvailable(ptrOnBoardColorProcessEnabled)) or (not PySpin.IsWritable(ptrOnBoardColorProcessEnabled)):
#         print('Unable to retrieve OnBoardColorProcessEnabled. Aborting...')
#         return False
#     ptrOnBoardColorProcessEnabled.SetValue(False)
#     print('Set OnBoardColorProcessEnabled to False')
#     return True


def enableFrameRateSetting(nodemap):
    # Turn off "AcquisitionFrameRateAuto"    
    acqFrameRateAuto = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionFrameRateAuto"))
    
    if (not PySpin.IsAvailable(nodemap.GetNode("AcquisitionFrameRateEnable"))) or (not PySpin.IsWritable(nodemap.GetNode("AcquisitionFrameRateEnable"))):
        print('Unable to retrieve AcquisitionFrameRateAuto. Aborting...')
        return False
    acqFrameRateAutoOff = acqFrameRateAuto.GetEntryByName('True')
    if (not PySpin.IsAvailable(acqFrameRateAutoOff)) or (not PySpin.IsReadable(acqFrameRateAutoOff)):
        print('Unable to set Buffer Handling mode (Value retrieval). Aborting...')
        return False
    acqFrameRateAuto.SetIntValue(acqFrameRateAutoOff.GetValue())  # setting to Off
    print('Set AcquisitionFrameRateAuto to off')
    # Turn on "AcquisitionFrameRateEnabled"
    acqFrameRateEnabled = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnabled"))
    if (not PySpin.IsAvailable(acqFrameRateEnabled)) or (not PySpin.IsWritable(acqFrameRateEnabled)):
        print('Unable to retrieve AcquisitionFrameRateEnabled. Aborting...')
        return False
    acqFrameRateEnabled.SetValue(True)
    print('Set AcquisitionFrameRateEnabled to True')
    return True


def setFrameRate(nodemap, frameRate):
    """
    Set frame rate

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap
    frameRate:float
        frame rate

    Returns
    -------
    result:bool
        result
    """
    # First enable framerate setting    
    if not enableFrameRateSetting(nodemap):
        return False
    # frame rate should be a float number. Get the node and check availability   
    ptrAcquisitionFramerate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if (not PySpin.IsAvailable(ptrAcquisitionFramerate)) or (not PySpin.IsWritable(ptrAcquisitionFramerate)):
        print('Unable to retrieve AcquisitionFrameRate. Aborting...')
        return False
    # Set framerate value
    ptrAcquisitionFramerate.SetValue(frameRate)
    print('AcquisitionFrameRate set to %3.3f Hz' % frameRate)
    return True


def enableExposureAuto(nodemap):
    # Get the node "ExposureAuto" and convert it to Enumeration class
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)):
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    # Get the "Continuous" entry
    ExposureAuto_on = ptrExposureAuto.GetEntryByName("Continuous")
    if (not PySpin.IsAvailable(ExposureAuto_on)) or (not PySpin.IsReadable(ExposureAuto_on)):
        print('Unable to set ExposureAuto mode to Continuous. Aborting...')
        return False
    # set the "Continuous" entry to ExposureAuto
    ptrExposureAuto.SetIntValue(ExposureAuto_on.GetValue())
    print('ExposureAuto mode is set to "Continuous"')
    return True


def disableExposureAuto(nodemap):
    # Get the node "ExposureAuto" and convert it to Enumeration class
    ptrExposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if (not PySpin.IsAvailable(ptrExposureAuto)) or (not PySpin.IsWritable(ptrExposureAuto)):
        print('Unable to retrieve ExposureAuto. Aborting...')
        return False
    # Get the "Off" entry
    ExposureAuto_off = ptrExposureAuto.GetEntryByName("Off")
    if (not PySpin.IsAvailable(ExposureAuto_off)) or (not PySpin.IsReadable(ExposureAuto_off)):
        print('Unable to set ExposureAuto mode to Off. Aborting...')
        return False
    # set the "Off" entry to ExposureAuto
    ptrExposureAuto.SetIntValue(ExposureAuto_off.GetValue())
    print('ExposureAuto mode is set to "off"')
    return True


def disableExposureCompensationAuto(nodemap):
    # Get the node "ExposureCompensationAuto" and convert it to Enumeration class
    ptrExposureCompensationAuto = PySpin.CEnumerationPtr(nodemap.GetNode("pgrExposureCompensationAuto"))
    if (not PySpin.IsAvailable(ptrExposureCompensationAuto)) or (not PySpin.IsWritable(ptrExposureCompensationAuto)):
        print('Unable to retrieve ExposureCompensationAuto. Aborting...')
        return False
    # Get the "Off" entry
    ExposureCompensationAuto_off = ptrExposureCompensationAuto.GetEntryByName("Off")
    if (not PySpin.IsAvailable(ExposureCompensationAuto_off)) or (not PySpin.IsReadable(ExposureCompensationAuto_off)):
        print('Unable to set ExposureCompensationAuto mode to Off. Aborting...')
        return False
    # set the "Off" entry to ExposureAuto
    ptrExposureCompensationAuto.SetIntValue(ExposureCompensationAuto_off.GetValue())
    print('ExposureCompensationAuto mode is set to "off"')
    return True


def setExposureMode(nodemap, exposureModeToSet):
    """
    Sets the operation mode of the exposure (shutter). Toggles the Trigger
    Selector. Timed = Exposure Start; Trigger Width = Exposure Active

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    exposureModeToSet:str
        ExposureModeEnums, must be one of {"Timed", "TriggerWidth"}

    Returns
    -------
    result:bool
        result
    """
    # Get the node "ExposureMode" and check if it is available and writable
    ptrExposureMode = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureMode"))
    if (not PySpin.IsAvailable(ptrExposureMode)) or (not PySpin.IsWritable(ptrExposureMode)):
        print('Unable to retrieve ExposureMode. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    ExposureMode_selected = ptrExposureMode.GetEntryByName(exposureModeToSet)
    if (not PySpin.IsAvailable(ExposureMode_selected)) or (not PySpin.IsReadable(ExposureMode_selected)):
        print('Unable to set ExposureMode to %s. Aborting...' % exposureModeToSet)
        return False
        # Set the entry to the node
    ptrExposureMode.SetIntValue(ExposureMode_selected.GetValue())
    print('ExposureMode is set to %s' % exposureModeToSet)
    return True


def setTriggerMode(nodemap, TriggerModeToSet):
    """
    Controls whether the selected trigger is active

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    TriggerModeToSet:str
        TriggerModeEnums, must be one of {"Off", "On"}

    Returns
    -------
    result:bool
        result
    """
    # Get the node "TriggerMode" and check if it is available and writable
    ptrTriggerMode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
    if (not PySpin.IsAvailable(ptrTriggerMode)) or (not PySpin.IsWritable(ptrTriggerMode)):
        print('Unable to retrieve TriggerMode. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerMode_selected = ptrTriggerMode.GetEntryByName(TriggerModeToSet)
    if (not PySpin.IsAvailable(TriggerMode_selected)) or (not PySpin.IsReadable(TriggerMode_selected)):
        print('Unable to set TriggerMode to %s. Aborting...' % TriggerModeToSet)
        return False
    ptrTriggerMode.SetIntValue(TriggerMode_selected.GetValue())
    print('TriggerMode is set to %s...' % TriggerModeToSet)
    return True


def setTriggerActivation(nodemap, TriggerActivationToSet):
    """
    Specifies the activation mode of the trigger

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    TriggerActivationToSet:str
       TriggerActivationEnums, must be one of {"FallingEdge", "RisingEdge"}

    Returns
    -------
    result:bool
        result.
    """
    # Get the node "TriggerActivation" and check if it is available and writable
    ptrTriggerActivation = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerActivation"))
    if (not PySpin.IsAvailable(ptrTriggerActivation)) or (not PySpin.IsWritable(ptrTriggerActivation)):
        print('Unable to retrieve TriggerActivation. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerActivation_selected = ptrTriggerActivation.GetEntryByName(TriggerActivationToSet)
    if (not PySpin.IsAvailable(TriggerActivation_selected)) or (not PySpin.IsReadable(TriggerActivation_selected)):
        print('Unable to set TriggerActivation to %s. Aborting...' % TriggerActivationToSet)
        return False
    ptrTriggerActivation.SetIntValue(TriggerActivation_selected.GetValue())
    print('TriggerActivation is set to %s...' % TriggerActivationToSet)
    return True


def setTriggerOverlap(nodemap, TriggerOverlapToSet):
    """
    Overlapped Exposure Readout Trigger

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    TriggerOverlapToSet:str
        TriggerOverlapEnums, must be one of {"Off", "ReadOut"}

    Returns
    -------
    result:bool
        result.
    """
    # Get the node "TriggerOverlap" and check if it is available and writable
    ptrTriggerOverlap = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerOverlap"))
    if (not PySpin.IsAvailable(ptrTriggerOverlap)) or (not PySpin.IsWritable(ptrTriggerOverlap)):
        print('Unable to retrieve TriggerOverlap. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerOverlap_selected = ptrTriggerOverlap.GetEntryByName(TriggerOverlapToSet)
    if (not PySpin.IsAvailable(TriggerOverlap_selected)) or (not PySpin.IsReadable(TriggerOverlap_selected)):
        print('Unable to set TriggerOverlap to %s. Aborting...' % TriggerOverlapToSet)
        return False
    ptrTriggerOverlap.SetIntValue(TriggerOverlap_selected.GetValue())
    print('TriggerOverlap is set to %s..' % TriggerOverlapToSet)
    return True


def setTriggerSelector(nodemap, TriggerSelectorToSet):
    """
    Selects the type of trigger to configure. Derived from Exposure Mode.

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    TriggerSelectorToSet:str
        TriggerSelectorEnums, must be one of {"FrameStart", "AquisitionStart", "FrameStart"}.

    Returns
    -------
    result:bool
        result
    """
    # Get the node "TriggerOverlap" and check if it is available and writable
    ptrTriggerSelector = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSelector"))
    if (not PySpin.IsAvailable(ptrTriggerSelector)) or (not PySpin.IsWritable(ptrTriggerSelector)):
        print('Unable to retrieve TriggerSelector. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerSelector_selected = ptrTriggerSelector.GetEntryByName(TriggerSelectorToSet)
    if (not PySpin.IsAvailable(TriggerSelector_selected)) or (not PySpin.IsReadable(TriggerSelector_selected)):
        print('Unable to set TriggerSelector to %s. Aborting...' % TriggerSelectorToSet)
        return False
    ptrTriggerSelector.SetIntValue(TriggerSelector_selected.GetValue())
    print('TriggerSelector is set to %s...' % TriggerSelectorToSet)
    return True


def setTriggerSource(nodemap, TriggerSourceToSet):
    """
    Specifies the internal signal or physical input line to use as the trigger source.

    Parameters
    ----------
    nodemap:INodeMap
        Camara nodemap.
    TriggerSourceToSet:str
        TriggerSourceEnums, must be one of {"Software", "Line0", "Line1", "Line2", "Line3"}.

    Returns
    -------
    result:bool
        result
    """
    # Get the node "TriggerSource" and check if it is available and writable
    ptrTriggerSource = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
    if (not PySpin.IsAvailable(ptrTriggerSource)) or (not PySpin.IsWritable(ptrTriggerSource)):
        print('Unable to retrieve TriggerSource. Aborting...')
        return False
    # Get the Entry to be set and check if it is available and writable
    TriggerSource_selected = ptrTriggerSource.GetEntryByName(TriggerSourceToSet)
    if (not PySpin.IsAvailable(TriggerSource_selected)) or (not PySpin.IsReadable(TriggerSource_selected)):
        print('Unable to set TriggerSource to %s. Aborting...' % TriggerSourceToSet)
        return False
    ptrTriggerSource.SetIntValue(TriggerSource_selected.GetValue())
    print('TriggerSource is set to %s...' % TriggerSourceToSet)
    return True


def setExposureTime(nodemap, exposureTime=None):
    """
    Set exposure time in microseconds.

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap
    exposureTime:float
        exposure time in microseconds, if None is given, the maximum possible exposure time is used.

    Returns
    -------
    result:bool
        result
    """
    # First set the exposure mode to "timed"
    if not setExposureMode(nodemap, "Timed"):  ## either Timed or TriggerWidth
        return False
    # Second disable the ExposureAuto
    if not disableExposureAuto(nodemap):
        return False
        # Get the node "ExposureTime" and check if it is available and writable
    ptrExposureTime = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if (not PySpin.IsAvailable(ptrExposureTime)) or (not PySpin.IsWritable(ptrExposureTime)):
        print('Unable to retrieve Exposure Time. Aborting...')
        return False
    # Ensure desired exposure time does not exceed the maximum
    exposureTimeMax = ptrExposureTime.GetMax()
    if exposureTime is None:
        exposureTime = exposureTimeMax
    else:
        if exposureTime > exposureTimeMax:
            exposureTime = exposureTimeMax
    # Set the exposure time
    ptrExposureTime.SetValue(exposureTime)
    print('Exposure Time set to %5.2f us' % exposureTime)
    return True


def setAcquisitionMode(nodemap, AcquisitionModeName):
    """
    Explicitly set AcquisitionMode

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap.
    AcquisitionModeName:str
        must be one from the three: Continuous, SingleFrame, MultiFrame.

    Returns
    -------
    result:bool
        result
    """
    #  Retrieve enumeration node from nodemap

    # # In order to access the node entries, they have to be cast to a pointer type (CEnumerationPtr here)
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if (not PySpin.IsAvailable(node_acquisition_mode)) or (not PySpin.IsWritable(node_acquisition_mode)):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False
    # Retrieve entry node from enumeration node
    node_acquisition_mode_selected = node_acquisition_mode.GetEntryByName(AcquisitionModeName)
    if (not PySpin.IsAvailable(node_acquisition_mode_selected)) or (not PySpin.IsReadable(node_acquisition_mode_selected)):
        print('Unable to set acquisition mode to %s. Aborting...' % node_acquisition_mode_selected)
        return False
    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(node_acquisition_mode_selected.GetValue())
    print('Acquisition mode set to %s' % AcquisitionModeName)
    return True


def setStreamBufferHandlingMode(s_node_map, StreamBufferHandlingModeName):
    """
    Explicitly set StreamBufferHandlingModeName

    Parameters
    ----------
    s_node_map:INodemap
        steam nodemap.
    StreamBufferHandlingModeName:str
        must be one from the four: OldestFirst, OldestFirstOverwrite, NewestOnly, NewestFirst.

    Returns
    -------
    result:bool
        result
    """
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
    """
    Set manual buffer count

    Parameters
    ----------
    s_node_map:INodemap
        stream nodemap
    bufferCount:int
        buffer count number

    Returns
    -------
    result:bool
        result
    """
    # Retrieve and modify Stream Buffer Count
    buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
    if (not PySpin.IsAvailable(buffer_count)) or (not PySpin.IsWritable(buffer_count)):
        print('Unable to set Buffer Count (Integer node retrieval). Aborting...')
        return False
    buffer_count.SetValue(bufferCount)
    print('Buffer count now set to: %d' % buffer_count.GetValue())
    return True


def disableGainAuto(nodemap):
    """
    Disable Gain Auto

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap

    Returns
    -------
    result:bool
        result
    """
    gainAuto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    if (not PySpin.IsAvailable(gainAuto)) or (not PySpin.IsWritable(gainAuto)):
        print('Unable to retrieve GainAuto. Aborting...')
        return False
    gainAutoOff = gainAuto.GetEntryByName('Off')
    if (not PySpin.IsAvailable(gainAutoOff)) or (not PySpin.IsReadable(gainAutoOff)):
        print('Unable to set GainAuto to off (Value retrieval). Aborting...')
        return False
    # setting "Off" for the Gain auto
    gainAuto.SetIntValue(gainAutoOff.GetValue())  # setting to Off
    print('Set GainAuto to off')
    return True


def setGain(nodemap, gain):
    """
    Set camera gain value.

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap
    gain:float
        gain

    Returns
    -------
    result:bool
        result
    """
    # First disable gainAuto
    if not disableGainAuto(nodemap):
        return False
    # Get the node "Gain" and check the availability
    gainValue = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if (not PySpin.IsAvailable(gainValue)) or (not PySpin.IsWritable(gainValue)):
        print('Unable to retrieve Gain. Aborting...')
        return False
    # Set the gain value
    gainValue.SetValue(gain)
    print('Set Gain to %2.3f' % gain)
    return True


def setBlackLevel(nodemap, blackLevel):
    """
    Set the camera black level.

    Parameters
    ----------
    nodemap:INodemap
        Camera nodemap
    blackLevel:float
        black level

    Returns
    -------
    result:bool
        result
    """
    # Get the node "BlackLevel" and check the availability
    blackLevelValue = PySpin.CFloatPtr(nodemap.GetNode("BlackLevel"))
    if (not PySpin.IsAvailable(blackLevelValue)) or (not PySpin.IsWritable(blackLevelValue)):
        print('Unable to retrieve BlackLevel. Aborting...')
        return False
    # Set the BlackLevel value
    blackLevelValue.SetValue(blackLevel)
    print('Set BlackLevel to %2.3f' % blackLevel)
    return True


def setExposureCompensation(nodemap, pgrExposureCompensation):
    """
    Set the exposure compensation.

    Parameters
    ----------
    nodemap:INodemap
        camera nodemap
    pgrExposureCompensation:float
        exposure compensation

    Returns
    -------
    result:bool
        operation result
    """
    # Get the node "pgrExposureCompensation" and check the availability
    ExposureCompensationValue = PySpin.CFloatPtr(nodemap.GetNode("pgrExposureCompensation"))
    if (not PySpin.IsAvailable(ExposureCompensationValue)) or (not PySpin.IsWritable(ExposureCompensationValue)):
        print('Unable to retrieve pgrExposureCompensation. Aborting...')
        return False
    # Set the pgrExposureCompensation value
    ExposureCompensationValue.SetValue(pgrExposureCompensation)
    print('Set pgrExposureCompensation to %2.3f' % pgrExposureCompensation)
    return True


def trigger_configuration(nodemap, s_node_map, triggerType, verbose=True):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    ensured to be off in order to select the trigger source.

     :param nodemap: camera nodemap.
     :type nodemap: CNodemapPtr
     :param s_node_map: camera stream nodemap.
     :type s_node_map: CNodemapPtr
     :param triggerType: Trigger type, 'software' or 'hardware' or 'off'. If triggerType is "off", 
                         camera is configured for live view.
     :type triggerType: str
     :param verbose: verbose print out
     :type verbose: bool
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    print('\n*** CONFIGURING TRIGGER ***\n')
    if triggerType == 'software':
        print('Software trigger is chosen...')
    elif triggerType == 'hardware':
        print('Hardware trigger is chosen...')
    elif triggerType == 'off':
        print('Disable trigger mode for live view...')

    try:
        result = True

        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
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
    # setTriggerOverlap(nodemap, "ReadOut")    
    return result


def deactivate_trigger(nodemap):
    result = setTriggerMode(nodemap, "Off")
    return result


def print_camera_config(nodemap, s_node_map):
    
    """
    To select using which function to find the correct node name, the type of 
    node need to be verified. 3 of types in here:
        1. IEnumeration_node
        2. IFloat_node
        3. IBoolean_node
        
    For some new node that not sure which type it is, in SpinView GUI, onder feature
    section, right click node and select Display Node Information. On the pop-up window,
    there is a section called type.
    """
    get_IEnumeration_node_current_entry_name(nodemap, 'AcquisitionMode')
    get_IBoolean_node_current_val(nodemap, 'AcquisitionFrameRateEnable')
    get_IFloat_node_current_val(nodemap, 'AcquisitionFrameRate')
    get_IFloat_node_current_val(nodemap, 'AutoExposureEVCompensation')
    get_IEnumeration_node_current_entry_name(nodemap, 'ExposureAuto')
    # get_IFloat_node_current_val(nodemap, 'ExposureTime')
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

        # if exp_mode != "TriggerWidth":
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerMode')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerSelector')
        get_IEnumeration_node_current_entry_name(nodemap, 'TriggerActivation')
        get_IEnumeration_node_current_entry_name(s_node_map, 'StreamBufferHandlingMode')
        get_IFloat_node_current_val(nodemap, 'TriggerDelay')

def main():
    acquisition_index = 0
    num_images = 15
    triggerType = "Off"
    result, system, cam_list, num_camerasmeras = sysScan()

    if result:
        # Run example on each camera
        savedir = r"C:\Users\alexc\Documents\Research\Summer25_Img\TimestampTest2"
        clearDir(savedir)
        for i, cam in enumerate(cam_list):
            print('Running example for camera %d...' % i)
            result &= run_single_camera(cam=cam,
                                        savedir=savedir,
                                        acquisition_index=acquisition_index,
                                        num_images=num_images,
                                        triggerType=triggerType,
                                        exposureTime=50)
            print('Camera %d example complete...' % i)

        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        if cam_list:
            del cam
        else:
            print('Camera list is empty! No camera is detected, please check camera connection.')
    else:
        pass
    # Clear camera list before releasing system
    cam_list.Clear()
   
    # Release system instance
    system.ReleaseInstance()
    print("----------------HERE---------------------")

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
#     # Run example on each camera
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



