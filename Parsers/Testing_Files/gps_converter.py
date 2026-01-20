# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:31:02 2025

@author: nicho

this code will convert one gps coordinate and heading into the gps coordinates for all cameras in the system
"""

import numpy as np
import pandas as pd

CAMISLEFT_FLAG = False #change if the camera is to the right of gps point

gps_values = pd.read_csv(r"C:\Users\nicho\Downloads\parsed_gps_data.csv")
new_gps_values = []
updated_values = 0

""" distance between the gps module and the cameras """
l_dist = .6
r_dist = .4
m_dist = .1




def new_lat_west(lat,lon,alt,head, CAMISLEFT_FLAG):
    left_lat = l_dist
    right_lat = r_dist
    mid_lat = m_dist
    head = np.radians(head)
    left_lat = l_dist*np.sin(head)
    left_lat = left_lat / 111139
    left_lat = lat - left_lat
    right_lat = r_dist*np.sin(head)
    right_lat = right_lat / 111139
    right_lat = lat + right_lat
    if CAMISLEFT_FLAG:
        mid_lat = m_dist*np.sin(head)
        mid_lat = mid_lat / 111139
        mid_lat = lat - mid_lat
    else:
        mid_lat = m_dist*np.sin(head)
        mid_lat = mid_lat / 111139
        mid_lat = lat - mid_lat
    return left_lat, right_lat, mid_lat
def new_lat_east(lat,lon,alt,head, CAMISLEFT_FLAG):
    left_lat = l_dist
    right_lat = r_dist
    mid_lat = m_dist
    head = np.radians(head)
    left_lat = l_dist*np.sin(head)
    left_lat = left_lat / 111139
    left_lat = lat + left_lat
    right_lat = r_dist*np.sin(head)
    right_lat = right_lat / 111139
    right_lat = lat - right_lat
    if CAMISLEFT_FLAG:
        mid_lat = m_dist*np.sin(head)
        mid_lat = mid_lat / 111139
        mid_lat = lat - mid_lat
    else:
        mid_lat = m_dist*np.sin(head)
        mid_lat = mid_lat / 111139
        mid_lat = lat + mid_lat
    return left_lat, right_lat, mid_lat
def new_lon_north(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG):
    head = np.radians(head)
    if head > np.pi:
        head = head - np.pi
    left_lon = l_dist*np.cos(head)
    left_lon = left_lon / (111139*np.cos(left_lat))
    left_lon = lon - left_lon
    right_lon = r_dist*np.cos(head)
    right_lon = right_lon / (111139*np.cos(right_lat))
    right_lon = lon + right_lon
    if CAMISLEFT_FLAG:
        mid_lon = m_dist*np.cos(head)
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon - mid_lon
    else:
        mid_lon = m_dist*np.cos(head)
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon + mid_lon
    return left_lon, right_lon, mid_lon
def new_lon_south(lat,lon,alt,head,left_lat,right_lat,mid_lat, CAMISLEFT_FLAG):
    head = np.radians(head)
    if head > np.pi:
        head = head - np.pi
    left_lon = l_dist*np.cos(head)
    left_lon = left_lon / (111139*np.cos(left_lat))
    left_lon = lon - left_lon
    right_lon = r_dist*np.cos(head)
    right_lon = right_lon / (111139*np.cos(right_lat))
    right_lon = lon + right_lon
    if CAMISLEFT_FLAG:
        mid_lon = m_dist*np.cos(head)
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon - mid_lon
    else:
        mid_lon = m_dist*np.cos(head)
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon + mid_lon
    return left_lon, right_lon, mid_lon
def new_lon_0(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG):
    left_lon = l_dist
    left_lon = left_lon / (111139*np.cos(left_lat))
    left_lon = lon - left_lon
    right_lon = r_dist
    right_lon = right_lon / (111139*np.cos(right_lat))
    right_lon = lon + right_lon
    if CAMISLEFT_FLAG:
        mid_lon = m_dist
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon - mid_lon
    else:
        mid_lon = m_dist
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon + mid_lon
    return left_lon, right_lon, mid_lon
def new_lon_180(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG):
    left_lon = l_dist
    left_lon = left_lon / (111139*np.cos(left_lat))
    left_lon = lon + left_lon
    right_lon = r_dist
    right_lon = right_lon / (111139*np.cos(right_lat))
    right_lon = lon - right_lon
    if CAMISLEFT_FLAG:
        mid_lon = m_dist
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon + mid_lon
    else:
        mid_lon = m_dist
        mid_lon = mid_lon / (111139*np.cos(mid_lat))
        mid_lon = lon - mid_lon
    return left_lon, right_lon, mid_lon
def new_lat_90(lat,lon,alt,head, CAMISLEFT_FLAG):
    left_lat = l_dist
    left_lat = left_lat / 111139
    left_lat = lat + left_lat
    right_lat = r_dist
    right_lat = right_lat / 111139
    right_lat = lat - right_lat
    if CAMISLEFT_FLAG:
        mid_lat = m_dist
        mid_lat = mid_lat / 111139
        mid_lat = lat + mid_lat
    else:
        mid_lat = m_dist
        mid_lat = mid_lat / 111139
        mid_lat = lat - mid_lat
    return left_lat, right_lat, mid_lat
def new_lat_270(lat,lon,alt,head, CAMISLEFT_FLAG):
    left_lat = l_dist
    left_lat = left_lat / 111139
    left_lat = lat - left_lat
    right_lat = r_dist
    right_lat = right_lat / 111139
    right_lat = lat + right_lat
    if CAMISLEFT_FLAG:
        mid_lat = m_dist
        mid_lat = mid_lat / 111139
        mid_lat = lat - mid_lat
    else:
        mid_lat = m_dist
        mid_lat = mid_lat / 111139
        mid_lat = lat + mid_lat
    return left_lat, right_lat, mid_lat






def conversion(lat, lon, alt, head, TOW, WNc, CAMISLEFT_FLAG):
    lat = lat
    lon = lon
    alt = alt
    head = head
    TOW = TOW
    WNc = WNc
    if head > 0 and head < 180 and head != 90:
        left_lat, right_lat, mid_lat = new_lat_east(lat,lon,alt,head, CAMISLEFT_FLAG)
    elif head > 180 and head <360 and head != 270:
        left_lat, right_lat, mid_lat = new_lat_west(lat,lon,alt,head, CAMISLEFT_FLAG)
    if head > 270 or head < 90 and head != 0 and head != 360:
        left_lon, right_lon, mid_lon = new_lon_north(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG)
    elif head > 90 and head < 270 and head != 180:
        left_lon, right_lon, mid_lon = new_lon_south(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG)
    elif head == 0:
        left_lat = lat
        right_lat = lat
        mid_lat = lat
        left_lon, right_lon, mid_lon = new_lon_0(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG)
    elif head == 180:
        left_lat = lat
        right_lat = lat
        mid_lat = lat
        left_lon, right_lon, mid_lon = new_lon_180(lat,lon,alt,head, left_lat, right_lat, mid_lat, CAMISLEFT_FLAG)

    elif head == 90:
        left_lon = lon
        right_lon = lon
        mid_lon = lon
        left_lat, right_lat, mid_lat = new_lat_90(lat,lon,alt,head, CAMISLEFT_FLAG)

    elif head == 270:
        left_lon = lon
        right_lon = lon
        mid_lon = lon
        left_lat, right_lat, mid_lat = new_lat_270(lat,lon,alt,head, CAMISLEFT_FLAG)
    values = [left_lat, left_lon, mid_lat, mid_lon, right_lat, right_lon, alt, head, TOW, WNc]
    return values
    


for i in range(len(gps_values)):
    gps_value = gps_values.iloc[i]
    gps_lat = gps_value.iloc[2]
    gps_lon = gps_value.iloc[3]
    gps_alt = gps_value.iloc[4]
    gps_head = gps_value.iloc[5]
    gps_TOW = gps_value.iloc[0]
    gps_WNc = gps_value.iloc[1]
    new_gps_values.append(conversion(gps_lat, gps_lon, gps_alt, gps_head, gps_TOW, gps_WNc, CAMISLEFT_FLAG))
updated_gps_values = pd.DataFrame(new_gps_values, columns = ['left lat', 'left lon', 'Mid Lat', 'Mid lon', 'right lat', 'right lon', 'altitude', 'heading', 'TOW', 'WNc'])
updated_gps_values.to_csv(r'C:\Users\nicho\Downloads\gps_converted_values_flagtest(right).csv', index = False)