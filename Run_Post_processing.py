import cam_array_post_processing

#IMU_GPS_post_processing.parse_mosaic_txt_data("C:\\Users\\alexc\\Documents\\Research\\GPS\\gpsjune24.sbf_Messages.txt")

cam_array_post_processing.propagate_gps_with_imu(
    r"C:\Users\alexc\Documents\Research\GPS\IMU_June24_test2_corrected.csv",
    r"C:\Users\alexc\Documents\Research\GPS\mosaic_parsed_output06_24_test2.csv"
)

