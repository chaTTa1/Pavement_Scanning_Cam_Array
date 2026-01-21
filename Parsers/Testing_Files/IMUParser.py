import pandas as pd
from datetime import datetime, timedelta
import io

# 1. Configuration & Filename parsing
# Filename: DataLogin_NodeID-491_01-19-2026_13-53-19_...
start_time_str = "13:53:19"
start_dt = datetime.strptime(start_time_str, "%H:%M:%S")

files = [
    "DataLogin_NodeID-491_01-19-2026_13-53-19_Gravity.txt",
    "DataLogin_NodeID-491_01-19-2026_13-53-19_Q_s1_e.txt",
    "DataLogin_NodeID-491_01-19-2026_13-53-19_RawData.txt",
    "DataLogin_NodeID-491_01-19-2026_13-53-19_RPY.txt"
]

def load_imu_file(filename):
    # We now use a regex that treats commas, whitespace, AND semicolons as delimiters.
    # This prevents the "extra column" issue regardless of spacing.
    df = pd.read_csv(filename, skiprows=4, skipfooter=1, engine='python', 
                     header=None, sep=r'[,\s;]+')
    
    # Clean up: remove any columns that are entirely NaN (caused by trailing delimiters)
    df = df.dropna(axis=1, how='all')
    return df

# 2. Load and Merge
print("Loading files...")
df_gravity = load_imu_file(files[0])
df_q = load_imu_file(files[1])
df_raw = load_imu_file(files[2])
df_rpy = load_imu_file(files[3])

# Set names based on file headers
df_gravity.columns = ['timeStamp', 'GravityX', 'GravityY', 'GravityZ']
df_q.columns = ['timeStamp', 'q1', 'q2', 'q3', 'q4']
df_raw.columns = ['timeStamp', 'temp', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ']
df_rpy.columns = ['timeStamp', 'Roll', 'Pitch', 'Yaw']

# Merge all data on the timeStamp column
print("Merging data...")
merged = df_raw.merge(df_rpy, on='timeStamp') \
               .merge(df_q, on='timeStamp') \
               .merge(df_gravity, on='timeStamp')

# 3. Calculate "Time of Day"
# The filename timestamp (13:53:19) corresponds to the first row of data
first_ts = merged['timeStamp'].iloc[0]
merged.insert(0, 'Time_of_Day', merged['timeStamp'].apply(
    lambda x: (start_dt + timedelta(seconds=(x - first_ts))).strftime("%H:%M:%S.%f")[:-2]
))

# 4. Save to CSV
output_file = "consolidated_imu_data.csv"
merged.to_csv(output_file, index=False)
print(f"Success! '{output_file}' has been created with {len(merged)} rows.")