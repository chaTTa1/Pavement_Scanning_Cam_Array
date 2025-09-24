import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from exif import Image

def dms_to_dd(dms, ref):
    """
    Converts GPS coordinates from DMS (Degrees, Minutes, Seconds) to DD (Decimal Degrees).

    Args:
        dms (tuple): A tuple of (degrees, minutes, seconds).
        ref (str): The direction reference, e.g., 'N', 'S', 'E', 'W'.

    Returns:
        float: The coordinate in Decimal Degrees.
    """
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if ref in ['S', 'W']:
        dd *= -1
    return dd

def get_gps_data(image_path):
    """
    Extracts GPS coordinates and filename from an image file's EXIF data.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        dict or None: A dictionary with 'filename', 'latitude', and 'longitude',
                      or None if no GPS data is found.
    """
    try:
        with open(image_path, 'rb') as f:
            img = Image(f)

        if not all(hasattr(img, attr) for attr in ['gps_latitude', 'gps_longitude', 'gps_latitude_ref', 'gps_longitude_ref']):
            return None

        lat_dms = img.gps_latitude
        lat_ref = img.gps_latitude_ref
        lon_dms = img.gps_longitude
        lon_ref = img.gps_longitude_ref

        latitude = dms_to_dd(lat_dms, lat_ref)
        longitude = dms_to_dd(lon_dms, lon_ref)

        return {
            'filename': os.path.basename(image_path),
            'latitude': latitude,
            'longitude': longitude
        }

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

def create_geotag_plot(image_directory, output_filename='geotag_plot.png'):
    """
    Scans a directory for images, extracts their geotags, and creates a scatter plot.

    Args:
        image_directory (str): The path to the directory containing images.
        output_filename (str): The name of the output PNG plot file.
    """
    image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(image_directory, '*.jpeg')))
    image_paths.extend(glob.glob(os.path.join(image_directory, '*.png')))

    if not image_paths:
        print(f"No images found in '{image_directory}'.")
        return

    print(f"Found {len(image_paths)} images. Processing...")
    
    # Extract GPS data from all images
    all_gps_data = [get_gps_data(p) for p in image_paths]
    
    # Filter out any images that didn't have GPS data
    locations_data = [d for d in all_gps_data if d is not None]

    if not locations_data:
        print("\nCould not find GPS data in any of the images. No plot was generated.")
        return

    # Convert the list of dictionaries to a pandas DataFrame for easy plotting
    df = pd.DataFrame(locations_data)

    print(f"Plotting {len(df)} locations...")

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12)) # Create a square plot

    sns.scatterplot(
        data=df,
        x='longitude',
        y='latitude',
        ax=ax,
        s=100, # Set marker size
        edgecolor='black',
        linewidth=0.5
    )
    
    # Improve the plot's appearance
    ax.set_title('Scatter Plot of Image Geotags', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Ensure the aspect ratio is equal, so spatial relationships are not distorted
    ax.set_aspect('equal', adjustable='box')
    
    # Use a tight layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_filename, dpi=300) # Save as a high-resolution image
    
    print(f"\nPlot successfully created! See the file '{output_filename}'.")

if __name__ == '__main__':

    path_to_your_images = r"C:\Users\drone\Documents\Demos\test_july31"
    
    # On Windows: path_to_your_images = 'C:\\Users\\YourUser\\Pictures\\Vacation'
    # On macOS/Linux: path_to_your_images = '/home/YourUser/Pictures/Vacation'

    if not os.path.isdir(path_to_your_images):
        print(f"Error: The directory '{path_to_your_images}' does not exist.")
        print("Please update the 'path_to_your_images' variable in the script.")
    else:
        create_geotag_plot(path_to_your_images)