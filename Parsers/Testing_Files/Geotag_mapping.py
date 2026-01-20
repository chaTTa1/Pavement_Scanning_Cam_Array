import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt

def get_gps_info(exif_data):
    gps_info = {}
    for key, val in exif_data.items():
        tag = TAGS.get(key)
        if tag == 'GPSInfo':
            for t in val:
                sub_tag = GPSTAGS.get(t)
                gps_info[sub_tag] = val[t]
    return gps_info

def dms_to_dd(dms, ref):
    # Convert IFDRational or tuple to float
    def to_float(x):
        try:
            return float(x)
        except TypeError:
            return float(x[0]) / float(x[1])
    degrees = to_float(dms[0])
    minutes = to_float(dms[1])
    seconds = to_float(dms[2])
    dd = degrees + minutes / 60 + seconds / 3600
    if ref in ['S', 'W']:
        dd *= -1
    return dd

def extract_gps_from_image(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if not exif_data:
        return None
    gps_info = get_gps_info(exif_data)
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        lat = dms_to_dd(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef', 'N'))
        lon = dms_to_dd(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef', 'E'))
        return lat, lon
    return None

def plot_geotagged_images(image_folder):
    coords = []
    labels = []
    for fname in sorted(os.listdir(image_folder)):
        if fname.lower().endswith('.jpg'):
            img_path = os.path.join(image_folder, fname)
            gps = extract_gps_from_image(img_path)
            if gps:
                coords.append(gps)
                labels.append(fname.split('.')[0])  # frame ID

    if not coords:
        print("No geotagged images found.")
        return

    lats, lons = zip(*coords)
    plt.figure(figsize=(10, 8))
    plt.scatter(lons, lats, c='blue', s=20)

    # Annotate every Nth frame for readability
    N = max(1, len(labels) // 50)
    for i, label in enumerate(labels):
        if i % N == 0:
            plt.annotate(label, (lons[i], lats[i]), fontsize=8, ha='right')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geotagged Image Positions')
    plt.tight_layout()
    plt.show()

# Example usage:
plot_geotagged_images(r"c:\Users\drone\Documents\Demos\MergedImages")