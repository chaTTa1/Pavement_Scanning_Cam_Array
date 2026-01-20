import cv2
import os
import re
import shutil

# --- Parameters ---
input_folder = r"C:\Users\drone\Documents\Demos\SeptTest"
output_folder = "selected_images"
os.makedirs(output_folder, exist_ok=True)

# Regex to extract frame numbers like 000123 from "frame_000123.jpg"
frame_regex = re.compile(r"frame_(\d+)\.jpg")

def image_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1  # in case of bad image
    return cv2.Laplacian(img, cv2.CV_64F).var()

# Get all images + extract frame numbers
images = []
for f in os.listdir(input_folder):
    match = frame_regex.match(f)
    if match:
        frame_num = int(match.group(1))
        images.append((frame_num, f))

# Sort by frame number
images.sort(key=lambda x: x[0])

# --- Group into nodes (consecutive IDs) ---
nodes = []
current_node = [images[0]]
for i in range(1, len(images)):
    prev_frame, prev_name = images[i-1]
    curr_frame, curr_name = images[i]
    if curr_frame == prev_frame + 1:  # still same node
        current_node.append(images[i])
    else:  # gap → start new node
        nodes.append(current_node)
        current_node = [images[i]]
# Add last group
if current_node:
    nodes.append(current_node)

print(f"Found {len(nodes)} nodes")

# --- Select sharpest per node ---
selected = []
for idx, node in enumerate(nodes):
    sharpest = max(node, key=lambda x: image_sharpness(os.path.join(input_folder, x[1])))
    selected.append(sharpest[1])
    shutil.copy(os.path.join(input_folder, sharpest[1]),
                os.path.join(output_folder, sharpest[1]))
    print(f"Node {idx+1}: selected {sharpest[1]}")

print("Done! Sharpest images saved to:", output_folder)
