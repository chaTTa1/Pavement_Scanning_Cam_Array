import os
import shutil
import glob

def merge_and_rename_images(parent_folder, output_folder):
    """
    Merges all images named 'frame_xxxxx.jpg' from subdirectories of parent_folder
    into output_folder, renaming them to maintain unique IDs.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all matching images in subdirectories
    image_paths = glob.glob(os.path.join(parent_folder, "*", "frame_*.jpg"))
    image_paths.sort()  # Optional: sort for consistent ordering

    for idx, img_path in enumerate(image_paths):
        new_name = f"frame_{idx:06d}.jpg"
        dest_path = os.path.join(output_folder, new_name)
        shutil.copy2(img_path, dest_path)
        print(f"Copied {img_path} -> {dest_path}")

# Example usage:
merge_and_rename_images(r"c:\Users\drone\Documents\Demos\SeptTest", r"c:\Users\drone\Documents\Demos\MergedImages")