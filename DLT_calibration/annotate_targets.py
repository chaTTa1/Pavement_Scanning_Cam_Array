import os
import json
import glob
import cv2
import numpy as np

# Updated base directory and target count based on your folder structure
BASE_DIR = r"C:\Users\alexc\Documents\Research\dlt\DLT_test_Sept17"
CAMERAS = ["camera_1", "camera_2", "camera_3"]
NUM_TARGETS = 18
OUTPUT_FILE = os.path.join(BASE_DIR, "manual_2d_coordinates.json")

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

def refine_centroid(img_gray, click_x, click_y, window_size=25):
    h, w = img_gray.shape
    x1, x2 = max(0, click_x - window_size), min(w, click_x + window_size)
    y1, y2 = max(0, click_y - window_size), min(h, click_y + window_size)
    
    roi = img_gray[y1:y2, x1:x2]
    roi_inv = 255 - roi
    _, thresh = cv2.threshold(roi_inv, np.percentile(roi_inv, 70), 255, cv2.THRESH_TOZERO)
    
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = x1 + (M["m10"] / M["m00"])
        cy = y1 + (M["m01"] / M["m00"])
        return float(cx), float(cy)
    return float(click_x), float(click_y)

def main():
    global clicked_point
    results = {}

    cv2.namedWindow("Target Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Target Annotator", mouse_callback)

    for cam in CAMERAS:
        results[cam] = {}
        for idx in range(1, NUM_TARGETS + 1):
            # Matches target_001.* regardless of extension (.jpg, .jpeg, .png)
            pattern = os.path.join(BASE_DIR, cam, f"target_{idx:03d}.*")
            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"Skipping missing file: target_{idx:03d} in {cam}")
                continue

            img_path = matching_files[0]
            filename = os.path.basename(img_path)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clicked_point = None
            final_coord = None

            while True:
                display = img.copy()

                if clicked_point is not None:
                    cx, cy = refine_centroid(gray, clicked_point[0], clicked_point[1])
                    final_coord = (cx, cy)
                    cv2.circle(display, (int(round(cx)), int(round(cy))), 6, (0, 255, 0), -1)
                    cv2.putText(display, f"({cx:.2f}, {cy:.2f})", (int(cx) + 10, int(cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(display, f"{cam} - {filename} | Click dot, SPACE to save, 'r' to reset, 'q' to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Target Annotator", display)
                key = cv2.waitKey(20) & 0xFF

                if key == ord(' ') and final_coord is not None:
                    results[cam][filename] = final_coord
                    print(f"[{cam}] {filename} -> ({final_coord[0]:.3f}, {final_coord[1]:.3f})")
                    break
                elif key == ord('r'):
                    clicked_point = None
                    final_coord = None
                elif key == ord('q'):
                    print("Annotation aborted.")
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved coordinates to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()