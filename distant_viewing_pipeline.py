import os
import cv2
import pandas as pd
import numpy as np
import face_recognition
from ultralytics import YOLO
from tqdm import tqdm

# Load YOLO model
model = YOLO("yolov5s.pt")

# Paths
folder = "/Users/rsmirnov/Desktop/screenshots"
output_csv = "distant_viewing_dataset.csv"
error_log = "errors.log"

# Load processed files if CSV already exists
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    processed_files = set(df["filename"])
else:
    df = pd.DataFrame()
    processed_files = set()

batch_size = 10
data_batch = []

files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
files_to_process = [f for f in files if f not in processed_files]

with open(error_log, "a") as log_file:
    for i, file in enumerate(tqdm(files_to_process, desc="Processing images"), start=1):
        try:
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Unreadable image")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Average color and brightness
            avg_color = img_rgb.mean(axis=(0, 1))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()

            # Color histograms
            hist_r = np.histogram(img_rgb[:, :, 0], bins=8, range=(0, 256))[0]
            hist_g = np.histogram(img_rgb[:, :, 1], bins=8, range=(0, 256))[0]
            hist_b = np.histogram(img_rgb[:, :, 2], bins=8, range=(0, 256))[0]

            # Face detection
            face_locations = face_recognition.face_locations(img_rgb)
            num_faces = len(face_locations)

            # YOLO object detection
            results = model.predict(img_rgb, verbose=False)
            labels = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    labels.append(r.names[cls_id])

            # Save row
            row = {
                "filename": file,
                "width": w,
                "height": h,
                "avg_R": avg_color[0],
                "avg_G": avg_color[1],
                "avg_B": avg_color[2],
                "brightness": brightness,
                "num_faces": num_faces,
                "objects_detected": ", ".join(labels),
                **{f"hist_r_{j}": hist_r[j] for j in range(8)},
                **{f"hist_g_{j}": hist_g[j] for j in range(8)},
                **{f"hist_b_{j}": hist_b[j] for j in range(8)},
            }
            data_batch.append(row)

            # Save batch
            if i % batch_size == 0 or i == len(files_to_process):
                df = pd.concat([df, pd.DataFrame(data_batch)], ignore_index=True)
                df.to_csv(output_csv, index=False)
                data_batch = []

        except Exception as e:
            log_file.write(f"{file}: {str(e)}\n")

print("All images processed! Results saved to:", output_csv)
print("⚠Errors (if any) are in:", error_log)
