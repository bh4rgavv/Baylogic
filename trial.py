import cv2
import re
import json
import os
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from datetime import datetime
from LPRNet import build_lprnet


normal_model = YOLO("runs/detect/train-4/weights/best_er.pt")

cap = cv2.VideoCapture("video.mp4")
total_seconds = 0*3600 + 18*60 + 0
in_frames = total_seconds * 9
cap.set(cv2.CAP_PROP_POS_FRAMES, in_frames)
print("Video opened:", cap.isOpened())
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

roi_x1,  roi_y1,  roi_x2,  roi_y2  = 300,  100, 1000, 970
roi2_x1, roi2_y1, roi2_x2, roi2_y2 = 1200, 211, 1700, 900

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

frame_count           = 0
bay1_entry_time       = None
bay2_entry_time       = None
bay1_prev             = False
bay2_prev             = False
bay1_last_plate       = ""
bay2_last_plate       = ""
bay1_last_plate_color = "Unknown"
bay2_last_plate_color = "Unknown"

while frame_count < 5400:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    cv2.rectangle(frame, (roi_x1,  roi_y1),  (roi_x2,  roi_y2),  (0, 255, 0), 2)
    cv2.rectangle(frame, (roi2_x1, roi2_y1), (roi2_x2, roi2_y2), (0, 255, 0), 2)

    base_results   = normal_model(frame, verbose=False)
    print(base_results[0].boxes)
    print(normal_model.names)

    bay1_vehicle      = False
    bay2_vehicle      = False
    bay1_Plate        = False
    bay2_Plate        = False
    bay1_plate_text   = "Not Detected"
    bay2_plate_text   = "Not Detected"
    bay1_plate_color  = "Unknown"
    bay2_plate_color  = "Unknown"

    for box in base_results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 1:
            label, color = "Person", (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif cls == 2:
            label, color = "technician", (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()