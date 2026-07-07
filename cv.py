#as this is a sample of the project I made during the internship i cannot add the pt files as i trained the models on the dataset that the company provided
#so please use your own pt files and make the changes as needed to the classes detections
import cv2
from collections import Counter
from ultralytics import YOLO
from config import ROI_X1, ROI_Y1, ROI_X2, ROI_Y2, ROI2_X1, ROI2_Y1, ROI2_X2, ROI2_Y2
from plate_utils import read_plate_lpr, get_plate_color
from detection_utils import is_person_shaped, boxes_overlap, in_bay
from logger import load_log, save_log
from db import create_table, insert_record

create_table()

custom_model = YOLO("runs/detect/train-4/weights/best.pt").to("cuda")
normal_model = YOLO("runs/detect/train-4/weights/best_er.pt")

cap = cv2.VideoCapture("video.mp4")
total_seconds = 1*3600 + 2*60 + 0
start_frame   = total_seconds * 9
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
print("Video opened:", cap.isOpened())
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

def get_video_time(cap):
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_secs    = current_frame / fps
    hours         = int(total_secs // 3600)
    minutes       = int((total_secs % 3600) // 60)
    seconds       = int(total_secs % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

frame_count             = 0
bay1_entry_time         = None
bay2_entry_time         = None
bay1_prev               = False
bay2_prev               = False
bay1_plate_votes        = []
bay2_plate_votes        = []
bay1_last_plate_color   = "Unknown"
bay2_last_plate_color   = "Unknown"
bay1_tech_start_frame   = None
bay2_tech_start_frame   = None
bay1_tech_working       = False
bay2_tech_working       = False
bay1_tech_total_seconds = 0
bay2_tech_total_seconds = 0
bay1_last_cx            = None
bay2_last_cx            = None

while frame_count < 5400:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.rectangle(frame, (ROI_X1,  ROI_Y1),  (ROI_X2,  ROI_Y2),  (0, 255, 0), 2)
    cv2.rectangle(frame, (ROI2_X1, ROI2_Y1), (ROI2_X2, ROI2_Y2), (0, 255, 0), 2)

    custom_results = custom_model(frame, verbose=False)
    base_results   = normal_model(frame, verbose=False)

    bay1_vehicle         = False
    bay2_vehicle         = False
    bay1_Plate           = False
    bay2_Plate           = False
    bay1_plate_text      = "Not Detected"
    bay2_plate_text      = "Not Detected"
    bay1_plate_color     = "Unknown"
    bay2_plate_color     = "Unknown"
    bay1_vehicle_box     = None
    bay2_vehicle_box     = None
    bay1_technician_box  = None
    bay2_technician_box  = None
    current_bay1_cx      = None
    current_bay2_cx      = None
    outside_bay1_cx      = None
    outside_bay2_cx      = None

    for box in base_results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 1:
            if is_person_shaped(x1, x2, y1, y2) and conf > 0.4:
                label, color = "Person", (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif cls == 2:
            label, color = "Technician", (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if in_bay(cx, cy, 1): bay1_technician_box = (x1, y1, x2, y2)
            if in_bay(cx, cy, 2): bay2_technician_box = (x1, y1, x2, y2)

    for box in custom_results[0].boxes:
        cls        = int(box.cls[0])
        conf       = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = custom_results[0].names[cls]

        if class_name == "Vehicle-Detection":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_b1  = in_bay(cx, cy, 1)
            in_b2  = in_bay(cx, cy, 2)
            if not (in_b1 or in_b2):
                outside_bay1_cx = cx
                outside_bay2_cx = cx
                continue
            if in_b1:
                bay1_vehicle     = True
                bay1_vehicle_box = (x1, y1, x2, y2)
                current_bay1_cx  = cx
            if in_b2:
                bay2_vehicle     = True
                bay2_vehicle_box = (x1, y1, x2, y2)
                current_bay2_cx  = cx
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Car {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif class_name == "Number-Plate":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not (in_bay(cx, cy, 1) or in_bay(cx, cy, 2)):
                continue
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            plate_color = get_plate_color(plate_crop)
            plate_text  = read_plate_lpr(plate_crop)
            if in_bay(cx, cy, 1):
                bay1_Plate, bay1_plate_text, bay1_plate_color = True, plate_text, plate_color
                if plate_text:
                    bay1_plate_votes.append(plate_text)
                    bay1_last_plate_color = plate_color
            if in_bay(cx, cy, 2):
                bay2_Plate, bay2_plate_text, bay2_plate_color = True, plate_text, plate_color
                if plate_text:
                    bay2_plate_votes.append(plate_text)
                    bay2_last_plate_color = plate_color
            box_color = (0, 255, 0) if plate_color == "Green" else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"Plate {conf*100:.1f}% {plate_color}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    if bay1_vehicle_box and bay1_technician_box and boxes_overlap(bay1_vehicle_box, bay1_technician_box):
        if not bay1_tech_working:
            bay1_tech_start_frame = frame_count
            bay1_tech_working     = True
    else:
        if bay1_tech_working:
            bay1_tech_total_seconds += int((frame_count - bay1_tech_start_frame) / fps)
            bay1_tech_working = False

    if bay2_vehicle_box and bay2_technician_box and boxes_overlap(bay2_vehicle_box, bay2_technician_box):
        if not bay2_tech_working:
            bay2_tech_start_frame = frame_count
            bay2_tech_working     = True
    else:
        if bay2_tech_working:
            bay2_tech_total_seconds += int((frame_count - bay2_tech_start_frame) / fps)
            bay2_tech_working = False

    if bay1_vehicle and not bay1_prev:
        if bay1_last_cx is not None and current_bay1_cx is not None:
            bay1_direction = "entering" if current_bay1_cx > bay1_last_cx else "leaving"
        else:
            bay1_direction = "entering"
        bay1_entry_time         = get_video_time(cap)
        bay1_entry_frame        = frame_count
        bay1_plate_votes        = []
        bay1_last_plate_color   = "Unknown"
        bay1_tech_total_seconds = 0
        print(f"[BAY 1 ENTRY] Vehicle entered at {bay1_entry_time} direction: {bay1_direction}")

    if not bay1_vehicle and bay1_prev:
        if bay1_tech_working:
            bay1_tech_total_seconds += int((frame_count - bay1_tech_start_frame) / fps)
            bay1_tech_working = False
        if bay1_entry_time is not None:
            exit_time      = get_video_time(cap)
            duration_secs  = int((frame_count - bay1_entry_frame) / fps)
            tech_secs      = bay1_tech_total_seconds
            time_the_vehicle_was_operated_on= duration_secs-tech_secs
            tech_human     = f"{tech_secs // 3600}h {(tech_secs % 3600) // 60}m {tech_secs % 60}s"
            direction      = "leaving"
            print(f"[BAY 1 EXIT] {exit_time}")
            records = load_log()
            record  = {
                "bay":                 1,
                "direction":           direction,
                "entry_time":          bay1_entry_time,
                "exit_time":           exit_time,
                "duration_seconds":    duration_secs,
                "time_the_vehicle_was_operated_on": time_the_vehicle_was_operated_on,
                "technician_seconds":  tech_secs,
                "technician_duration": tech_human
            }
            if duration_secs >= 5:
                records.append(record)
                save_log(records)
                insert_record(record)
                print(f"[LOG SAVED] {record}")
            else:
                print(f"[SKIPPED] Bay 1 duration too short ({duration_secs}s)")
        bay1_entry_time = None

    if bay2_vehicle and not bay2_prev:
        if bay2_last_cx is not None and current_bay2_cx is not None:
            bay2_direction = "entering" if current_bay2_cx > bay2_last_cx else "leaving"
        else:
            bay2_direction = "entering"
        bay2_entry_time         = get_video_time(cap)
        bay2_entry_frame        = frame_count
        bay2_plate_votes        = []
        bay2_last_plate_color   = "Unknown"
        bay2_tech_total_seconds = 0
        print(f"[BAY 2 ENTRY] Vehicle entered at {bay2_entry_time} direction: {bay2_direction}")

    if not bay2_vehicle and bay2_prev:
        if bay2_tech_working:
            bay2_tech_total_seconds += int((frame_count - bay2_tech_start_frame) / fps)
            bay2_tech_working = False
        if bay2_entry_time is not None:
            exit_time      = get_video_time(cap)
            duration_secs  = int((frame_count - bay2_entry_frame) / fps)
            tech_secs      = bay2_tech_total_seconds
            time_the_vehicle_was_operated_on= bay2_tech_total_seconds-duration_secs
            tech_human     = f"{tech_secs // 3600}h {(tech_secs % 3600) // 60}m {tech_secs % 60}s"
            direction      = "leaving"
            print(f"[BAY 2 EXIT] {exit_time}")
            records = load_log()
            record  = {
                "bay":                 2,
                "direction":           direction,
                "entry_time":          bay2_entry_time,
                "exit_time":           exit_time,
                "duration_seconds":    duration_secs,
                "technician_seconds":  tech_secs,
                "time_the_vehicle_was_operated_on": time_the_vehicle_was_operated_on,
                "technician_duration": tech_human
            }
            if duration_secs >= 5:
                records.append(record)
                save_log(records)
                insert_record(record)
                print(f"[LOG SAVED] {record}")
            else:
                print(f"[SKIPPED] Bay 2 duration too short ({duration_secs}s)")
        bay2_entry_time = None

    if outside_bay1_cx is not None:
        bay1_last_cx = outside_bay1_cx
    elif current_bay1_cx is not None:
        bay1_last_cx = current_bay1_cx
    elif not bay1_vehicle:
        bay1_last_cx = None

    if outside_bay2_cx is not None:
        bay2_last_cx = outside_bay2_cx
    elif current_bay2_cx is not None:
        bay2_last_cx = current_bay2_cx
    elif not bay2_vehicle:
        bay2_last_cx = None

    bay1_prev = bay1_vehicle
    bay2_prev = bay2_vehicle

    cv2.putText(frame, "Bay 1: Car Present" if bay1_vehicle else "Bay 1: Empty",            (30, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
    cv2.putText(frame, "Bay 2: Car Present" if bay2_vehicle else "Bay 2: Empty",            (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
    cv2.putText(frame, "Number Plate 1: Present" if bay1_Plate else "Number Plate 1: Empty", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
    cv2.putText(frame, "Number Plate 2: Present" if bay2_Plate else "Number Plate 2: Empty", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 255, 255), 2)
    cv2.putText(frame, f"Bay1 Plate: {bay1_plate_text}",                                     (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Bay2 Plate: {bay2_plate_text}",                                     (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if bay1_entry_time is not None:
        current_vid  = get_video_time(cap)
        entry_secs   = int(sum(int(x) * 60**i for i, x in enumerate(reversed(bay1_entry_time.split(":")))))
        current_secs = int(sum(int(x) * 60**i for i, x in enumerate(reversed(current_vid.split(":")))))
        elapsed      = current_secs - entry_secs
        cv2.putText(frame, f"Bay1 Time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if bay2_entry_time is not None:
        current_vid  = get_video_time(cap)
        entry_secs   = int(sum(int(x) * 60**i for i, x in enumerate(reversed(bay2_entry_time.split(":")))))
        current_secs = int(sum(int(x) * 60**i for i, x in enumerate(reversed(current_vid.split(":")))))
        elapsed      = current_secs - entry_secs
        cv2.putText(frame, f"Bay2 Time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if bay1_tech_working and bay1_tech_start_frame:
        tech_elapsed = int((frame_count - bay1_tech_start_frame) / fps)
        cv2.putText(frame, f"Bay1 Tech Time: {tech_elapsed//3600:02d}:{(tech_elapsed%3600)//60:02d}:{tech_elapsed%60:02d}", (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    if bay2_tech_working and bay2_tech_start_frame:
        tech_elapsed = int((frame_count - bay2_tech_start_frame) / fps)
        cv2.putText(frame, f"Bay2 Tech Time: {tech_elapsed//3600:02d}:{(tech_elapsed%3600)//60:02d}:{tech_elapsed%60:02d}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
