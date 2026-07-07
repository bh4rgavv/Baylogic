import json
import os
from config import LOG_FILE

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_log(records):
    with open(LOG_FILE, "w") as f:
        json.dump(records, f, indent=4)

def log_exit(bay, plate, plate_color, entry_time, exit_time, technician_seconds):
    records  = load_log()
    duration = exit_time - entry_time
    total_s  = int(duration.total_seconds())
    if total_s < 5:
        print(f"[SKIPPED] Bay {bay} duration too short ({total_s}s)")
        return
    record = {
        "bay":                 bay,
        "plate":               plate if plate else "Unknown",
        "plate_color":         plate_color,
        "vehicle_type":        "EV" if plate_color == "Green" else "Regular",
        "entry_time":          entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time":           exit_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds":    total_s,
        "technician_seconds":  technician_seconds,
        "time the vehicle was operated on" : f"{total_s-technician_seconds}"
        "technician_duration": f"{technician_seconds // 3600}h {(technician_seconds % 3600) // 60}m {technician_seconds % 60}s"
    }
    records.append(record)
    save_log(records)
    print(f"[LOG SAVED] {record}")
