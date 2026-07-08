# Intelligent Service Bay Monitoring

This project monitors vehicle service bays using YOLOv8 and OpenCV. It detects vehicles, technicians, and number plates, tracks when vehicles enter and leave each bay, calculates service and technician time, and stores the results in a SQL database.

> **Note:** This is a sample version of the project developed during my internship. The trained YOLO weight files are not included because they were trained using proprietary company data. To run the project, use your own `.pt` files and update the class mappings if required.

## Features

* Detects vehicles inside predefined service bays
* Detects technicians working on vehicles
* Reads Indian number plates using OCR
* Identifies electric vehicles using number plate color (HSV)
* Tracks vehicle entry and exit
* Calculates total bay occupancy and technician working time
* Saves logs to MySQL and JSON
* Displays live detection results using OpenCV

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* EasyOCR
* SQLite/MySQL

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python cv.py
```

## Files

```
cv.py                 Main application
config.py             ROI configuration
detection_utils.py    Helper functions for detection
plate_utils.py        OCR and plate color detection
logger.py             JSON logging
db.py                 MySQL database functions
requirements.txt
```

## Output

<img width="1032" height="1017" alt="Screenshot 2026-07-07 152103" src="https://github.com/user-attachments/assets/fea59ad6-f9ee-4855-82a4-ca9c0f6cc986" />

The program displays the live video feed with:

* Vehicle detection
* Technician detection
* Number plate recognition
* Bay occupancy status
* Vehicle timer (present in bay)
* Technician timer (working on the vehicle)

Completed service records are automatically stored in the database.

## Future Improvements

* Live CCTV stream support
* Dashboard for viewing logs
* Multi-camera support
* Web interface

