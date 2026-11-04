"""Sends the camera to the dashboard so you can check the Camera box.

Run while the dashboard server is running and no robot mode is active
(only one program can use the camera at a time):

    python dashboard/camera_test.py            # camera 0
    python dashboard/camera_test.py 1          # another camera number

Press Ctrl+C to stop.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dashboard.telemetry_client import send_frame

import cv2

cam = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
if not cam.isOpened():
    sys.exit("Could not open the camera. Close other programs using it, or try another camera number.")

print("Sending camera frames to the dashboard. Press Ctrl+C to stop.")
try:
    while True:
        ok, frame = cam.read()
        if ok:
            send_frame(frame)
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
finally:
    cam.release()
