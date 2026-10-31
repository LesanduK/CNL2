"""Optional: lets your existing scripts feed the dashboard. Copy-paste usage:

    from dashboard.telemetry_client import send, send_frame

    # inside the control loop (a few times per second is plenty)
    send(pose=[x, y, theta],              # metres, metres, radians (x right, y up, theta CCW)
         tof=[left, middle, right],       # metres
         rpm=[left_rpm, right_rpm],
         target={"bearing": deg, "dist": m},     # human follower: bearing > 0 means left
         path=[[x, y], ...],                     # navigation: planned waypoints
         obstacles=[[x, y], ...])                # navigation: world coordinates
    send_frame(bgr_image)                 # camera view with boxes drawn, throttled to ~8 fps

Every field is optional; the dashboard shows whatever it receives.
Data goes over UDP to localhost only, and is silently dropped if the dashboard is not running.
"""
import json
import socket
import time

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_addr = ("127.0.0.1", 9999)
_last_frame = 0.0


def send(**fields):
    try:
        _sock.sendto(json.dumps(fields).encode(), _addr)
    except OSError:
        pass


def send_frame(bgr, width=320, quality=55, max_fps=8):
    global _last_frame
    now = time.time()
    if now - _last_frame < 1.0 / max_fps:
        return
    _last_frame = now
    try:
        import cv2
        h = int(bgr.shape[0] * width / bgr.shape[1])
        ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (width, h)), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(buf) < 60000:
            _sock.sendto(b"F" + buf.tobytes(), _addr)
    except (OSError, ImportError):
        pass
