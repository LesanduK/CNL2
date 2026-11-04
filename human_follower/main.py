"""
human_follower/main.py
----------------------
CLI entry point for Stage 1 Human Follower (Stationary Turret Tracking).

Usage:
    # Live run on Raspberry Pi 4 (with camera and ESP32 connected):
    python3 human_follower/main.py

    # Headless run over SSH (no monitor attached):
    python3 human_follower/main.py --headless

    # Mock simulation mode (no hardware required, test anywhere):
    python3 human_follower/main.py --mock --headless

    # Custom tuning parameters:
    python3 human_follower/main.py --kp 0.50 --deadband 0.06 --max-speed 0.35
"""

import argparse
import math
import os
import sys
import time

# Support running directly or as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from human_follower.config import FollowerConfig
from human_follower.turret_follower import TurretFollower

try:  # live data for the web dashboard (does nothing if the dashboard folder is missing)
    from dashboard.telemetry_client import send, send_frame
except ImportError:
    def send(**fields):
        pass

    def send_frame(*args, **kwargs):
        pass


def draw_turret_overlay(frame, telem):
    """Draw tracking crosshairs, rotation indicator, and wheel speeds on camera preview."""
    try:
        import cv2
    except ImportError:
        return frame

    h, w = frame.shape[:2]
    cx_frame = w // 2
    cy_frame = h // 2

    # Draw center neutral zone (deadband lines)
    # The deadband defines the "sweet spot" where the robot stays completely still
    deadband_px = int(0.08 * cx_frame)
    cv2.line(frame, (cx_frame - deadband_px, 0), (cx_frame - deadband_px, h), (60, 60, 60), 1)
    cv2.line(frame, (cx_frame + deadband_px, 0), (cx_frame + deadband_px, h), (60, 60, 60), 1)

    # Frame center crosshair
    cv2.drawMarker(frame, (cx_frame, cy_frame), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    state = telem.get("state", "SEARCHING")
    vl = telem.get("v_left", 0.0)
    vr = telem.get("v_right", 0.0)
    h_err = telem.get("h_err", 0.0)
    bearing = telem.get("bearing_deg", 0.0)
    conf = telem.get("confidence", 0.0)

    # Draw target bounding box if available
    bbox = telem.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Bounding box (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Target center crosshair (Red)
        tcx = (x1 + x2) // 2
        tcy = (y1 + y2) // 2
        cv2.drawMarker(frame, (tcx, tcy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        # Vector line to center (Yellow)
        cv2.line(frame, (cx_frame, cy_frame), (tcx, tcy), (0, 255, 255), 1)
        # Tracking label
        cv2.putText(
            frame, f"TRACKING: {conf:.0%}", (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    # State colors and banner
    if state == "LOCKED_CENTER":
        color = (0, 255, 0)  # Green
        banner = "TARGET CENTERED (MOTORS IDLE)"
    elif state == "PIVOT_RIGHT":
        color = (0, 165, 255)  # Orange
        banner = f"PIVOTING RIGHT >>> (L: {vl:+.2f}, R: {vr:+.2f} m/s)"
    elif state == "PIVOT_LEFT":
        color = (0, 165, 255)  # Orange
        banner = f"<<< PIVOTING LEFT (L: {vl:+.2f}, R: {vr:+.2f} m/s)"
    elif "SEARCH_RECOVERY" in state:
        color = (0, 165, 255)  # Orange
        direction = "RIGHT >>>" if "RIGHT" in state else "<<< LEFT"
        banner = f"TARGET LOST: SWEEPING {direction} TO REACQUIRE..."
    else:
        color = (0, 0, 255)  # Red
        banner = "SEARCHING FOR PERSON..."

    # Top banner
    cv2.putText(frame, banner, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Telemetry footer
    footer = f"State: {state} | Err: {h_err:+.2f} | Bearing: {bearing:+.1f} deg | FPS: {telem.get('fps', 0.0)}"
    cv2.putText(frame, footer, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Human Follower: Stationary Turret Tracking")
    parser.add_argument("--mock", action="store_true", help="Run with simulated camera, detector, and serial")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--port", type=str, default="/dev/serial0", help="ESP32 serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--kp", type=float, default=0.35, help="Proportional turn gain (default: 0.35)")
    parser.add_argument("--kd", type=float, default=0.12, help="Derivative damping gain (default: 0.12)")
    parser.add_argument("--deadband-exit", type=float, default=0.10, help="Error threshold to start pivoting (default: 0.10)")
    parser.add_argument("--deadband-enter", type=float, default=0.05, help="Error threshold to settle and stop (default: 0.05)")
    parser.add_argument("--min-speed", type=float, default=0.12, help="Minimum gentle turn speed (default: 0.12 m/s)")
    parser.add_argument("--max-speed", type=float, default=0.28, help="Maximum turn speed clamp (default: 0.28 m/s)")
    parser.add_argument("--confidence", type=float, default=0.40, help="Person detection confidence threshold")
    args = parser.parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless = args.headless or not has_display

    config = FollowerConfig(
        serial_port=args.port,
        serial_baud=args.baud,
        kp_turn=args.kp,
        kd_turn=args.kd,
        deadband_exit=args.deadband_exit,
        deadband_enter=args.deadband_enter,
        min_turn_speed=args.min_speed,
        max_turn_speed=args.max_speed,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        enable_preview=not headless,
        headless=headless,
        mock=args.mock,
    )

    print("=" * 65)
    print("  STAGE 1 HUMAN FOLLOWER: STATIONARY TURRET TRACKING")
    print("=" * 65)
    print(f"  Mode:            {'MOCK SIMULATION' if config.mock else 'HARDWARE (Camera + ESP32)'}")
    print(f"  Serial Port:     {config.serial_port} @ {config.serial_baud} baud")
    print(f"  Gains:           Kp={config.kp_turn:.2f}, Kd={config.kd_turn:.2f} (Damping virtual brake)")
    print(f"  Deadband:        Exit={config.deadband_exit:.2f} (10%) | Enter={config.deadband_enter:.2f} (5%)")
    print(f"  Turn Speeds:     {config.min_turn_speed:.2f} m/s min -> {config.max_turn_speed:.2f} m/s max")
    print(f"  Forward Speed:   0.00 m/s (Stationary Pivot Only)")
    print(f"  Headless:        {config.headless}")
    print("=" * 65)
    print("Press Ctrl+C (or 'q' in preview window) to stop.\n")

    follower = TurretFollower(config)
    if not follower.start():
        print("Error: Could not start Turret Follower.")
        sys.exit(1)

    cv2 = None
    if not config.headless:
        try:
            import cv2
        except ImportError:
            pass

    last_log_t = 0.0
    last_dash_t = 0.0

    try:
        while True:
            telem = follower.get_telemetry()
            now = time.monotonic()

            # Live data for the web dashboard (harmless if the dashboard is not running)
            if now - last_dash_t >= 0.12:
                last_dash_t = now
                rpm_k = 60.0 / (2 * math.pi * config.wheel_radius_m)  # commanded m/s -> rpm
                fields = {"rpm": [float(telem.get("v_left", 0.0) or 0.0) * rpm_k, float(telem.get("v_right", 0.0) or 0.0) * rpm_k]}
                if telem.get("target_valid"):
                    # stage 1 has no distance estimate; the dashboard draws the person at a nominal range
                    fields["target"] = {"bearing": -telem.get("bearing_deg", 0.0), "dist": None}
                else:
                    fields["target"] = None
                send(**fields)
                if config.headless:  # with a preview window the frame is sent from the preview code below
                    ret, frame, _ = follower.vision.camera.read()
                    if ret and frame is not None and hasattr(frame, "shape"):
                        send_frame(draw_turret_overlay(frame.copy(), telem))

            # Terminal log update every 0.4s
            if now - last_log_t >= 0.4:
                state = telem.get("state", "SEARCHING")
                vl = telem.get("v_left", 0.0)
                vr = telem.get("v_right", 0.0)
                herr = telem.get("h_err", 0.0)
                b_deg = telem.get("bearing_deg", 0.0)
                fps = telem.get("fps", 0.0)
                print(
                    f"[{state:^13}] H-Err: {herr:+5.2f} | Bearing: {b_deg:+5.1f}° | "
                    f"Wheels: L={vl:+5.2f}, R={vr:+5.2f} m/s | FPS: {fps:4.1f}"
                )
                last_log_t = now

            # GUI Preview Window
            if cv2 is not None and not config.headless:
                ret, frame, _ = follower.vision.camera.read()
                if ret and frame is not None and hasattr(frame, "shape"):
                    annotated = draw_turret_overlay(frame.copy(), telem)
                    send_frame(annotated)  # also show it in the web dashboard
                    cv2.imshow("Stage 1 Turret Follower", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\nUser requested stop via window.")
                        break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopping Turret Follower...")

    finally:
        follower.stop()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Turret Follower safely stopped. Motors halted.")


if __name__ == "__main__":
    main()

