"""
human_follower/main_stage2.py
-----------------------------
CLI entry point for Stage 2 Human Follower: Combined Distance Tracking & Heading Steering.

Usage:
    # Live run on Raspberry Pi 4 (with camera, VL53L0X ToF lasers, and ESP32 connected):
    python3 human_follower/main_stage2.py

    # Headless run over SSH (no monitor attached):
    python3 human_follower/main_stage2.py --headless

    # Mock simulation mode (no hardware required, test anywhere):
    python3 human_follower/main_stage2.py --mock --headless

    # Custom following distance (e.g. follow at 1.5m, max speed 0.35 m/s):
    python3 human_follower/main_stage2.py --target-dist 1.50 --max-speed 0.35
"""

import argparse
import math
import os
import sys
import time

# Support running directly or as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from human_follower.stage2_follower import Stage2Config, Stage2DistanceFollower

try:  # live data for the web dashboard (does nothing if the dashboard folder is missing)
    from dashboard.telemetry_client import send, send_frame
except ImportError:
    def send(**fields):
        pass

    def send_frame(*args, **kwargs):
        pass


def draw_stage2_overlay(frame, telem):
    """Draw tracking bounding box, distance meters, steering indicators, and state on preview."""
    try:
        import cv2
    except ImportError:
        return frame

    h, w = frame.shape[:2]
    cx_frame = w // 2
    cy_frame = h // 2

    # Draw center neutral zone (deadband lines)
    deadband_px = int(0.08 * cx_frame)
    cv2.line(frame, (cx_frame - deadband_px, 0), (cx_frame - deadband_px, h), (60, 60, 60), 1)
    cv2.line(frame, (cx_frame + deadband_px, 0), (cx_frame + deadband_px, h), (60, 60, 60), 1)

    # Frame center crosshair
    cv2.drawMarker(frame, (cx_frame, cy_frame), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    state = telem.get("state") or "SEARCHING"
    raw_dist = telem.get("dist_m")
    dist = float(raw_dist) if raw_dist is not None else 0.0
    source = telem.get("dist_source") or "UNK"
    vl = float(telem.get("v_left", 0.0) or 0.0)
    vr = float(telem.get("v_right", 0.0) or 0.0)
    h_err = float(telem.get("h_err", 0.0) or 0.0)
    bearing = float(telem.get("bearing_deg", 0.0) or 0.0)
    raw_conf = telem.get("confidence")
    conf = float(raw_conf) if raw_conf is not None else 0.0

    # Draw target bounding box if available
    bbox = telem.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Target bounding box (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Target center crosshair (Red)
        tcx = (x1 + x2) // 2
        tcy = (y1 + y2) // 2
        cv2.drawMarker(frame, (tcx, tcy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

        # Vector line connecting frame center to target center (Yellow)
        cv2.line(frame, (cx_frame, cy_frame), (tcx, tcy), (0, 255, 255), 1)

        # Tracking badge above the box
        label = f"TRACKING: {conf:.0%} | {dist:.2f}m [{source}]"
        cv2.putText(
            frame, label, (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

    # State colors and banners
    if state == "HOLD_SWEET_SPOT":
        color = (0, 255, 0)      # Green
        banner = f"IN SWEET SPOT ({dist:.2f}m) - HOLDING POSITION"
    elif "FOLLOW" in state:
        color = (255, 200, 0)    # Cyan / Yellow
        banner = f"FOLLOWING ({dist:.2f}m via {source}) -> Wheels: L={vl:+.2f}, R={vr:+.2f} m/s"
    elif "PIVOT" in state:
        color = (0, 165, 255)    # Orange
        banner = f"PIVOTING TO CENTER ({bearing:+.1f} deg) -> Wheels: L={vl:+.2f}, R={vr:+.2f} m/s"
    elif "SEARCH_RECOVERY" in state:
        color = (0, 165, 255)    # Orange
        direction = "RIGHT >>>" if "RIGHT" in state else "<<< LEFT"
        banner = f"TARGET LOST: SWEEPING {direction} TO REACQUIRE..."
    elif state == "SAFETY_STOP":
        color = (0, 0, 255)      # Red
        banner = f"SAFETY STOP: TARGET TOO CLOSE ({dist:.2f}m < 0.45m)"
    else:
        color = (0, 0, 255)      # Red
        banner = "SEARCHING FOR PERSON..."

    # Top banner
    cv2.putText(frame, banner, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    # Telemetry footer
    footer = f"State: {state} | Dist: {dist:.2f}m [{source}] | Err: {h_err:+.2f} | FPS: {telem.get('fps', 0.0)}"
    cv2.putText(frame, footer, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Human Follower: Distance & Heading Tracking")
    parser.add_argument("--mock", action="store_true", help="Run with simulated camera, sensors, and serial")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--port", type=str, default="/dev/serial0", help="ESP32 serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--target-dist", type=float, default=1.20, help="Desired following distance in meters (default: 1.20m)")
    parser.add_argument("--max-speed", type=float, default=0.40, help="Maximum forward speed (default: 0.40 m/s)")
    parser.add_argument("--kp-dist", type=float, default=0.55, help="Proportional distance gain (default: 0.55)")
    parser.add_argument("--kp-turn", type=float, default=0.35, help="Proportional turn gain (default: 0.35)")
    parser.add_argument("--kd-turn", type=float, default=0.12, help="Derivative damping gain (default: 0.12)")
    parser.add_argument("--confidence", type=float, default=0.40, help="Person detection confidence threshold")
    args = parser.parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless = args.headless or not has_display

    config = Stage2Config(
        serial_port=args.port,
        serial_baud=args.baud,
        target_distance_m=args.target_dist,
        max_forward_speed=args.max_speed,
        kp_dist=args.kp_dist,
        kp_turn=args.kp_turn,
        kd_turn=args.kd_turn,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        enable_preview=not headless,
        headless=headless,
        mock=args.mock,
    )

    print("=" * 65)
    print("  STAGE 2 HUMAN FOLLOWER: DISTANCE REGULATION & HEADING STEERING")
    print("=" * 65)
    print(f"  Mode:            {'MOCK SIMULATION' if config.mock else 'HARDWARE (Camera + ToF + ESP32)'}")
    print(f"  Serial Port:     {config.serial_port} @ {config.serial_baud} baud")
    print(f"  Target Distance: {config.target_distance_m:.2f} m (Sweet Spot: {config.target_distance_m - config.distance_deadband_m:.2f}m - {config.target_distance_m + config.distance_deadband_m:.2f}m)")
    print(f"  Forward Speeds:  {config.min_forward_speed:.2f} m/s min -> {config.max_forward_speed:.2f} m/s max")
    print(f"  Turn Speeds:     {config.min_turn_speed:.2f} m/s min -> {config.max_turn_speed:.2f} m/s max")
    print(f"  Safety Stop:     < {config.emergency_stop_dist_m:.2f} m")
    print(f"  Headless:        {config.headless}")
    print("=" * 65)
    print("Press Ctrl+C (or 'q' in preview window) to stop.\n")

    follower = Stage2DistanceFollower(config)
    if not follower.start():
        print("Error: Could not start Stage 2 Follower.")
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
                    fields["target"] = {"bearing": -telem.get("bearing_deg", 0.0), "dist": telem.get("dist_m")}
                else:
                    fields["target"] = None
                send(**fields)
                if config.headless:  # with a preview window the frame is sent from the preview code below
                    ret, frame, _ = follower.vision.camera.read()
                    if ret and frame is not None and hasattr(frame, "shape"):
                        send_frame(draw_stage2_overlay(frame.copy(), telem))

            if now - last_log_t >= 0.4:
                state = telem.get("state") or "SEARCHING"
                dist = telem.get("dist_m")
                dist_val = float(dist) if dist is not None else 0.0
                source = telem.get("dist_source") or "UNK"
                vl = float(telem.get("v_left", 0.0) or 0.0)
                vr = float(telem.get("v_right", 0.0) or 0.0)
                herr = float(telem.get("h_err", 0.0) or 0.0)
                fps = float(telem.get("fps", 0.0) or 0.0)
                tof_m = float(telem.get("tof_mid", 9.99) or 9.99)
                print(
                    f"[{state:^17}] Dist: {dist_val:4.2f}m [{source:^12}] | "
                    f"ToF_M: {tof_m:4.2f}m | Err: {herr:+5.2f} | Wheels: L={vl:+5.2f}, R={vr:+5.2f} m/s | FPS: {fps:4.1f}"
                )
                last_log_t = now

            if cv2 is not None and not config.headless:
                ret, frame, _ = follower.vision.camera.read()
                if ret and frame is not None and hasattr(frame, "shape"):
                    annotated = draw_stage2_overlay(frame.copy(), telem)
                    send_frame(annotated)  # also show it in the web dashboard
                    cv2.imshow("Stage 2 Human Follower", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\nUser requested stop via window.")
                        break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopping Stage 2 Follower...")

    finally:
        follower.stop()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Stage 2 Follower safely stopped. Motors halted.")


if __name__ == "__main__":
    main()

