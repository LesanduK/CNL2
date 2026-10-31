#!/usr/bin/env python3
"""
rpi_speed_controller.py
-----------------------
Interactive speed controller for Raspberry Pi communicating with ESP32-S3.

Features:
- Continuously streams speed commands at 20 Hz to satisfy the ESP32 500 ms watchdog.
- Allows user to change speeds on the fly:
    - '0.2'        -> drives both wheels at 0.2 m/s
    - '0.15, 0.25' -> left wheel at 0.15 m/s, right wheel at 0.25 m/s
    - 'stop' / 's' -> stops both wheels
    - 'enc'        -> prints live encoder telemetry
    - 'q'          -> safe stop and exit
- Reads and displays encoder telemetry (ticks + RPM) received from the ESP32.
"""

import sys
import time
import threading
import argparse
import serial

DEFAULT_PORT = "/dev/serial0"
DEFAULT_BAUD = 115200
TX_RATE_HZ = 20.0  # Send at 20 Hz (every 50 ms)
MAX_SPEED_MPS = 1.0  # Max speed clamp for safety (m/s)

class RobotComm:
    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = False

        # Speed targets (m/s)
        self.target_left = 0.0
        self.target_right = 0.0
        self.push_until = 0.0
        self.speed_lock = threading.Lock()

        # Telemetry from ESP32
        self.left_ticks = 0
        self.right_ticks = 0
        self.left_rpm = 0.0
        self.right_rpm = 0.0
        self.last_telemetry_time = 0.0
        self.telemetry_lock = threading.Lock()

        self.tx_thread = None
        self.rx_thread = None

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.1,
                write_timeout=0.5
            )
            print(f"Connected to ESP32 on {self.port} at {self.baud} baud.")
            time.sleep(1.0)  # Allow ESP32 serial link to stabilize
            return True
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")
            print("Tip: If running on a PC for testing, check /dev/ttyACM0 or /dev/ttyUSB0.")
            return False

    def start(self):
        self.running = True
        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.tx_thread.start()
        self.rx_thread.start()

    def set_speed(self, left, right):
        """Set new target speeds in m/s (clamped to +-MAX_SPEED_MPS)."""
        left = max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, float(left)))
        right = max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, float(right)))
        with self.speed_lock:
            # Trigger a 180 ms startup kick if transitioning from 0 to break static friction
            was_stopped = (abs(self.target_left) < 0.01 and abs(self.target_right) < 0.01)
            is_moving = (abs(left) > 0.01 or abs(right) > 0.01)
            if was_stopped and is_moving:
                self.push_until = time.time() + 0.18

            self.target_left = left
            self.target_right = right
        print(f"-> Speed updated: Left = {left:+.2f} m/s, Right = {right:+.2f} m/s")

    def stop_motors(self):
        """Instantly set speeds to 0."""
        with self.speed_lock:
            self.push_until = 0.0
        self.set_speed(0.0, 0.0)

    def get_telemetry(self):
        with self.telemetry_lock:
            return {
                "left_ticks": self.left_ticks,
                "right_ticks": self.right_ticks,
                "left_rpm": self.left_rpm,
                "right_rpm": self.right_rpm,
                "fresh": (time.time() - self.last_telemetry_time) < 1.0
            }

    def _tx_loop(self):
        """Continuously transmits the current speed at 20 Hz to keep the watchdog alive."""
        interval = 1.0 / TX_RATE_HZ
        while self.running:
            start_t = time.time()
            with self.speed_lock:
                l = self.target_left
                r = self.target_right
                push_active = time.time() < self.push_until

            # Provide initial push at 0.80 m/s (demands 255 PWM max power) then reduce to intended speed
            if push_active:
                out_l = (0.80 if l > 0 else -0.80) if abs(l) > 0.01 else 0.0
                out_r = (0.80 if r > 0 else -0.80) if abs(r) > 0.01 else 0.0
            else:
                out_l = l
                out_r = r

            msg = f"{out_l:.2f},{out_r:.2f}\n".encode("ascii")
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(msg)
                    self.ser.flush()
            except (serial.SerialException, OSError):
                pass

            elapsed = time.time() - start_t
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _rx_loop(self):
        """Continuously reads incoming telemetry lines from the ESP32."""
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    time.sleep(0.05)
                    continue

                line = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                # Expected format: ENC,<leftTicks>,<rightTicks>[,<leftRPM>,<rightRPM>]
                parts = [p.strip() for p in line.split(",")]
                if parts and parts[0] == "ENC":
                    with self.telemetry_lock:
                        if len(parts) >= 3:
                            self.left_ticks = int(parts[1])
                            self.right_ticks = int(parts[2])
                        if len(parts) >= 5:
                            self.left_rpm = float(parts[3])
                            self.right_rpm = float(parts[4])
                        self.last_telemetry_time = time.time()

            except (ValueError, serial.SerialException):
                time.sleep(0.01)

    def close(self):
        """Gracefully stop motors and close connection."""
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                # Send 0,0 multiple times to ensure the ESP32 stops
                for _ in range(3):
                    self.ser.write(b"0.00,0.00\n")
                    self.ser.flush()
                    time.sleep(0.05)
                self.ser.close()
            except Exception:
                pass
        print("Serial connection closed. Robot stopped.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Interactive Speed Controller")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (default: /dev/serial0)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default: 115200)")
    args = parser.parse_args()

    robot = RobotComm(port=args.port, baud=args.baud)
    if not robot.connect():
        sys.exit(1)

    robot.start()

    print("\n" + "=" * 60)
    print("ROBOT SPEED CONTROLLER (Raspberry Pi -> ESP32-S3)")
    print("=" * 60)
    print("Commands:")
    print("  <speed>             Single speed for both wheels in m/s (e.g. '0.2')")
    print("  <left>, <right>     Different left and right speeds (e.g. '0.2, 0.3')")
    print("  stop  (or 's')      Stop both motors immediately")
    print("  enc                 Print latest encoder ticks & RPM telemetry")
    print("  q     (or 'quit')   Stop and exit")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                cmd = input("Speed (m/s) > ").strip().lower()
            except EOFError:
                break

            if not cmd:
                continue

            if cmd in ("q", "quit", "exit"):
                break

            if cmd in ("stop", "s", "brake", "b"):
                robot.stop_motors()
                continue

            if cmd == "enc":
                t = robot.get_telemetry()
                status = "FRESH" if t["fresh"] else "STALE/WAITING"
                print(f"[Telemetry ({status})]")
                print(f"  Left  Ticks: {t['left_ticks']:8d} | RPM: {t['left_rpm']:+6.1f}")
                print(f"  Right Ticks: {t['right_ticks']:8d} | RPM: {t['right_rpm']:+6.1f}")
                continue

            # Parse numeric inputs
            try:
                if "," in cmd:
                    parts = cmd.split(",")
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                else:
                    speed = float(cmd)
                    left = speed
                    right = speed

                robot.set_speed(left, right)

            except ValueError:
                print("Invalid input. Enter a number (e.g. '0.20') or two comma-separated numbers (e.g. '0.15, 0.25').")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        robot.close()


if __name__ == "__main__":
    main()

