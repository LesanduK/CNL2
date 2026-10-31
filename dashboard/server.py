"""CNL2 dashboard backend.

Run from the repo root on the Pi:   python3 dashboard/server.py
Test on any laptop, no hardware:    python3 dashboard/server.py --sim
Then open http://<pi-address>:8000 in a browser on the same network.

One mode runs at a time (it owns the serial port, camera and ToF sensors):
  follow1/2/3 and nav are launched as subprocesses of your existing scripts,
  manual drives the ESP32 directly over serial using the documented protocol.
Mode scripts can push live data with dashboard/telemetry_client.py (optional).
"""
import argparse, asyncio, contextlib, json, math, signal, sys, threading, time
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
SERIAL_PORT, BAUD = "/dev/serial0", 115200
WHEEL_R, BASE, TPR = 0.0215, 0.134, 720.0   # from Config.h / config.py
UDP_PORT = 9999
MODES = {
    "follow1": [PY, "human_follower/main.py", "--headless"],
    "follow2": [PY, "human_follower/main_stage2.py", "--headless"],
    "follow3": [PY, "human_follower/main_stage3.py", "--headless"],
    "nav": [PY, "-m", "autonomous_navigation.main"],
}
MAX_DRIVE = 0.5  # m/s clamp for manual commands
args = None


class Hub:
    def __init__(self):
        self.clients, self.state, self.log = set(), {}, deque(maxlen=200)
        self.mode, self.proc, self.manual, self.frame = "idle", None, None, None
        self.hb, self.drive = time.time(), (0.0, 0.0, 0.0)
        self.settings = {"target_dist": 1.2, "max_speed": 0.35}


hub = Hub()


async def bcast(obj):
    msg, dead = json.dumps(obj), []
    for w in list(hub.clients):
        try:
            await w.send_text(msg)
        except Exception:
            dead.append(w)
    for w in dead:
        hub.clients.discard(w)


async def logline(text):
    hub.log.append(text)
    await bcast({"t": "log", "line": text})


class Manual:
    """Direct serial drive: sends 'vl,vr\\n' at 20 Hz, reads ENC telemetry for odometry."""

    def __init__(self):
        self.l = self.r = 0.0
        self.last, self.run, self.ser = 0.0, True, None
        self.x = self.y = self.th = 0.0
        self.prev = None

    def open(self):
        try:
            import serial
            self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.02)
        except Exception as e:  # missing pyserial, wrong port, port busy
            return str(e)
        threading.Thread(target=self.loop, daemon=True).start()

    def drive(self, l, r):
        self.l, self.r, self.last = l, r, time.time()

    def loop(self):
        buf = b""
        while self.run:
            if time.time() - self.last > 0.4:  # joystick released or browser silent
                self.l = self.r = 0.0
            self.ser.write(f"{self.l:.3f},{self.r:.3f}\n".encode())
            buf += self.ser.read(self.ser.in_waiting or 1)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                self.parse(ln.decode(errors="ignore").strip())
            time.sleep(0.05)

    def parse(self, ln):
        try:
            _, lt, rt, lrpm, rrpm = ln.split(",")
            lt, rt = int(lt), int(rt)
        except ValueError:
            return
        if self.prev:
            k = 2 * math.pi * WHEEL_R / TPR
            dl, dr = (lt - self.prev[0]) * k, (rt - self.prev[1]) * k
            d, dth = (dl + dr) / 2, (dr - dl) / BASE
            self.x += d * math.cos(self.th + dth / 2)
            self.y += d * math.sin(self.th + dth / 2)
            self.th += dth
        self.prev = (lt, rt)
        hub.state.update(pose=[self.x, self.y, self.th], rpm=[float(lrpm), float(rrpm)])

    def close(self):
        self.run = False
        time.sleep(0.1)
        with contextlib.suppress(Exception):
            self.ser.write(b"0.000,0.000\n")
            self.ser.close()


async def stop_all():
    p, hub.proc = hub.proc, None
    if p and p.returncode is None:
        p.send_signal(signal.SIGINT)  # lets the script send its own stop
        try:
            await asyncio.wait_for(p.wait(), 2)
        except asyncio.TimeoutError:
            p.kill()  # ESP32 watchdog cuts the motors within 500 ms anyway
    if hub.manual:
        await asyncio.to_thread(hub.manual.close)
        hub.manual = None
    hub.mode, hub.state, hub.frame = "idle", {}, None


async def pump(p):
    async for raw in p.stdout:
        await logline(raw.decode(errors="replace").rstrip())
    await p.wait()
    await logline(f"Process exited with code {p.returncode}")
    if hub.proc is p:
        hub.proc, hub.mode, hub.state = None, "idle", {}


async def set_mode(m):
    await stop_all()
    if m == "idle":
        return
    hub.mode = m
    if args.sim:
        await logline(f"Simulator: {m} started")
    elif m == "manual":
        hub.manual = Manual()
        err = await asyncio.to_thread(hub.manual.open)
        if err:
            hub.manual = None
            hub.mode = "idle"
            await logline(f"Cannot open {SERIAL_PORT}: {err}")
    else:
        cmd = list(MODES[m])
        if m.startswith("follow"):
            cmd += ["--target-dist", str(hub.settings["target_dist"]), "--max-speed", str(hub.settings["max_speed"])]
        if args.mock:
            cmd.append("--mock")
        hub.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=ROOT, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        asyncio.create_task(pump(hub.proc))
        await logline("Started: " + " ".join(cmd[1:]))


async def goto(x, y):
    if hub.mode != "nav":
        return
    hub.state["goal"] = [x, y]
    if hub.proc and hub.proc.stdin:
        hub.proc.stdin.write(f"goto {x:.2f} {y:.2f}\n".encode())
        await hub.proc.stdin.drain()
    await logline(f"Goal set to ({x:.2f}, {y:.2f}) m")


async def simulate():
    """Fake robot so the UI can be tried without hardware (--sim)."""
    x = y = th = 0.0
    while True:
        await asyncio.sleep(0.05)
        t, m, v, w = time.time(), hub.mode, 0.0, 0.0
        if m == "manual" and t - hub.drive[2] < 0.4:
            l, r = hub.drive[:2]
            v, w = (l + r) / 2, (r - l) / BASE
        elif m == "nav" and "goal" in hub.state:
            gx, gy = hub.state["goal"]
            err = (math.atan2(gy - y, gx - x) - th + math.pi) % (2 * math.pi) - math.pi
            if math.hypot(gx - x, gy - y) < 0.08:
                hub.state.pop("goal")
            else:
                w, v = 2 * err, 0.3 * max(0.0, math.cos(err))
                hub.state["path"] = [[x, y], [gx, gy]]
        elif m.startswith("follow"):
            hub.state["target"] = {"bearing": 20 * math.sin(t / 2), "dist": 1.2 + 0.3 * math.sin(t / 3)}
        x, y, th = x + v * math.cos(th) * 0.05, y + v * math.sin(th) * 0.05, th + w * 0.05
        if m != "idle":
            hub.state.update(pose=[x, y, th], rpm=[v * 60, v * 60],
                             tof=[1.5 + 0.2 * math.sin(t), 1.2 + 0.3 * math.sin(t / 2), 1.8],
                             obstacles=[[1.5, 0.5], [1.6, 0.6], [2.5, -0.8]])


class UDP(asyncio.DatagramProtocol):
    """Receives optional JSON telemetry (or b'F'+jpeg frames) from the mode scripts."""

    def datagram_received(self, data, addr):
        if data[:1] == b"F":
            hub.frame = data[1:]
        else:
            with contextlib.suppress(ValueError):
                hub.state.update(json.loads(data))


async def ticker():
    while True:
        await asyncio.sleep(0.1)
        if hub.clients:
            await bcast({"t": "state", "mode": hub.mode, "s": hub.state})
        if hub.mode != "idle" and args.stop_on_disconnect and time.time() - hub.hb > 2:
            await logline("Browser connection lost, stopping the robot")
            await stop_all()


@contextlib.asynccontextmanager
async def life(app):
    await asyncio.get_running_loop().create_datagram_endpoint(UDP, local_addr=("127.0.0.1", UDP_PORT))
    tasks = [asyncio.create_task(ticker())]
    if args.sim:
        tasks.append(asyncio.create_task(simulate()))
    yield
    await stop_all()
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=life)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/video")
async def video():
    async def gen():
        last = None
        while True:
            f = hub.frame
            if f is not None and f is not last:
                last = f
                yield b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f) + f + b"\r\n"
            await asyncio.sleep(0.08)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.websocket("/ws")
async def ws(w: WebSocket):
    await w.accept()
    hub.clients.add(w)
    hub.hb = time.time()
    await w.send_text(json.dumps({"t": "hello", "log": list(hub.log), "settings": hub.settings}))
    try:
        while True:
            m = json.loads(await w.receive_text())
            hub.hb, c = time.time(), m.get("cmd")
            if c == "mode" and (m["mode"] in MODES or m["mode"] in ("idle", "manual")):
                await set_mode(m["mode"])
            elif c == "estop":
                await stop_all()
                await logline("Emergency stop pressed")
            elif c == "goto":
                await goto(float(m["x"]), float(m["y"]))
            elif c == "drive":
                l, r = (max(-MAX_DRIVE, min(MAX_DRIVE, float(m[k]))) for k in ("l", "r"))
                if hub.manual:
                    hub.manual.drive(l, r)
                hub.drive = (l, r, time.time())
            elif c == "settings":
                for k in hub.settings:
                    if k in m:
                        hub.settings[k] = float(m[k])
    except (WebSocketDisconnect, ValueError):
        pass
    finally:
        hub.clients.discard(w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sim", action="store_true", help="fake robot, no hardware or scripts")
    ap.add_argument("--mock", action="store_true", help="pass --mock to the mode scripts")
    ap.add_argument("--no-disconnect-stop", dest="stop_on_disconnect", action="store_false",
                    help="keep running if the browser disconnects (default: stop)")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
