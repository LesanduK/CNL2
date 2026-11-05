"""CNL2 dashboard backend.

Run from the repo root on the Pi:   python3 dashboard/server.py
Test on any laptop, no hardware:    python3 dashboard/server.py --sim
Then open http://<pi-address>:8000 in a browser on the same network.

One mode runs at a time (it owns the serial port, camera and ToF sensors):
  follow1/2/3 and nav are launched as subprocesses of your existing scripts,
  manual drives the ESP32 directly over serial using the documented protocol.
Mode scripts can push live data with dashboard/telemetry_client.py (optional).
"""
import argparse, asyncio, contextlib, json, math, os, re, signal, subprocess, sys, threading, time
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
        self.map = (12.0, 8.0)  # warehouse length, width in metres (used by --sim)
        self.route = {"a": None, "b": None}  # back-and-forth points chosen on the map
        self.shuttle = {"on": False}
        self.arrived = self.pose_seen = self.pose_warned = False
        self.pose_polls, self.last_poll = 0, 0.0


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
        try:
            p.send_signal(signal.SIGINT)  # lets the script send its own stop
        except (ValueError, OSError):  # Windows cannot send SIGINT to a subprocess
            p.terminate()
        try:
            await asyncio.wait_for(p.wait(), 2)
        except asyncio.TimeoutError:
            p.kill()  # ESP32 watchdog cuts the motors within 500 ms anyway
    if hub.manual:
        await asyncio.to_thread(hub.manual.close)
        hub.manual = None
    hub.mode, hub.state, hub.frame = "idle", {}, None
    hub.shuttle, hub.arrived, hub.pose_seen, hub.pose_polls, hub.pose_warned = {"on": False}, False, False, 0, False


# The navigation shell is read for two things while a back-and-forth run is active: a reply to
# "pose" (X=.. Y=..) and an arrival message. Either one tells the server the robot got there.
ARRIVED = re.compile(r"(goal|destination|target|waypoint)\s+(has\s+been\s+)?reached|reached\s+(the\s+)?(goal|destination|target)|\barrived\b|navigation\s+complete", re.I)


def read_pose(line):
    mx = re.search(r"\bx\s*[=:]\s*(-?\d+(?:\.\d+)?)", line, re.I)
    my = re.search(r"\by\s*[=:]\s*(-?\d+(?:\.\d+)?)", line, re.I)
    if not (mx and my):
        return False
    pose = hub.state.get("pose") or [0.0, 0.0, 0.0]
    hub.state["pose"] = [float(mx.group(1)), float(my.group(1)), pose[2]]
    hub.pose_seen = True
    return True


async def pump(p):
    async for raw in p.stdout:
        line = raw.decode(errors="replace").rstrip()
        if hub.shuttle.get("on"):
            arrived = bool(ARRIVED.search(line))
            hub.arrived = hub.arrived or arrived
            if read_pose(line) and not arrived:
                continue  # position replies are consumed instead of filling the log
        await logline(line)
    await p.wait()
    await logline(f"Process exited with code {p.returncode}")
    if hub.proc is p:
        hub.proc, hub.mode, hub.state = None, "idle", {}


_help = {}


def supports(base, flag):
    """True if the script lists the flag in its --help (stage 1 has no --target-dist)."""
    key = tuple(base)
    if key not in _help:
        try:
            r = subprocess.run(base + ["--help"], cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=30)
            _help[key] = r.stdout + r.stderr
        except Exception:
            _help[key] = ""
    return flag in _help[key]


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
            for flag, key in (("--target-dist", "target_dist"), ("--max-speed", "max_speed")):
                if await asyncio.to_thread(supports, MODES[m], flag):
                    cmd += [flag, str(hub.settings[key])]
        if args.mock:
            cmd.append("--mock")
        hub.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=ROOT, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})  # unbuffered so the log fills live
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


async def start_shuttle(pause):
    a, b = hub.route["a"], hub.route["b"]
    if hub.mode != "nav":
        return await logline("Start navigation first, then the back-and-forth run")
    if not (a and b):
        return await logline("Set both the start point and the end point first")
    pose = hub.state.get("pose") or a
    first = math.hypot(pose[0] - a[0], pose[1] - a[1])
    hub.shuttle = {"on": True, "target": "A", "phase": "drive", "trips": 0, "count": False,
                   "pause": max(0.0, min(60.0, pause)), "t0": time.time(), "until": 0.0,
                   "limit": 30 + max(first, math.hypot(b[0] - a[0], b[1] - a[1])) / 0.08}
    hub.arrived = False
    await goto(*a)
    await logline(f"Back-and-forth run started between A ({a[0]:.2f}, {a[1]:.2f}) and B ({b[0]:.2f}, {b[1]:.2f})")


async def stop_shuttle(reason="Back-and-forth run stopped"):
    if not hub.shuttle.get("on"):
        return
    hub.shuttle = {"on": False}
    hub.state.pop("goal", None)
    if hub.proc and hub.proc.stdin:
        hub.proc.stdin.write(b"stop\n")
        await hub.proc.stdin.drain()
    await logline(reason)


async def shuttle_loop():
    """Drives to A, then B, then A... Arrival = pose within 15 cm of the point, or the script saying so."""
    while True:
        await asyncio.sleep(0.2)
        sh, now = hub.shuttle, time.time()
        if not sh.get("on"):
            continue
        if hub.mode != "nav":
            hub.shuttle = {"on": False}
            continue
        if sh["phase"] == "pause":
            if now >= sh["until"]:
                sh["target"] = "B" if sh["target"] == "A" else "A"
                sh["phase"], sh["t0"], hub.arrived = "drive", now, False
                await goto(*hub.route[sh["target"].lower()])
            continue
        gx, gy = hub.route[sh["target"].lower()]
        pose = hub.state.get("pose")
        if not args.sim and hub.pose_polls >= 4 and not hub.pose_seen and not hub.pose_warned:
            hub.pose_warned = True
            await logline("Could not read the robot position from the navigation script; relying on its arrival message")
        if hub.arrived or (pose and math.hypot(pose[0] - gx, pose[1] - gy) <= 0.15):
            hub.arrived = False
            if sh["count"]:
                sh["trips"] += 1
            sh["count"], sh["phase"], sh["until"] = True, "pause", now + sh["pause"]
            await logline(f"Reached point {sh['target']}")
        elif now - sh["t0"] > sh["limit"]:
            await stop_shuttle(f"Stopped: the robot did not reach point {sh['target']} in time")
        elif not args.sim and hub.proc and hub.proc.stdin and now - hub.last_poll >= 1.0 and (hub.pose_seen or hub.pose_polls < 4):
            hub.last_poll = now
            hub.pose_polls += 0 if hub.pose_seen else 1
            hub.proc.stdin.write(b"pose\n")
            await hub.proc.stdin.drain()


async def simulate():
    """Fake robot so the UI can be tried without hardware (--sim).
    Pillars and crates are laid out for a 12 x 8 m floor and scaled to the chosen warehouse. Navigate steers
    around them, the follow modes chase a walking fake person (full follow also avoids obstacles), and
    nothing can drive through an obstacle or a wall."""
    x = y = th = 0.0
    base = [[-4.0, -2.5, 0.3], [4.0, -2.5, 0.3], [4.0, 2.5, 0.3], [-4.0, 2.5, 0.3],   # pillars: x, y, radius (m)
            [1.5, 0.5, 0.25], [1.7, 0.75, 0.2], [2.6, -0.8, 0.3], [-1.4, 1.3, 0.25],   # crates
            [0.0, -2.0, 0.35], [-2.5, -0.5, 0.3], [3.2, 1.2, 0.25]]
    obstacles = base

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def ray(a):  # gap to the nearest obstacle surface inside a cone, 2 m if clear
        best = 2.0
        for ox, oy, r in obstacles:
            d = math.hypot(ox - x, oy - y)
            if abs(wrap(math.atan2(oy - y, ox - x) - th - a)) < 0.35 + math.atan2(r, max(d, 0.01)):
                best = min(best, max(0.0, d - r))
        return best

    while True:
        await asyncio.sleep(0.05)
        t, m, v, w = time.time(), hub.mode, 0.0, 0.0
        L, Wd = hub.map
        obstacles = [[ox * L / 12, oy * Wd / 8, r] for ox, oy, r in base]
        if m == "manual" and t - hub.drive[2] < 0.4:
            l, r = hub.drive[:2]
            v, w = (l + r) / 2, (r - l) / BASE
        elif m == "nav" and "goal" in hub.state:
            gx, gy = hub.state["goal"]
            dg = math.hypot(gx - x, gy - y)
            if dg < 0.08:
                hub.state.pop("goal")
            else:
                # pull toward the goal, push away (with a sideways nudge, so head-on cases slide past) from obstacles
                fx, fy, nearest = (gx - x) / dg, (gy - y) / dg, 9.0
                for ox, oy, r in obstacles:
                    d = math.hypot(x - ox, y - oy)
                    nearest = min(nearest, d - r)
                    if d - r < 0.9:
                        ux, uy = (x - ox) / (d or 1.0), (y - oy) / (d or 1.0)
                        k = 1.2 * (1 / max(d - r, 0.08) - 1 / 0.9)
                        fx, fy = fx + k * (ux - 0.7 * uy), fy + k * (uy + 0.7 * ux)
                err = wrap(math.atan2(fy, fx) - th)
                w, v = 2.5 * err, 0.3 * max(0.0, math.cos(err)) * min(1.0, max(0.25, nearest / 0.6))
        elif m.startswith("follow"):
            px, py = 0.32 * L * math.cos(t * 0.12), 0.32 * Wd * math.sin(t * 0.18)  # the fake person
            dist = math.hypot(px - x, py - y)
            err = wrap(math.atan2(py - y, px - x) - th)  # positive = person is to the left
            hub.state["target"] = {"bearing": math.degrees(err), "dist": None if m == "follow1" else dist}
            w = max(-1.5, min(1.5, 2.0 * err)) if abs(err) > 0.05 else 0.0
            td = hub.settings["target_dist"]
            if m != "follow1" and abs(err) < 0.7 and dist > td + 0.1:
                v = min(hub.settings["max_speed"], 0.8 * (dist - td))
            if m == "follow3":
                for ox, oy, r in obstacles:
                    d, eb = math.hypot(ox - x, oy - y) - r, wrap(math.atan2(oy - y, ox - x) - th)
                    if d < 0.8 and abs(eb) < 1.2:
                        w -= (1.0 if eb >= 0 else -1.0) * 1.5 * (0.8 - d) / 0.8  # steer away
                        v = 0.0 if d < 0.3 else min(v, 0.15)
        x, y, th = x + v * math.cos(th) * 0.05, y + v * math.sin(th) * 0.05, th + w * 0.05
        x, y = max(-L / 2 + 0.2, min(L / 2 - 0.2, x)), max(-Wd / 2 + 0.2, min(Wd / 2 - 0.2, y))  # stay inside the walls
        for ox, oy, r in obstacles:  # robot body is about 12 cm in radius: it cannot enter an obstacle
            d = math.hypot(x - ox, y - oy)
            if d < r + 0.12:
                x, y = ox + (x - ox) / (d or 1.0) * (r + 0.12), oy + (y - oy) / (d or 1.0) * (r + 0.12)
        if m != "idle":
            hub.state.update(pose=[x, y, th], rpm=[(v - w * BASE / 2) * 60 / (2 * math.pi * WHEEL_R),
                                                   (v + w * BASE / 2) * 60 / (2 * math.pi * WHEEL_R)],
                             tof=[ray(math.pi / 4), ray(0.0), ray(-math.pi / 4)], obstacles=obstacles)


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
            await bcast({"t": "state", "mode": hub.mode, "s": hub.state, "route": hub.route,
                         "shuttle": {k: hub.shuttle.get(k) for k in ("on", "target", "phase", "trips")}})
        if hub.mode != "idle" and args.stop_on_disconnect and time.time() - hub.hb > 2:
            await logline("Browser connection lost, stopping the robot")
            await stop_all()


@contextlib.asynccontextmanager
async def life(app):
    await asyncio.get_running_loop().create_datagram_endpoint(UDP, local_addr=("127.0.0.1", UDP_PORT))
    tasks = [asyncio.create_task(ticker()), asyncio.create_task(shuttle_loop())]
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
    await w.send_text(json.dumps({"t": "hello", "log": list(hub.log), "settings": hub.settings, "route": hub.route}))
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
                if not hub.shuttle.get("on"):
                    await goto(float(m["x"]), float(m["y"]))
            elif c == "route":
                for k in ("a", "b"):
                    if k in m:
                        hub.route[k] = [float(m[k][0]), float(m[k][1])] if m[k] else None
            elif c == "shuttle":
                if m.get("on"):
                    await start_shuttle(float(m.get("pause", 2)))
                else:
                    await stop_shuttle()
            elif c == "drive":
                l, r = (max(-MAX_DRIVE, min(MAX_DRIVE, float(m[k]))) for k in ("l", "r"))
                if hub.manual:
                    hub.manual.drive(l, r)
                hub.drive = (l, r, time.time())
            elif c == "map":
                hub.map = tuple(max(2.0, min(200.0, float(m[k]))) for k in ("length", "width"))
            elif c == "settings":
                for k in hub.settings:
                    if k in m:
                        hub.settings[k] = float(m[k])
    except (WebSocketDisconnect, ValueError, KeyError, IndexError, TypeError):
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
