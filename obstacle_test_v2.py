#!/usr/bin/env python3
"""
obstacle_test_v2.py - bench test: object TRACKING + cross-sensor blind-spot alerts.

SAFE SETUP (nothing here can move the drone):
  * Pi powered from its own USB-C supply. Motors/ESCs unpowered, props OFF,
    flight controller not connected. The script never sends any command.
  * Laptop <-> Pi over SSH (ssh <user>@<pi>.local). Run inside `tmux` so the
    session survives a dropped connection. Alerts ring your laptop's terminal bell.

Sensors (same as v1): forward AI Camera, downward ArduCam (QR), 3 lidars L/R/F.

What it does
  * Tracks camera detections frame-to-frame (persistent IDs) and estimates
    time-to-collision (TTC) from box growth.
  * Cross-checks the sensors and tells you when one covers the other's blind spot:
      CONFIRMED   camera + forward lidar agree                    (high confidence)
      CAM-ONLY    camera sees a close object, forward lidar does not
                  -> object is off the lidar's narrow beam
      LIDAR-ONLY  forward lidar sees something, camera has no detection
                  -> unclassified obstacle (wall/pole/pillar the model can't name)
      SIDE        side lidar close; the forward camera cannot see the sides
      HANDOFF     a tracked object leaves the camera view at the left/right edge and
                  the matching side lidar then reacts -> lidar keeps tracking it
      LOST-CLOSE  camera lost a close object; reports what the forward lidar says
      SENSOR      a sensor stopped updating; says what is now uncovered

  python3 obstacle_test_v2.py --sim --duration 26        # logic check, no hardware
  python3 obstacle_test_v2.py --lidar-ports /dev/ttyUSB0,/dev/ttyUSB1,/dev/ttyUSB2 \
        --down-index 1 --log run1.csv                    # order: left,right,forward

Velocity convention: vx +forward, vy +right (body frame, m/s). Print-only.
"""
import argparse
import csv
import math
import random
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

STATES = ["CLEAR", "SLOW", "STOP"]
INF = math.inf
EVENTS = deque(maxlen=500)  # tracker thread -> main thread: ("lost", Track, t, (W, H))

COVER = {
    "F": "FORWARD LIDAR lost: no forward range. Only the camera covers ahead (classified "
         "objects, no distance). Treat as STOP.",
    "L": "LEFT LIDAR lost: left side is blind; the forward camera does not cover the sides.",
    "R": "RIGHT LIDAR lost: right side is blind; the forward camera does not cover the sides.",
    "CAM": "FORWARD CAMERA lost: nothing is classified or tracked; forward lidar sees only a "
           "narrow beam (misses objects above/below/beside it).",
    "QR": "DOWNWARD CAMERA lost: no QR / red-zone / ground view.",
}


# ----------------------------------------------------------------- data types
@dataclass
class Det:
    label: str
    conf: float
    x: float
    y: float
    w: float
    h: float  # pixels, forward-camera frame


@dataclass
class Track:
    id: int
    label: str
    x: float
    y: float
    w: float
    h: float
    conf: float
    t_first: float
    t_last: float
    hits: int = 1
    hist: deque = field(default_factory=lambda: deque(maxlen=30))  # (t, box height)


@dataclass
class FrameDets:
    dets: list
    tracks: list  # confirmed tracks seen in the last ~0.2 s
    size: tuple   # (W, H)


@dataclass
class QR:
    data: str
    dx: float
    dy: float
    size: float


class Latest:
    def __init__(self):
        self._lock = threading.Lock()
        self._v = None
        self._t = None

    def set(self, v):
        with self._lock:
            self._v = v
            self._t = time.monotonic()

    def get(self):
        with self._lock:
            return self._v, (INF if self._t is None else time.monotonic() - self._t)


class Worker(threading.Thread):
    needs_ready = False

    def __init__(self, name, out):
        super().__init__(name=name, daemon=True)
        self.out = out
        self.stop_evt = threading.Event()
        self.ready = threading.Event()
        self.error = None

    def run(self):
        try:
            self.loop()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.ready.set()

    def stop(self):
        self.stop_evt.set()


SHARED = {}


# -------------------------------------------------------------------- tracker
class Tracker:
    """Greedy IoU tracker (same label). Emits ('lost', track, t, size) events."""

    def __init__(self, events, iou_min=0.15, max_age=0.6, min_hits=3):
        self.tracks, self.next_id, self.events = {}, 1, events
        self.iou_min, self.max_age, self.min_hits = iou_min, max_age, min_hits

    @staticmethod
    def iou(a, b):
        ax2, ay2, bx2, by2 = a.x + a.w, a.y + a.h, b.x + b.w, b.y + b.h
        iw = max(0.0, min(ax2, bx2) - max(a.x, b.x))
        ih = max(0.0, min(ay2, by2) - max(a.y, b.y))
        inter = iw * ih
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    def update(self, dets, t, size):
        pairs = []
        for tr in self.tracks.values():
            for di, d in enumerate(dets):
                if d.label == tr.label:
                    i = self.iou(tr, d)
                    if i >= self.iou_min:
                        pairs.append((i, tr.id, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            tr, d = self.tracks[tid], dets[di]
            tr.x, tr.y, tr.w, tr.h, tr.conf = d.x, d.y, d.w, d.h, d.conf
            tr.t_last, tr.hits = t, tr.hits + 1
            tr.hist.append((t, d.h))
        for di, d in enumerate(dets):
            if di not in used_d:
                tr = Track(self.next_id, d.label, d.x, d.y, d.w, d.h, d.conf, t, t)
                tr.hist.append((t, d.h))
                self.tracks[tr.id] = tr
                self.next_id += 1
        for tid in [k for k, v in self.tracks.items() if t - v.t_last > self.max_age]:
            tr = self.tracks.pop(tid)
            if tr.hits >= self.min_hits:
                self.events.append(("lost", tr, t, size))
        return [v for v in self.tracks.values()
                if v.hits >= self.min_hits and t - v.t_last <= 0.2]


def ttc_of(tr):
    """Time-to-collision from box growth: TTC = h / (dh/dt). INF if not closing."""
    pts = list(tr.hist)
    if len(pts) < 5:
        return INF
    t1 = pts[-1][0]
    pts = [p for p in pts if p[0] >= t1 - 1.0]
    if len(pts) < 5 or pts[-1][0] - pts[0][0] < 0.4:
        return INF
    n = len(pts)
    mt = sum(p[0] for p in pts) / n
    mv = sum(p[1] for p in pts) / n
    den = sum((p[0] - mt) ** 2 for p in pts)
    if den <= 0:
        return INF
    slope = sum((p[0] - mt) * (p[1] - mv) for p in pts) / den
    if slope <= 0 or slope / tr.h < 0.05:  # not growing at least 5 %/s
        return INF
    return tr.h / slope


# --------------------------------------------------------------------- lidar
class TFMini(Worker):
    def __init__(self, name, port, a, out):
        super().__init__(name, out)
        self.port, self.a = port, a

    def loop(self):
        import serial
        ser = serial.Serial(self.port, self.a.lidar_baud, timeout=0.05)
        buf = bytearray()
        while not self.stop_evt.is_set():
            buf += ser.read(ser.in_waiting or 1)
            while len(buf) >= 9:
                if buf[0] == 0x59 and buf[1] == 0x59 and (sum(buf[:8]) & 0xFF) == buf[8]:
                    d = (buf[2] | buf[3] << 8) / 100.0
                    s = buf[4] | buf[5] << 8
                    ok = s >= self.a.lidar_min_strength and 0.05 < d <= self.a.lidar_max_m
                    self.out.set(d if ok else None)
                    del buf[:9]
                else:
                    del buf[0]
        ser.close()


# ---------------------------------------------------------- forward AI camera
class IMX500Worker(Worker):
    needs_ready = True

    def __init__(self, a, out):
        super().__init__("CAM", out)
        self.a = a

    def loop(self):
        import numpy as np
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics

        a = self.a
        imx500 = IMX500(a.model)
        intr = imx500.network_intrinsics
        if not intr:
            intr = NetworkIntrinsics()
            intr.task = "object detection"
        labels = intr.labels or []
        SHARED["ai_cam"] = imx500.camera_num
        picam2 = Picamera2(imx500.camera_num)
        cfg = picam2.create_preview_configuration(
            controls={"FrameRate": intr.inference_rate}, buffer_count=12)
        picam2.start(cfg, show_preview=False)
        size = tuple(picam2.camera_configuration()["main"]["size"])
        tracker = Tracker(EVENTS)
        self.ready.set()

        while not self.stop_evt.is_set():
            md = picam2.capture_metadata()
            outs = imx500.get_outputs(md, add_batch=True)
            if outs is None:
                continue
            boxes, scores, classes = outs[0][0], outs[1][0], outs[2][0]
            in_w, in_h = imx500.get_input_size()
            if getattr(intr, "bbox_normalization", False):
                boxes = boxes / in_h
            if getattr(intr, "bbox_order", "yx") == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = zip(*np.array_split(boxes, 4, axis=1))
            dets = []
            for box, score, cls in zip(boxes, scores, classes):
                if score < a.det_thr:
                    continue
                x, y, w, h = imx500.convert_inference_coords(box, md, picam2)
                ci = int(cls)
                lab = labels[ci] if 0 <= ci < len(labels) else str(ci)
                dets.append(Det(lab, float(score), float(x), float(y), float(w), float(h)))
            tracks = tracker.update(dets, time.monotonic(), size)
            self.out.set(FrameDets(dets, tracks, size))


# ------------------------------------------------------------ downward QR cam
class QRWorker(Worker):
    needs_ready = True

    def __init__(self, a, out):
        super().__init__("QR", out)
        self.a = a

    def loop(self):
        import cv2
        import numpy as np

        a = self.a
        w, h = (int(v) for v in a.down_res.lower().split("x"))
        if a.down_backend == "picamera2":
            from picamera2 import Picamera2
            if a.down_index == SHARED.get("ai_cam"):
                raise RuntimeError("--down-index is the AI camera; run --list-cameras")
            cam = Picamera2(a.down_index)
            cam.configure(cam.create_video_configuration(
                main={"size": (w, h), "format": "RGB888"}))
            cam.start()
            grab = lambda: cam.capture_array("main")
        else:
            cap = cv2.VideoCapture(a.down_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            grab = lambda: cap.read()[1]
        self.ready.set()

        det = (cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco")
               else cv2.QRCodeDetector())
        while not self.stop_evt.is_set():
            frame = grab()
            if frame is None:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            ok, infos, pts, _ = det.detectAndDecodeMulti(gray)
            res = []
            if ok and pts is not None:
                H, W = gray.shape
                for text, p in zip(infos, pts):
                    if not text:
                        continue
                    c = p.mean(axis=0)
                    side = float(np.mean([np.linalg.norm(p[i] - p[(i + 1) % 4])
                                          for i in range(4)]))
                    res.append(QR(text, float(c[0] - W / 2), float(c[1] - H / 2), side))
            self.out.set(res)


# ------------------------------------------------------------------ simulator
class SimWorld:
    """24 s loop that exercises every alert type:
       0-8 pillar seen by camera+lidar | 8-12 wall only lidar sees | 12-16 box only camera
       sees | 16-19 person drifts out of view on the left | 19-21 left lidar reacts | rest clear
    """

    def __init__(self):
        self.t0 = time.monotonic()

    def t(self):
        return time.monotonic() - self.t0

    def sample(self):
        t = self.t()
        p = t % 24.0
        off = 0.55 * math.sin(0.7 * t)
        left, right, front, obj = 1.75 + off, 1.75 - off, 6.0, None
        if p < 8.0:
            front = max(0.35, 7.0 - 0.85 * p)
            if front < 5.0:
                obj = ("sim-pillar", 0.5, min(0.95, 0.9 / front))
        elif p < 12.0:
            front = 1.8
        elif p < 16.0:
            dist = 4.0 - 0.75 * (p - 12.0)
            front = None  # lidar: no valid return
            obj = ("sim-box", 0.5, min(0.95, 0.9 / dist))
        elif p < 19.0:
            obj = ("sim-person", 0.5 - 0.48 * (p - 16.0) / 3.0, 0.40)
        elif p < 21.0:
            left = 0.5

        def n(v):
            return None if v is None else v + random.gauss(0, 0.01)
        return dict(L=n(left), R=n(right), F=n(front), obj=obj)


class SimLidar(Worker):
    def __init__(self, name, out, world):
        super().__init__(name, out)
        self.world = world

    def loop(self):
        while not self.stop_evt.is_set():
            self.out.set(self.world.sample()[self.name])
            time.sleep(0.02)


class SimCam(Worker):
    def __init__(self, out, world):
        super().__init__("CAM", out)
        self.world = world

    def loop(self):
        W, H = 640, 480
        tracker = Tracker(EVENTS)
        while not self.stop_evt.is_set():
            s = self.world.sample()
            dets = []
            if s["obj"]:
                lab, cx, hf = s["obj"]
                h = hf * H
                w = 0.5 * h
                dets.append(Det(lab, 0.9, cx * W - w / 2, H - h, w, h))
            tracks = tracker.update(dets, time.monotonic(), (W, H))
            self.out.set(FrameDets(dets, tracks, (W, H)))
            time.sleep(0.05)


class SimQR(Worker):
    def __init__(self, out, world):
        super().__init__("QR", out)
        self.world = world

    def loop(self):
        while not self.stop_evt.is_set():
            t = self.world.t()
            self.out.set([QR("SIM-TARGET-A", 40 * math.sin(t), -25.0, 180.0)]
                         if (t % 6.0) < 2.0 else [])
            time.sleep(0.2)


# --------------------------------------------------------------------- fusion
class Hysteresis:
    def __init__(self, n_up=2, n_down=6):
        self.level, self.n_up, self.n_down = 0, n_up, n_down
        self.up = self.down = 0

    def update(self, raw, immediate=False):
        if raw > self.level:
            self.down = 0
            self.up += 1
            if immediate or self.up >= self.n_up:
                self.level, self.up = raw, 0
        elif raw < self.level:
            self.up = 0
            self.down += 1
            if self.down >= self.n_down:
                self.level, self.down = self.level - 1, 0
        else:
            self.up = self.down = 0
        return self.level


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fmt(x):
    return " -- " if x is None else ("  inf" if x == INF else f"{x:5.2f}")


def fmt_ttc(x):
    return "--" if x == INF else f"{x:.1f}s"


def path_tracks(a, fd):
    """Tracks whose box overlaps the central 'in my path' strip -> (track, box_h_frac, ttc)."""
    W, H = fd.size
    lo, hi = W * (0.5 - a.corridor_frac / 2), W * (0.5 + a.corridor_frac / 2)
    return [(t, t.h / H, ttc_of(t)) for t in fd.tracks if t.x < hi and t.x + t.w > lo]


def fuse(a, L, R, F, pt, cam_ok, hyst):
    (l, la), (r, ra), (f, fa) = L, R, F
    why, raw, immediate = [], 0, False
    stale = [n for n, age in (("L", la), ("R", ra), ("F", fa)) if age > a.stale_s]
    if stale:
        raw, immediate = 2, True
        why.append("LIDAR STALE " + ",".join(stale))
    lv = INF if l is None else l
    rv = INF if r is None else r
    fv = INF if f is None else f
    if fv <= a.stop_m:
        raw = 2
        immediate = immediate or fv <= 0.6 * a.stop_m
        why.append(f"F<={a.stop_m}m")
    elif fv <= a.slow_m:
        raw = max(raw, 1)
        why.append("F slow zone")
    side = min(lv, rv)
    if side <= a.side_stop_m:
        raw = 2
        why.append("side too close")
    elif side <= a.side_warn_m:
        raw = max(raw, 1)
        why.append("side close")

    hf, lab, ttc_min = 0.0, "", INF
    if not a.no_cam:
        if not cam_ok:
            why.append("cam stale")
        else:
            for tr, h_, t_ in pt:
                if h_ > hf:
                    hf, lab = h_, f"{tr.label}#{tr.id}"
                ttc_min = min(ttc_min, t_)
            if hf >= a.cam_stop_frac or ttc_min <= a.ttc_stop:
                raw = 2
                why.append(f"cam:{lab or '?'} ttc={fmt_ttc(ttc_min)}")
            elif hf >= a.cam_slow_frac or ttc_min <= a.ttc_slow:
                raw = max(raw, 1)
                why.append(f"cam:{lab or '?'} ttc={fmt_ttc(ttc_min)}")
    level = hyst.update(raw, immediate)
    if level == 2:
        vx = 0.0
    else:
        span = max(a.slow_m - a.stop_m, 1e-6)
        vx = a.vmax * clamp((fv - a.stop_m) / span, 0.0, 1.0) if fv != INF else a.vmax
        if level == 1:
            vx = min(vx, 0.5 * a.vmax)
    vy = 0.0
    if l is not None and r is not None:
        vy = clamp(a.kp * (rv - lv) / 2.0, -a.vy_max, a.vy_max)
    return dict(raw=raw, level=level, vx=vx, vy=vy, hf=hf, label=lab, ttc=ttc_min, why=why)


# ------------------------------------------------------------ notifications
class Notifier:
    COLORS = {"INFO": "\033[36m", "WARN": "\033[33m", "ALERT": "\033[31;1m"}

    def __init__(self, a, t0):
        self.a, self.t0, self.last, self.counts = a, t0, {}, {}
        self.color = sys.stdout.isatty()
        self.csv = None
        if a.log:
            self.csv = csv.writer(open(a.log + ".events.csv", "w", newline=""))
            self.csv.writerow(["t", "level", "kind", "message"])

    def __call__(self, now, level, kind, key, msg):
        if now - self.last.get(key, -1e9) < self.a.cooldown_s:
            return
        self.last[key] = now
        self.counts[kind] = self.counts.get(kind, 0) + 1
        t = now - self.t0
        c0, c1 = (self.COLORS[level], "\033[0m") if self.color else ("", "")
        print(f"{c0}>>> {level:<5} {kind:<10} [{t:6.1f}s] {msg}{c1}", flush=True)
        if level != "INFO" and not self.a.no_beep:
            sys.stdout.write("\a")
            sys.stdout.flush()
        if self.csv:
            self.csv.writerow([f"{t:.2f}", level, kind, msg])


def side_baseline(hist, t_ref):
    vals = [v for (t, v) in hist if t_ref - 3.0 <= t <= t_ref - 0.4 and v is not None]
    return statistics.median(vals) if len(vals) >= 3 else None


def handle_lost(a, ev, now, F, hist, pending, notify):
    _, tr, t_ev, (W, H) = ev
    lab = f"{tr.label}#{tr.id}"
    left_edge = tr.x <= a.edge_frac * W
    right_edge = tr.x + tr.w >= (1 - a.edge_frac) * W
    if left_edge or right_edge:
        side = "L" if left_edge else "R"
        pending.append(dict(side=side, tr=tr, lab=lab, base=side_baseline(hist[side], tr.t_last),
                            deadline=now + a.handoff_window_s))
        return
    lo, hi = W * (0.5 - a.corridor_frac / 2), W * (0.5 + a.corridor_frac / 2)
    hf = tr.h / H
    if hf >= a.cam_slow_frac and tr.x < hi and tr.x + tr.w > lo:
        fv = F[0] if F[1] <= a.stale_s else None
        if fv is not None and fv <= a.slow_m:
            tail = f"forward lidar still holds it at {fv:.2f} m"
        else:
            tail = "NO forward lidar return: it may be above/below the beam, treat it as still present"
        notify(now, "WARN", "LOST-CLOSE", f"lostclose-{tr.id}",
               f"camera lost {lab} while close (box {hf:.0%} of frame); {tail}")


def process_pending(a, now, L, R, pending, notify):
    vals = {"L": L, "R": R}
    for p in list(pending):
        v, age = vals[p["side"]]
        name = "LEFT" if p["side"] == "L" else "RIGHT"
        base = p["base"]
        hit = (v is not None and age <= a.stale_s and
               ((base is not None and base - v >= a.handoff_drop_m) or v <= a.side_warn_m))
        if hit:
            was = f"was ~{base:.2f} m" if base is not None else "no baseline"
            notify(now, "WARN", "HANDOFF", f"handoff-{p['tr'].id}",
                   f"{p['lab']} left camera view at the {name.lower()} edge -> {name} lidar now "
                   f"{v:.2f} m ({was}): object is in the camera's blind side, lidar has it")
            pending.remove(p)
        elif now > p["deadline"]:
            notify(now, "INFO", "HANDOFF", f"handoff-{p['tr'].id}",
                   f"{p['lab']} left camera view at the {name.lower()} edge; no {name} lidar "
                   f"reaction: it passed by or is out of range")
            pending.remove(p)


def crosscheck(a, now, L, R, F, pt, cam_ok, notify):
    f_ok = F[1] <= a.stale_s
    fv = INF if (F[0] is None or not f_ok) else F[0]
    if cam_ok and f_ok:
        close = [(t, h, c) for t, h, c in pt if h >= a.cam_slow_frac or c <= a.ttc_slow]
        for tr, hf, ttc in close:
            lab = f"{tr.label}#{tr.id}"
            if fv <= a.slow_m:
                notify(now, "INFO", "CONFIRMED", f"conf-{tr.id}",
                       f"{lab} seen by camera (box {hf:.0%}, ttc {fmt_ttc(ttc)}) AND forward lidar "
                       f"({fv:.2f} m): high confidence")
            elif fv > 1.5 * a.slow_m:
                what = "nothing" if fv == INF else f"{fv:.1f} m"
                notify(now, "WARN", "CAM-ONLY", f"camonly-{tr.id}",
                       f"{lab} close in camera (box {hf:.0%}, ttc {fmt_ttc(ttc)}) but forward lidar "
                       f"sees {what}: object is off the lidar's narrow beam, trust the camera")
        if fv <= a.slow_m and not any(h >= 0.5 * a.cam_slow_frac for _, h, _ in pt):
            notify(now, "WARN", "LIDAR-ONLY", "lidaronly",
                   f"forward lidar sees an obstacle at {fv:.2f} m but the camera has no matching "
                   f"detection: unclassified obstacle (wall/pole/pillar the model does not know, "
                   f"or glare)")
    for name, (v, age) in (("LEFT", L), ("RIGHT", R)):
        if age <= a.stale_s and v is not None and v <= a.side_warn_m:
            notify(now, "WARN", "SIDE", f"side-{name}",
                   f"{name} lidar {v:.2f} m: wall closing on the {name.lower()} side "
                   f"(the forward camera cannot see the sides)")


def monitor_health(a, now, t_start, health, checks, notify):
    if now - t_start < a.warmup_s:
        return
    for name, ok, enabled in checks:
        if not enabled:
            continue
        prev = health.get(name, True)
        if prev and not ok:
            notify(now, "ALERT", "SENSOR", f"lost-{name}", COVER[name])
        elif not prev and ok:
            notify(now, "INFO", "SENSOR", f"back-{name}", f"{name} sensor recovered")
        health[name] = ok


# ----------------------------------------------------------------------- main
def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--hz", type=float, default=10)
    p.add_argument("--status-every", type=int, default=5, help="print a status line every N loops")
    p.add_argument("--log", help="CSV path; events go to <path>.events.csv")
    p.add_argument("--no-cam", action="store_true")
    p.add_argument("--no-qr", action="store_true")
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--cooldown-s", type=float, default=2.0, help="min seconds between repeats")
    p.add_argument("--warmup-s", type=float, default=3.0)
    g = p.add_argument_group("lidar")
    g.add_argument("--lidar-ports", default="", help="LEFT,RIGHT,FORWARD serial ports")
    g.add_argument("--lidar-baud", type=int, default=115200)
    g.add_argument("--lidar-min-strength", type=int, default=100)
    g.add_argument("--lidar-max-m", type=float, default=12.0)
    g = p.add_argument_group("thresholds")
    g.add_argument("--stop-m", type=float, default=1.0)
    g.add_argument("--slow-m", type=float, default=2.5)
    g.add_argument("--side-warn-m", type=float, default=0.6)
    g.add_argument("--side-stop-m", type=float, default=0.3)
    g.add_argument("--stale-s", type=float, default=0.5)
    g.add_argument("--vmax", type=float, default=0.8)
    g.add_argument("--vy-max", type=float, default=0.4)
    g.add_argument("--kp", type=float, default=0.5)
    g = p.add_argument_group("camera / tracking")
    g.add_argument("--model", default="/usr/share/imx500-models/"
                   "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    g.add_argument("--det-thr", type=float, default=0.5)
    g.add_argument("--corridor-frac", type=float, default=0.5)
    g.add_argument("--cam-slow-frac", type=float, default=0.35)
    g.add_argument("--cam-stop-frac", type=float, default=0.60)
    g.add_argument("--ttc-slow", type=float, default=3.0)
    g.add_argument("--ttc-stop", type=float, default=1.5)
    g.add_argument("--edge-frac", type=float, default=0.06,
                   help="box within this fraction of the image edge counts as leaving view")
    g.add_argument("--handoff-drop-m", type=float, default=0.5,
                   help="side-lidar drop vs its recent baseline that counts as a handoff")
    g.add_argument("--handoff-window-s", type=float, default=2.5)
    g = p.add_argument_group("downward QR camera")
    g.add_argument("--down-backend", choices=["picamera2", "opencv"], default="picamera2")
    g.add_argument("--down-index", type=int, default=1)
    g.add_argument("--down-res", default="1920x1080")
    return p.parse_args()


def main():
    a = parse()
    if a.list_cameras:
        from picamera2 import Picamera2
        for i, c in enumerate(Picamera2.global_camera_info()):
            print(i, c)
        return

    outs = {k: Latest() for k in ("L", "R", "F", "cam", "qr")}
    workers = []
    if a.sim:
        w = SimWorld()
        workers += [SimLidar(n, outs[n], w) for n in "LRF"]
        if not a.no_cam:
            workers.append(SimCam(outs["cam"], w))
        if not a.no_qr:
            workers.append(SimQR(outs["qr"], w))
    else:
        ports = [s.strip() for s in a.lidar_ports.split(",") if s.strip()]
        if len(ports) != 3:
            sys.exit("--lidar-ports needs exactly 3 ports: LEFT,RIGHT,FORWARD")
        workers += [TFMini(n, pt, a, outs[n]) for n, pt in zip("LRF", ports)]
        if not a.no_cam:
            workers.append(IMX500Worker(a, outs["cam"]))
        if not a.no_qr:
            workers.append(QRWorker(a, outs["qr"]))
    for wk in workers:
        wk.start()
        if wk.needs_ready:
            wk.ready.wait(timeout=180)

    t0 = time.monotonic()
    notify = Notifier(a, t0)
    hyst = Hysteresis()
    hist = {"L": deque(maxlen=300), "R": deque(maxlen=300)}
    pending, health, reported = [], {}, set()
    time_in, seen_qr = [0.0, 0.0, 0.0], set()
    log = None
    if a.log:
        log = csv.writer(open(a.log, "w", newline=""))
        log.writerow(["t", "L", "R", "F", "cam_h", "cam_track", "ttc", "raw", "state",
                      "vx", "vy", "qr_n", "qr_first"])
    period = 1.0 / a.hz
    print("t(s)   L     R     F    | top path track        | state (raw)  vx    vy   | QR | notes")
    loop_i = 0
    try:
        while a.duration <= 0 or time.monotonic() - t0 < a.duration:
            now = time.monotonic()
            for wk in workers:
                if wk.error and wk.name not in reported:
                    print(f"[!] {wk.name} worker stopped: {wk.error}")
                    reported.add(wk.name)
            L, R, F = outs["L"].get(), outs["R"].get(), outs["F"].get()
            for k, s in (("L", L), ("R", R)):
                if s[1] <= a.stale_s:
                    hist[k].append((now, s[0]))
            fd, cam_age = outs["cam"].get()
            qrs, qr_age = outs["qr"].get()
            qrs = qrs or []
            cam_ok = (not a.no_cam) and fd is not None and cam_age <= 1.0
            pt = path_tracks(a, fd) if cam_ok else []

            d = fuse(a, L, R, F, pt, cam_ok, hyst)
            while EVENTS:
                handle_lost(a, EVENTS.popleft(), now, F, hist, pending, notify)
            process_pending(a, now, L, R, pending, notify)
            crosscheck(a, now, L, R, F, pt, cam_ok, notify)
            monitor_health(a, now, t0, health,
                           [("F", F[1] <= a.stale_s, True), ("L", L[1] <= a.stale_s, True),
                            ("R", R[1] <= a.stale_s, True), ("CAM", cam_age <= 1.0, not a.no_cam),
                            ("QR", qr_age <= 3.0, not a.no_qr)], notify)

            time_in[d["level"]] += period
            for q in qrs:
                seen_qr.add(q.data)
            t = now - t0
            if loop_i % max(1, a.status_every) == 0:
                top = max(pt, key=lambda x: x[1]) if pt else None
                camtxt = (f"{top[0].label[:10]}#{top[0].id} {top[1]:.2f} ttc={fmt_ttc(top[2])}"
                          if top else "-")
                qtxt = (f"'{qrs[0].data}' dx={qrs[0].dx:+.0f} dy={qrs[0].dy:+.0f}" if qrs else "-")
                print(f"{t:6.1f} {fmt(L[0])} {fmt(R[0])} {fmt(F[0])} | {camtxt:<21} | "
                      f"{STATES[d['level']]:<5} ({STATES[d['raw']]:<5}) {d['vx']:+.2f} {d['vy']:+.2f}"
                      f" | {qtxt} | {'; '.join(d['why'])}")
            if log:
                log.writerow([f"{t:.2f}", L[0], R[0], F[0], f"{d['hf']:.3f}", d["label"],
                              "" if d["ttc"] == INF else f"{d['ttc']:.2f}",
                              STATES[d["raw"]], STATES[d["level"]], f"{d['vx']:.2f}",
                              f"{d['vy']:.2f}", len(qrs), qrs[0].data if qrs else ""])
            loop_i += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        for wk in workers:
            wk.stop()
    tot = sum(time_in) or 1
    print("\n--- summary ---")
    for s, tt in zip(STATES, time_in):
        print(f"{s:<5} {tt:6.1f}s ({100 * tt / tot:4.1f}%)")
    print("alerts:", dict(sorted(notify.counts.items())) or "none")
    print("QR codes seen:", sorted(seen_qr) or "none")


if __name__ == "__main__":
    main()
