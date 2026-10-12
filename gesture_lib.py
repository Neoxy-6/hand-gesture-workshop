"""Lib for example codes"""
from collections import Counter, deque
from pathlib import Path
import math

import numpy as np

# === Constants === 
WRIST = 0
MIDDLE_MCP = 9
FINGERTIPS = [4, 8, 12, 16, 20]

FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}
FINGER_NAMES = list(FINGER_CHAINS.keys())

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FEATURE_SETS = ("all", "geom", "coords", "raw")
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"

# =============== 

class HandTracker:
    def __init__(self, max_num_hands=1,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.5,
                 swap_handedness=True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not MODEL_PATH.is_file():
            raise SystemExit(f"找不到模型 {MODEL_PATH}，請確認 assets 資料夾完整。")

        self._mp = mp
        self.swap_handedness = swap_handedness
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def process(self, frame_bgr):
        import time

        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        self.last_ms = max(self.last_ms + 1, int(time.perf_counter() * 1000))
        res = self.landmarker.detect_for_video(image, self.last_ms)

        out = []
        for lm, hd in zip(res.hand_landmarks, res.handedness):
            cat = hd[0]
            side = cat.category_name
            if self.swap_handedness:
                side = {"Left": "Right", "Right": "Left"}.get(side, side)
            out.append({"pts": landmarks_to_pixels(lm, w, h), "handedness": side,
                        "score": cat.score})
        return out

    def close(self):
        self.landmarker.close()


def landmarks_to_pixels(hand_landmarks, w, h):
    return np.array([[p.x * w, p.y * h, p.z * w] for p in hand_landmarks],
                    dtype=np.float64)


def open_camera(index=0, width=640, height=480):
    import sys

    import cv2
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        # 炸開
        raise SystemExit(f"打不開攝影機 {index}。試試 --camera 1，或確認沒有其他程式占用。")
    return cap


def draw_hand(img, pts, color=(0, 255, 0), show_index=False):
    import cv2
    p2 = pts[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p2[a]), tuple(p2[b]), color, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(p2):
        r = 5 if i in FINGERTIPS else 3
        cv2.circle(img, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (40, 40, 40), 1, cv2.LINE_AA)
        if show_index:
            cv2.putText(img, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1, cv2.LINE_AA)


class Sliders:
    def __init__(self, window="params", width=420):
        import cv2
        self.window = window
        self.items = {}
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, width, 40)

    def add(self, name, value, lo, hi, scale=1):
        import cv2
        self.items[name] = scale
        cv2.createTrackbar(name, self.window, int(round(value * scale)),
                           int(round(hi * scale)), lambda _: None)
        cv2.setTrackbarMin(name, self.window, int(round(lo * scale)))
        cv2.setTrackbarPos(name, self.window, int(round(value * scale)))
        cv2.resizeWindow(self.window, 420, 40 + 45 * len(self.items))

    def get(self, name):
        import cv2
        scale = self.items[name]
        pos = cv2.getTrackbarPos(name, self.window)
        return pos if scale == 1 else pos / scale

    def set(self, name, value):
        import cv2
        cv2.setTrackbarPos(name, self.window, int(round(value * self.items[name])))

    def values(self):
        return {k: self.get(k) for k in self.items}


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1, bg=True):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def normalize(pts, handedness="Right", rotate=False):
    p = np.asarray(pts, dtype=np.float64).copy()
    p = p - p[WRIST]
    scale = np.linalg.norm(p[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1.0
    p = p / scale
    if handedness == "Left":
        p[:, 0] *= -1

    if rotate:
        vx, vy = p[MIDDLE_MCP, 0], p[MIDDLE_MCP, 1]
        ang = math.atan2(vx, -vy)
        c, s = math.cos(ang), math.sin(ang)
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = c * x + s * y
        p[:, 1] = -s * x + c * y
    return p


def angle_at(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    cosv = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return math.degrees(math.acos(cosv))


def joint_angles(p):
    out = []
    for chain in FINGER_CHAINS.values():
        for i in range(1, 4):
            out.append(angle_at(p[chain[i - 1]], p[chain[i]], p[chain[i + 1]]))
    return np.array(out)


def tip_distances(p):
    return np.array([np.linalg.norm(p[4] - p[t]) for t in FINGERTIPS[1:]])


def extract_features(pts, handedness="Right", rotate=False, feature_set="all"):
    if feature_set == "raw":
        return np.asarray(pts, dtype=np.float64).reshape(-1)

    p = normalize(pts, handedness, rotate)
    if feature_set == "coords":
        return p.reshape(-1)

    ang = joint_angles(p) / 180.0
    dist = tip_distances(p)
    if feature_set == "geom":
        return np.concatenate([ang, dist])
    return np.concatenate([p.reshape(-1), ang, dist])


def feature_names(feature_set="all"):
    coords = [f"{a}{i}" for i in range(21) for a in "xyz"]
    angs = [f"ang_{f}{j}" for f in FINGER_NAMES for j in range(1, 4)]
    dists = [f"d_thumb_{f}" for f in FINGER_NAMES[1:]]
    return {"all": coords + angs + dists, "geom": angs + dists,
            "coords": coords, "raw": coords}[feature_set]


def augment_pts(pts, rng, max_rot_deg=15.0, scale_range=(0.6, 1.0), noise_px=2.0):
    p = np.asarray(pts, dtype=np.float64).copy()
    c = p[WRIST].copy()
    p -= c

    ang = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    cs, sn = math.cos(ang), math.sin(ang)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0] = cs * x - sn * y
    p[:, 1] = sn * x + cs * y

    p *= rng.uniform(*scale_range)
    p += c
    p += rng.normal(0.0, noise_px, size=p.shape)
    return p


class NoFilter:
    name = "none"

    def update(self, x, t):
        return np.asarray(x, dtype=np.float64)

    def reset(self):
        pass

    def describe(self):
        return "none"


class MovingAverage:
    name = "moving avg"

    def __init__(self, n=5):
        self.n = n
        self.buf = deque(maxlen=n)

    def update(self, x, t):
        self.buf.append(np.asarray(x, dtype=np.float64))
        return np.mean(self.buf, axis=0)

    def reset(self):
        self.buf = deque(maxlen=self.n)

    def set_n(self, n):
        self.n = max(1, int(n))
        self.buf = deque(list(self.buf)[-self.n:], maxlen=self.n)

    def describe(self):
        return f"moving avg  N={self.n}  (lag ~{(self.n - 1) / 2:.1f} frames)"


class EMA:
    name = "EMA"

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.y = None

    def update(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None

    def describe(self):
        return f"EMA  alpha={self.alpha:.2f}"


def smoothing_factor(dt, cutoff):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    name = "One Euro"

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def update(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def describe(self):
        return f"One Euro  min_cutoff={self.min_cutoff:.2f}  beta={self.beta:.4f}"


class MedianFilter:
    name = "median"

    def __init__(self, n=5):
        self.n = n
        self.buf = deque(maxlen=n)

    def update(self, x, t):
        self.buf.append(np.asarray(x, dtype=np.float64))
        return np.median(self.buf, axis=0)

    def reset(self):
        self.buf = deque(maxlen=self.n)

    def set_n(self, n):
        self.n = max(1, int(n))
        self.buf = deque(list(self.buf)[-self.n:], maxlen=self.n)

    def describe(self):
        return f"median  N={self.n}"


class OutlierReject:
    def __init__(self, threshold=60.0, max_skip=3, inner=None):
        self.threshold = threshold
        self.max_skip = max_skip
        self.inner = inner or NoFilter()
        self.name = "reject+" + self.inner.name
        self.reset()

    def reset(self):
        self.last = None
        self.skipped = 0
        self.inner.reset()

    def update(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.last is not None:
            jump = np.max(np.abs(x - self.last))
            if jump > self.threshold and self.skipped < self.max_skip:
                self.skipped += 1
                x = self.last
            else:
                self.skipped = 0
        self.last = x
        return self.inner.update(x, t)

    def describe(self):
        return f"reject>{self.threshold:.0f}px + {self.inner.describe()}"


class PredictionStabilizer:
    UNKNOWN = "?"

    def __init__(self, window=7, hold=3, threshold=0.6):
        self.window = window
        self.hold = hold
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.votes = deque(maxlen=self.window)
        self.current = self.UNKNOWN
        self.candidate = None
        self.count = 0

    def update(self, proba, labels):
        k = int(np.argmax(proba))
        raw = labels[k] if proba[k] >= self.threshold else self.UNKNOWN
        self.votes.append(raw)

        winner = Counter(self.votes).most_common(1)[0][0]
        if winner == self.current:
            self.candidate, self.count = None, 0
        elif winner == self.candidate:
            self.count += 1
            if self.count >= self.hold:
                self.current, self.candidate, self.count = winner, None, 0
        else:
            self.candidate, self.count = winner, 1
            if self.hold <= 1:
                self.current, self.candidate, self.count = winner, None, 0
        return self.current

    def describe(self):
        return f"window={self.window}  hold={self.hold}  conf>={self.threshold:.2f}"


DATA_COLUMNS = (["label", "handedness", "take", "user"]
                + [f"{a}{i}" for i in range(21) for a in "xyz"])


def load_dataset(path):
    import csv
    pts, hand, lab, take, user = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab.append(row["label"])
            hand.append(row["handedness"])
            take.append(row["take"])
            user.append(row.get("user", ""))
            pts.append([float(row[f"{a}{i}"]) for i in range(21) for a in "xyz"])
    if not pts:
        raise SystemExit(f"{path} 裡沒有資料。先跑 02_data_collection.py 錄資料。")
    return (np.array(pts).reshape(-1, 21, 3), np.array(hand), np.array(lab),
            np.array(take), np.array(user))


def build_features(pts, handedness, rotate=False, feature_set="all"):
    return np.array([extract_features(p, h, rotate, feature_set)
                     for p, h in zip(pts, handedness)])
