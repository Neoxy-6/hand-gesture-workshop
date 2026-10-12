"""
小實作 3：讓它穩下來

    python 04_filtering.py
    python 04_filtering.py --model models/gesture.joblib

params 視窗的滑桿（即時生效）
    filter      0 none  1 moving avg  2 EMA  3 One Euro  4 median
    N           移動平均 / 中位數取幾幀
    alpha %     EMA（30 = 0.30）：大 → 跟手、會抖；小 → 平滑、會拖
    min cutoff % One Euro（100 = 1.00 Hz）：調小 → 靜止時更穩
    beta x1000  One Euro（7 = 0.007）：調大 → 快速移動時更跟手
    reject px   跳點剔除門檻（像素），按 o 開啟
    noise px    按 n 加的雜訊大小
    window      答案穩定器：最近幾幀投票
    hold        答案穩定器：新答案要連續贏幾次才切換（遲滯）
    conf %      答案穩定器：最高機率低於這個值（60 = 0.60）就回答 ?

按鍵
    f       切換濾波器（同 filter 滑桿）
    o       跳點剔除 開 / 關（包在目前的濾波器外面）
    n       加高斯雜訊（讓抖動更明顯）
    j       加跳點（偶爾整隻手飛走一幀）
    l       模擬舊筆電：幀率忽快忽慢
    s       答案穩定器 開 / 關（信心門檻 + 多數決 + 遲滯）
    r       重設濾波器
    q       離開（Esc 也可以）

看畫面下方的曲線：橘色點是原始的食指尖 x 座標（每一幀一個），綠線是濾波後。
最下面一條是「原始 − 濾波後」，自動放大：抖動被濾掉多少、移動時落後多少都看得到。
濾波器是 none 時它是一條平線。骨架也一樣：橘色是原始，綠色是濾波後。
    jitter  每一幀食指尖移動多少像素   → 手靜止時看，越小越穩
    lag     濾波後和原始位置差多少像素 → 手移動時看，越小越即時
目標：找到一組「靜止不抖、揮動不拖」的參數。
"""

import argparse
import os
import random
import time
from collections import deque

import cv2
import numpy as np

from gesture_lib import (EMA, MedianFilter, MovingAverage, NoFilter, OneEuroFilter,
                         OutlierReject, PredictionStabilizer, HandTracker, Sliders,
                         draw_hand, extract_features, open_camera, put_text)

TIP = 8                     # 食指尖
HISTORY = 200               # 曲線顯示幾幀
METRIC_WINDOW = 30          # jitter / lag 用最近幾幀算
RAW_COLOR = (0, 140, 255)   # 橘色（OpenCV 是 BGR）
FILT_COLOR = (0, 230, 0)    # 綠色
DEFAULT_FILTER = 2          # 一開始就用 EMA，才看得出原始和濾波後的差別


def make_filters(args):
    return [NoFilter(), MovingAverage(args.n), EMA(args.alpha),
            OneEuroFilter(args.min_cutoff, args.beta), MedianFilter(args.n)]


def make_sliders(args):
    s = Sliders("params")
    s.add("filter", DEFAULT_FILTER, 0, 4)
    s.add("N", args.n, 1, 30)
    s.add("alpha %", args.alpha, 0.05, 1.0, scale=100)
    s.add("min cutoff %", args.min_cutoff, 0.05, 10.0, scale=100)
    s.add("beta x1000", args.beta, 0.0, 0.1, scale=1000)
    s.add("reject px", args.reject, 5, 300)
    s.add("noise px", args.noise, 0, 30)
    s.add("window", args.window, 1, 20)
    s.add("hold", args.hold, 1, 10)
    s.add("conf %", args.threshold, 0.0, 1.0, scale=100)
    return s


def apply_sliders(s, filters, reject, stab, args):
    """每一幀把滑桿的值寫回濾波器和穩定器。改到 N 或 window 才需要清掉舊紀錄。"""
    n = s.get("N")
    for f in (filters[1], filters[4]):
        if f.n != n:
            f.set_n(n)
    filters[2].alpha = s.get("alpha %")
    filters[3].min_cutoff = s.get("min cutoff %")
    filters[3].beta = s.get("beta x1000")
    reject.threshold = s.get("reject px")
    args.noise = s.get("noise px")

    window, hold, conf = s.get("window"), s.get("hold"), s.get("conf %")
    if (stab.window, stab.hold) != (window, hold):
        stab.window, stab.hold = window, hold
        stab.reset()
    stab.threshold = conf


def draw_series(panel, series, to_xy, color, th, dots=False):
    seg = []
    for i, v in enumerate(series):
        if v is None:
            if len(seg) > 1:
                cv2.polylines(panel, [np.array(seg)], False, color, th, cv2.LINE_AA)
            seg = []
            continue
        seg.append(to_xy(i, v))
        if dots:
            cv2.circle(panel, seg[-1], 2, color, -1, cv2.LINE_AA)
    if len(seg) > 1:
        cv2.polylines(panel, [np.array(seg)], False, color, th, cv2.LINE_AA)


def draw_plot(raw_hist, filt_hist, width, filter_name="", height=200, diff_height=80):
    """
    上半：食指尖 x 座標。原始是橘色的點（每一幀一個），濾波後是綠線。
    下半：原始 − 濾波後，自動放大。濾掉的抖動和延遲都在這裡，none 時是一條平線。
    """
    panel = np.full((height, width, 3), 25, np.uint8)
    pairs = [(r, f) for r, f in zip(raw_hist, filt_hist) if r is not None]
    if len(pairs) < 2:
        put_text(panel, "show your hand", (10, height // 2), 0.6, bg=False)
        return panel

    top = height - diff_height
    vals = np.array(pairs)
    lo, hi = vals.min(), vals.max()
    pad = max((hi - lo) * 0.1, 10)
    lo, hi = lo - pad, hi + pad

    def x_at(i):
        return int(width - 1 - (len(raw_hist) - 1 - i) * (width / HISTORY))

    header = 30                       # 上面留給文字，曲線不畫進去

    def main_xy(i, v):
        return x_at(i), int(top - 4 - (v - lo) / (hi - lo) * (top - 4 - header))

    # 綠線先畫，橘點畫在上面：兩條完全重疊時也看得到原始的點
    draw_series(panel, filt_hist, main_xy, FILT_COLOR, 2)
    draw_series(panel, raw_hist, main_xy, RAW_COLOR, 1, dots=True)

    # 差值：上下對稱，至少放大到 ±5 像素，小抖動也看得出來
    diff = [None if r is None else r - f for r, f in zip(raw_hist, filt_hist)]
    span = max(5.0, max(abs(d) for d in diff if d is not None) * 1.1)
    mid = top + diff_height // 2

    def diff_xy(i, d):
        return x_at(i), int(mid - d / span * (diff_height // 2 - 4))

    cv2.line(panel, (0, top), (width, top), (70, 70, 70), 1)
    cv2.line(panel, (0, mid), (width, mid), (90, 90, 90), 1)
    cv2.rectangle(panel, (0, top + 1), (width, height), (35, 35, 35), -1)
    cv2.line(panel, (0, mid), (width, mid), (90, 90, 90), 1)
    draw_series(panel, diff, diff_xy, RAW_COLOR, 2)

    put_text(panel, "index tip x:", (10, 20), 0.5, bg=False)
    put_text(panel, "raw", (120, 20), 0.5, RAW_COLOR, bg=False)
    put_text(panel, "filtered", (165, 20), 0.5, FILT_COLOR, bg=False)
    put_text(panel, f"raw - filtered  (+/-{span:.0f} px)", (10, top + 18), 0.45, bg=False)
    if filter_name == "none":
        put_text(panel, "filter = none: raw and filtered are identical, press f",
                 (10, top + diff_height - 12), 0.5, (0, 200, 255), bg=False)
    return panel


def load_classifier(path):
    if not path or not os.path.exists(path):
        print(f"找不到 {path}，只示範座標濾波。先跑 03_classifier.py 就能一起看答案穩定化。")
        return None
    import joblib
    bundle = joblib.load(path)
    print(f"載入分類器 {path}：{bundle.get('model_name', '?')}，"
          f"特徵 {bundle['features']}，類別 {bundle['labels']}")
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="models/gesture.joblib")
    ap.add_argument("--n", type=int, default=5, help="移動平均 / 中位數的 N")
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA 的 alpha")
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.007)
    ap.add_argument("--reject", type=float, default=60.0, help="跳點剔除門檻（像素）")
    ap.add_argument("--noise", type=float, default=4.0, help="按 n 加的雜訊標準差（像素）")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    bundle = load_classifier(args.model)
    filters = make_filters(args)
    fi = DEFAULT_FILTER
    use_reject = False
    reject = OutlierReject(args.reject, inner=filters[fi])
    stab = PredictionStabilizer(args.window, args.hold, args.threshold)
    use_stab = True
    add_noise = add_jumps = laggy = False

    raw_hist = deque([None] * HISTORY, maxlen=HISTORY)
    filt_hist = deque([None] * HISTORY, maxlen=HISTORY)
    jit = deque(maxlen=METRIC_WINDOW)
    lag = deque(maxlen=METRIC_WINDOW)
    prev_tip = None
    last_seen = 0.0
    t_prev, fps = time.perf_counter(), 0.0

    cap = open_camera(args.camera)
    tracker = HandTracker(max_num_hands=1)
    sliders = make_sliders(args)
    print(__doc__)

    def active():
        return reject if use_reject else filters[fi]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if laggy:
            time.sleep(random.uniform(0.0, 0.06))
        frame = cv2.flip(frame, 1)
        t = time.perf_counter()

        # filter 滑桿換了濾波器：從頭開始濾
        if sliders.get("filter") != fi:
            fi = sliders.get("filter")
            filters[fi].reset()
            reject = OutlierReject(reject.threshold, inner=filters[fi])
            jit.clear(); lag.clear()
        apply_sliders(sliders, filters, reject, stab, args)
        hands = tracker.process(frame)

        raw_label = label = ""
        if hands:
            hand = hands[0]
            clean = hand["pts"]
            noisy = clean.copy()
            if add_noise:
                noisy += np.random.normal(0, args.noise, noisy.shape)
            if add_jumps and random.random() < 0.04:
                noisy += np.random.uniform(-150, 150, size=(1, 3)) * [1, 1, 0]

            if t - last_seen > 0.5:             # 手消失一陣子再出現：從頭開始濾
                active().reset()
                stab.reset()
                prev_tip = None
            last_seen = t

            filt = active().update(noisy.reshape(-1), t).reshape(21, 3)
            draw_hand(frame, noisy, color=RAW_COLOR)
            draw_hand(frame, filt, color=FILT_COLOR)

            tip = filt[TIP, :2]
            if prev_tip is not None:
                jit.append(np.linalg.norm(tip - prev_tip))
            prev_tip = tip
            lag.append(np.linalg.norm(tip - clean[TIP, :2]))
            raw_hist.append(noisy[TIP, 0])
            filt_hist.append(filt[TIP, 0])

            if bundle:
                x = extract_features(filt, hand["handedness"], bundle["rotate"],
                                     bundle["features"])
                proba = bundle["model"].predict_proba(x.reshape(1, -1))[0]
                labels = bundle["labels"]
                k = int(np.argmax(proba))
                raw_label = f"{labels[k]} {proba[k]:.2f}"
                label = stab.update(proba, labels) if use_stab else labels[k]
        else:
            raw_hist.append(None)
            filt_hist.append(None)

        # 文字
        dt = max(t - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 / dt
        t_prev = t
        y = 25
        for line in (
            f"FPS {fps:4.1f}" + ("   [l] laggy ON" if laggy else ""),
            f"[f] filter: {active().describe()}",
            f"noise[n]:{'ON' if add_noise else 'off'}  jumps[j]:{'ON' if add_jumps else 'off'}"
            f"  reject[o]:{'ON' if use_reject else 'off'}",
            f"jitter {np.mean(jit) if jit else 0:5.2f} px   lag {np.mean(lag) if lag else 0:5.2f} px",
        ):
            put_text(frame, line, (10, y), 0.55)
            y += 24
        if bundle:
            put_text(frame, f"raw: {raw_label}", (10, frame.shape[0] - 70), 0.6, (200, 200, 200))
            put_text(frame, f"[s] stabilizer {'ON  ' + stab.describe() if use_stab else 'off'}",
                     (10, frame.shape[0] - 45), 0.5)
            put_text(frame, f"{label}", (10, frame.shape[0] - 12), 0.9, (0, 255, 255), 2)

        view = np.vstack([frame, draw_plot(raw_hist, filt_hist, frame.shape[1], filters[fi].name)])
        cv2.imshow("04 filtering", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("f"):
            sliders.set("filter", (fi + 1) % len(filters))
        elif key == ord("o"):
            use_reject = not use_reject
            reject.reset()
        elif key == ord("n"):
            add_noise = not add_noise
        elif key == ord("j"):
            add_jumps = not add_jumps
        elif key == ord("l"):
            laggy = not laggy
        elif key == ord("s"):
            use_stab = not use_stab
            stab.reset()
        elif key == ord("r"):
            active().reset()
            stab.reset()
            jit.clear(); lag.clear()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
