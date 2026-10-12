"""
小實作 2-1：錄資料

    python 02_data_collection.py
    python 02_data_collection.py --labels none,rock,paper,scissors,ok --user amy

按鍵
    0 ~ 9   選類別（對應 --labels 的順序）
    a       倒數 1 秒後自動錄 N 筆（預設 150）
    空白鍵   錄 1 筆
    u       刪掉上一段錄的資料
    q       存檔離開（Esc 也可以）

錄的時候慢慢改變遠近、位置、角度，左右手都錄。
多錄一類 none（放鬆的手、什麼都不是的手）。

存的是「原始像素座標」，不是特徵。特徵在 03_classifier.py 才算，
所以之後改特徵不用重錄。每按一次 a 是一個 take，03 可以用 take 來切訓練 / 測試，
避免連續幀的「雙胞胎」讓分數虛高。
"""

import argparse
import csv
import os
import time
from collections import Counter

import cv2

from gesture_lib import DATA_COLUMNS, HandTracker, draw_hand, open_camera, put_text


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DATA_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def make_row(label, hand, take, user):
    row = {"label": label, "handedness": hand["handedness"], "take": take, "user": user}
    for i, (x, y, z) in enumerate(hand["pts"]):
        row[f"x{i}"], row[f"y{i}"], row[f"z{i}"] = f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="none,rock,paper,scissors",
                    help="類別名稱，用逗號分隔，最多 10 個（只能用英數字）")
    ap.add_argument("--out", default="data/gestures.csv")
    ap.add_argument("--n", type=int, default=150, help="按 a 一次錄幾筆")
    ap.add_argument("--user", default="me", help="錄的人，之後可以用來做「換人測試」")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()][:10]
    rows = load_rows(args.out)
    session = time.strftime("%m%d%H%M%S")
    take_no = 0
    current = 0
    auto_left, auto_start, auto_take = 0, 0.0, None
    manual_take = None
    takes_history = []                      # 可以 undo 的 take 順序

    cap = open_camera(args.camera)
    tracker = HandTracker(max_num_hands=1)
    print(__doc__)
    print("類別：", "  ".join(f"[{i}] {l}" for i, l in enumerate(labels)))
    print(f"已經有 {len(rows)} 筆資料在 {args.out}")

    def new_take():
        nonlocal take_no
        take_no += 1
        t = f"{args.user}-{session}-{take_no:03d}"
        takes_history.append(t)
        return t

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        hand = hands[0] if hands else None
        if hand:
            draw_hand(frame, hand["pts"])

        # 自動錄
        now = time.perf_counter()
        if auto_left > 0:
            if now < auto_start:
                put_text(frame, f"get ready... {auto_start - now:.1f}", (200, 240), 1.0,
                         (0, 255, 255), 2)
            elif hand:
                rows.append(make_row(labels[current], hand, auto_take, args.user))
                auto_left -= 1
                if auto_left == 0:
                    save_rows(args.out, rows)
                    print(f"錄好一段 {labels[current]}（take {auto_take}），已存檔")
            cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)
            put_text(frame, f"REC {args.n - auto_left}/{args.n}",
                     (frame.shape[1] - 170, 30), 0.6, (0, 0, 255))

        # 狀態列
        counts = Counter(r["label"] for r in rows)
        put_text(frame, f"class [{current}] {labels[current]}", (10, 30), 0.8, (0, 255, 0), 2)
        y = 60
        for i, l in enumerate(labels):
            mark = ">" if i == current else " "
            put_text(frame, f"{mark}{i} {l:<10} {counts.get(l, 0):5d}", (10, y), 0.5)
            y += 22
        put_text(frame, "[0-9] class  [a] auto  [space] one  [u] undo  [q] save+quit",
                 (10, frame.shape[0] - 15), 0.5)
        if not hand:
            put_text(frame, "no hand", (10, frame.shape[0] - 45), 0.6, (0, 0, 255))

        cv2.imshow("02 data collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif ord("0") <= key <= ord("9") and key - ord("0") < len(labels):
            if auto_left == 0:
                current = key - ord("0")
                manual_take = None
        elif key == ord("a") and auto_left == 0:
            auto_take = new_take()
            auto_left, auto_start = args.n, now + 1.0
        elif key == ord(" ") and hand and auto_left == 0:
            if manual_take is None:
                manual_take = new_take()
            rows.append(make_row(labels[current], hand, manual_take, args.user))
            save_rows(args.out, rows)
        elif key == ord("u") and auto_left == 0 and takes_history:
            t = takes_history.pop()
            before = len(rows)
            rows = [r for r in rows if r["take"] != t]
            if manual_take == t:
                manual_take = None
            save_rows(args.out, rows)
            print(f"刪掉 take {t}：{before - len(rows)} 筆")

    save_rows(args.out, rows)
    print(f"共 {len(rows)} 筆，存在 {args.out}")
    for l, c in sorted(Counter(r["label"] for r in rows).items()):
        print(f"  {l:<12}{c}")
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
