"""
小實作 1：看見你的手

    python 01_hand_tracking.py

params 視窗的滑桿（即時生效）
    det conf %     偵測到手的門檻（60 = 0.60）：調高比較不會亂抓，但手要更清楚
    track conf %   低於這個值就算追丟、重新偵測
    hands          最多偵測幾隻手
    bend angle     手指關節角大於這個值（度）才算「伸直」
    pinch dist %   拇指尖和食指尖距離小於這個值（30 = 0.30 個手掌長）才算碰到

按鍵
    i   顯示 / 隱藏關節點編號
    r   旋轉對齊 開 / 關（看右上角的正規化小視窗）
    p   在終端機印出 82 維特徵
    q   離開（Esc 也可以）

任務
    改下面的 my_gesture_rule()：用角度和距離寫規則，判斷數字 0 ~ 5、OK、捏。
"""

import argparse
import time

import cv2
import numpy as np

from gesture_lib import (FINGER_NAMES, HandTracker, Sliders, draw_hand,
                         extract_features, joint_angles, normalize, open_camera,
                         put_text, tip_distances)

# 規則用的門檻，滑桿會即時改這裡
RULE = {"bend angle": 160, "pinch dist": 0.3}


# ---------------------------------------------------------------------------
# 規則式手勢判斷：不用訓練任何模型，只寫 if
# ---------------------------------------------------------------------------

def finger_states(ang):
    """
    ang: 15 個關節角（度），順序是 拇指 1~3、食指 1~3、…、小指 1~3。
    回傳 dict：每根手指是否伸直。
    """
    a = ang.reshape(5, 3)
    bend = RULE["bend angle"]
    states = {}
    for i, name in enumerate(FINGER_NAMES):
        if name == "thumb":
            # 拇指比較特別：看最後兩個關節
            states[name] = a[i, 1] > bend - 10 and a[i, 2] > bend - 10
        else:
            # 其他手指：中間和指尖那兩個關節都接近伸直
            states[name] = a[i, 1] > bend and a[i, 2] > bend - 10
    return states


def my_gesture_rule(p, ang, dist):
    """
    p     正規化後的 21 個點 (21, 3)，單位是「手掌長」
    ang   15 個關節角（度）
    dist  拇指尖到 食指 / 中指 / 無名指 / 小指 指尖的距離（手掌長）

    回傳要顯示的文字。
    """
    up = finger_states(ang)
    thumb_index = dist[0]
    touch = RULE["pinch dist"]
    rest = up["middle"] and up["ring"] and up["pinky"]

    # OK：拇指和食指尖碰在一起，其他三指伸直
    if thumb_index < touch and rest:
        return "OK"

    # 捏：拇指和食指尖很近，而且食指只是微彎（握拳時食指彎很多，不算捏）
    index_pip = ang[4]                            # 食指中間那個關節
    if thumb_index < touch and index_pip > 120:
        return "pinch"

    # 數字 0 ~ 5：數有幾根手指伸直
    n = sum(up.values())
    return str(n)


# ---------------------------------------------------------------------------
# 畫面
# ---------------------------------------------------------------------------

def draw_normalized_inset(img, p, rotate, size=200, margin=10):
    """右上角畫正規化後的手：手腕在下方中間，單位是手掌長。"""
    h, w = img.shape[:2]
    x0, y0 = w - size - margin, margin
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1)

    s = size * 0.38                               # 1 個手掌長 = 幾個像素
    origin = np.array([x0 + size / 2, y0 + size * 0.85])
    pts = np.zeros((21, 3))
    pts[:, :2] = p[:, :2] * s + origin
    draw_hand(img, pts, color=(0, 200, 255))
    put_text(img, "normalized" + ("  +rotate" if rotate else ""),
             (x0 + 6, y0 + 18), 0.45, bg=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = open_camera(args.camera)
    sliders = Sliders("params")
    sliders.add("det conf %", 0.6, 0.05, 1.0, scale=100)
    sliders.add("track conf %", 0.5, 0.05, 1.0, scale=100)
    sliders.add("hands", 1, 1, 2)
    sliders.add("bend angle", RULE["bend angle"], 90, 180)
    sliders.add("pinch dist %", RULE["pinch dist"], 0.05, 1.0, scale=100)

    tracker_cfg = None
    tracker = None
    show_index, rotate = False, False
    t_prev, fps = time.perf_counter(), 0.0

    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                # 自拍視角，左右手才會對

        # 偵測參數改了就重建 tracker；規則門檻直接寫回 RULE
        cfg = (sliders.get("det conf %"), sliders.get("track conf %"), sliders.get("hands"))
        if cfg != tracker_cfg:
            if tracker:
                tracker.close()
            tracker = HandTracker(max_num_hands=cfg[2], min_detection_confidence=cfg[0],
                                  min_tracking_confidence=cfg[1])
            tracker_cfg = cfg
        RULE["bend angle"] = sliders.get("bend angle")
        RULE["pinch dist"] = sliders.get("pinch dist %")

        hands = tracker.process(frame)

        for hand in hands:
            pts, side = hand["pts"], hand["handedness"]
            draw_hand(frame, pts, show_index=show_index)

            p = normalize(pts, side, rotate)
            ang = joint_angles(p)
            dist = tip_distances(p)
            result = my_gesture_rule(p, ang, dist)

            wx, wy = pts[0, :2].astype(int)
            put_text(frame, f"{side}  {result}", (wx - 40, wy + 35), 0.9, (0, 255, 255), 2)
            draw_normalized_inset(frame, p, rotate)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        put_text(frame, f"FPS {fps:4.1f}   [i]ndex  [r]otate  [p]rint  [q]uit", (10, 25))
        put_text(frame, f"det {cfg[0]:.2f}  track {cfg[1]:.2f}  hands {cfg[2]}  "
                        f"bend>{RULE['bend angle']}  pinch<{RULE['pinch dist']:.2f}",
                 (10, frame.shape[0] - 12), 0.5)

        cv2.imshow("01 hand tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("i"):
            show_index = not show_index
        elif key == ord("r"):
            rotate = not rotate
            print("旋轉對齊：", "開" if rotate else "關")
        elif key == ord("p"):
            if not hands:
                print("畫面上沒有手")
                continue
            h = hands[0]
            f = extract_features(h["pts"], h["handedness"], rotate)
            np.set_printoptions(precision=3, suppress=True, linewidth=110)
            print(f"\n=== {h['handedness']} 手，{len(f)} 維特徵 ===")
            print("正規化座標 (63)：\n", f[:63].reshape(21, 3))
            print("關節角 / 180 (15)：\n", f[63:78].reshape(5, 3))
            print("拇指尖到 食/中/無名/小 指尖 (4)：", f[78:])

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
