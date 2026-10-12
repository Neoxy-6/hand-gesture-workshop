"""
環境檢查

    python 00_env_check.py
    python 00_env_check.py --camera 1

先檢查 Python 版本、套件和模型檔，再打開鏡頭。
把手舉到鏡頭前，看到骨架就過關，終端機會印出 ALL PASS。

按鍵
    q   離開（或 Esc）
"""

import argparse
import hashlib
import importlib
import sys
import time

MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
PACKAGES = ["cv2", "numpy", "mediapipe", "sklearn", "matplotlib", "joblib"]
HAND_FRAMES = 15            # 連續偵測到手幾幀才算過關
TIMEOUT = 60                # 幾秒內沒看到手就結束


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def static_checks():
    results = []
    v = sys.version_info
    results.append(check("Python 3.13", (v.major, v.minor) == (3, 13),
                         f"目前是 {v.major}.{v.minor}.{v.micro}"))

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            results.append(check(f"套件 {name}", True, getattr(mod, "__version__", "")))
        except Exception as exc:
            results.append(check(f"套件 {name}", False,
                                 f"{exc}；請執行 pip install -r requirements.txt"))

    from gesture_lib import MODEL_PATH
    if MODEL_PATH.is_file():
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        results.append(check("手部模型", digest == MODEL_SHA256,
                             "" if digest == MODEL_SHA256 else "檔案損壞，請重新下載專案"))
    else:
        results.append(check("手部模型", False, f"找不到 {MODEL_PATH}"))
    return all(results)


def camera_check(index):
    import cv2

    from gesture_lib import HandTracker, draw_hand, open_camera, put_text

    try:
        cap = open_camera(index)
    except SystemExit as exc:
        return check("攝影機", False, str(exc))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return check("攝影機", False, "開得起來但讀不到畫面，試試 --camera 1")
    check("攝影機", True, f"{frame.shape[1]}x{frame.shape[0]}")

    tracker = HandTracker(max_num_hands=2)
    streak, passed = 0, False
    start = time.perf_counter()
    print("\n把手舉到鏡頭前，看到骨架就過關……")

    while time.perf_counter() - start < TIMEOUT:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        for hand in hands:
            draw_hand(frame, hand["pts"])
            wx, wy = hand["pts"][0, :2].astype(int)
            put_text(frame, hand["handedness"], (wx - 20, wy + 30), 0.8, (0, 255, 255), 2)

        streak = streak + 1 if hands else 0
        if streak >= HAND_FRAMES:
            passed = True

        left = TIMEOUT - (time.perf_counter() - start)
        msg = "PASS! press q" if passed else f"show your hand ({left:.0f}s)"
        put_text(frame, msg, (10, 30), 0.8, (0, 255, 0) if passed else (0, 200, 255), 2)
        put_text(frame, "check: is Left / Right correct?", (10, frame.shape[0] - 15), 0.6)
        cv2.imshow("00 env check", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    return check("偵測到手", passed, "" if passed else "沒看到手：檢查光線、換 --camera，或靠近一點")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    print("環境檢查\n")
    ok = static_checks()
    if not ok:
        print("\n有項目沒通過，先修好再繼續。")
        sys.exit(1)

    ok = camera_check(args.camera)
    print("\nALL PASS" if ok else "\n有項目沒通過，先修好再繼續。")
    if ok:
        print("左右手標反了的話，把 gesture_lib.py 的 swap_handedness 預設值改成另一個。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
