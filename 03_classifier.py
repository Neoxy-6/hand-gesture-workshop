"""
小實作 2-2：訓練分類器

    python 03_classifier.py                      # 預設 SVM + 82 維特徵
    python 03_classifier.py --model knn --k 5
    python 03_classifier.py --augment 3          # 每筆訓練資料多做 3 筆變形
    python 03_classifier.py --compare            # 比較不同特徵和分類器的組合
    python 03_classifier.py --test-user amy      # 換人駕駛：用 amy 錄的資料當測試集
    python 03_classifier.py --live               # 開鏡頭即時看分類器的信心（用存好的分類器）

可以改的參數（改完重跑，比較混淆矩陣有什麼不同）
    --model knn|svm|rf|mlp     --features all|geom|coords|raw    --rotate
    --k 5   --C 10   --gamma scale   --kernel rbf|linear   --augment 3
    --split take|random        --test-size 0.25

會印出：訓練 / 測試準確率、交叉驗證、每類的 precision / recall / F1、混淆矩陣，
並把分類器存到 models/gesture.joblib 給 04_filtering.py 用。

切分方式（--split）
    take    （預設）同一段連續錄的資料只會在訓練或測試其中一邊，分數比較誠實
    random  隨機切。連續幀幾乎一樣，測試集每一筆在訓練集都有雙胞胎，分數會虛高
"""

import argparse
import os
import time
import warnings
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gesture_lib import FEATURE_SETS, augment_pts, build_features, load_dataset

MODELS = ("knn", "svm", "rf", "mlp")


# ---------------------------------------------------------------------------
# 分類器
# ---------------------------------------------------------------------------

def make_classifier(name, args):
    """kNN、SVM、MLP 都靠距離或梯度，要先標準化；隨機森林一次只看一個特徵的大小，不需要。"""
    if name == "knn":
        clf = KNeighborsClassifier(n_neighbors=args.k)
    elif name == "svm":
        # probability=True 才有 predict_proba（04 的信心門檻要用）
        clf = SVC(kernel=args.kernel, C=args.C, gamma=args.gamma,
                  probability=True, random_state=args.seed)
    elif name == "rf":
        # n_jobs 不要設 -1：一次只預測一筆時，開多執行緒的成本比計算本身還大
        return Pipeline([("clf", RandomForestClassifier(
            n_estimators=100, random_state=args.seed))])
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                            random_state=args.seed)
    else:
        raise ValueError(name)
    # 標準化的 μ、σ 只用訓練集算，包在 Pipeline 裡就不會不小心用到測試集
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

def split_by_take(labels, takes, test_size, rng):
    """
    每個類別各自挑大約 test_size 比例的 take 當測試集。
    只有一個 take 的類別，就把那段的最後 test_size 拿來測（時間上切開，比隨機切誠實）。
    """
    test_mask = np.zeros(len(labels), dtype=bool)
    warn = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        lab_takes = list(dict.fromkeys(takes[idx]))       # 保留錄的順序
        if len(lab_takes) >= 2:
            n_test = max(1, round(len(lab_takes) * test_size))
            chosen = rng.choice(lab_takes, size=n_test, replace=False)
            test_mask[idx[np.isin(takes[idx], chosen)]] = True
        else:
            n_test = max(1, int(len(idx) * test_size))
            test_mask[idx[-n_test:]] = True
            warn.append(lab)
    if warn:
        print(f"  注意：{', '.join(warn)} 只錄了一段，改成用那段的最後 {test_size:.0%} 測試。"
              "多錄幾段（多按幾次 a）分數會更可信。")
    return ~test_mask, test_mask


def split_random(labels, test_size, rng):
    test_mask = np.zeros(len(labels), dtype=bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        n_test = max(1, int(round(len(idx) * test_size)))
        test_mask[rng.choice(idx, size=n_test, replace=False)] = True
    return ~test_mask, test_mask


def make_cv(labels, takes, n_splits, by_take, seed):
    """交叉驗證：切 take 時用 StratifiedGroupKFold，take 不夠多就退回一般的 StratifiedKFold。"""
    min_count = min(np.unique(labels, return_counts=True)[1])
    n_splits = max(2, min(n_splits, min_count))
    if by_take:
        per_label = defaultdict(set)
        for l, t in zip(labels, takes):
            per_label[l].add(t)
        if min(len(s) for s in per_label.values()) >= n_splits:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed), takes
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), None


# ---------------------------------------------------------------------------
# 資料擴增（只做在訓練集上！）
# ---------------------------------------------------------------------------

def augment(pts, hand, labels, takes, copies, seed):
    if copies <= 0:
        return pts, hand, labels, takes
    rng = np.random.default_rng(seed)
    new = [augment_pts(p, rng) for _ in range(copies) for p in pts]
    return (np.concatenate([pts, np.array(new)]),
            np.concatenate([hand] * (copies + 1)),
            np.concatenate([labels] * (copies + 1)),
            np.concatenate([takes] * (copies + 1)))


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------

def print_confusion(cm, classes):
    w = max(6, max(len(c) for c in classes) + 1)
    print(" " * w + "預測 →")
    print("真實↓".ljust(w - 2) + "".join(c[:w - 1].rjust(w) for c in classes))
    for i, c in enumerate(classes):
        cells = "".join((f"[{v}]" if i == j else str(v)).rjust(w) for j, v in enumerate(cm[i]))
        print(c[:w - 1].ljust(w) + cells)


def most_confused(cm, classes):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.max() == 0:
        return None
    i, j = np.unravel_index(np.argmax(off), off.shape)
    return classes[i], classes[j], off[i, j], cm[i].sum()


def plot_confusion(cm, classes, path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(classes), 0.8 + 0.8 * len(classes)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"混淆矩陣圖存在 {path}")
    if show:
        plt.show()
    plt.close(fig)


def time_per_frame(model, x, n=200):
    """模擬即時辨識：一次只丟一筆進去，量 predict_proba 要幾毫秒。"""
    x = x.reshape(1, -1)
    model.predict_proba(x)
    t = time.perf_counter()
    for _ in range(n):
        model.predict_proba(x)
    return (time.perf_counter() - t) / n * 1000


# ---------------------------------------------------------------------------
# 即時模式：看分類器對每一類有多確定
# ---------------------------------------------------------------------------

def live(args):
    import cv2
    import joblib

    from gesture_lib import HandTracker, Sliders, draw_hand, extract_features, open_camera, put_text

    if not os.path.exists(args.out):
        raise SystemExit(f"找不到 {args.out}，先跑一次 python 03_classifier.py 訓練。")
    bundle = joblib.load(args.out)
    model, labels = bundle["model"], bundle["labels"]
    print(f"載入 {args.out}：{bundle['model_name']}，特徵 {bundle['features']}，類別 {labels}")
    print("params 視窗：conf % 是信心門檻（60 = 0.60），最高機率低於它就回答 ?。按 q 離開。")

    cap = open_camera(args.camera)
    tracker = HandTracker(max_num_hands=1)
    sliders = Sliders("params")
    sliders.add("conf %", 0.6, 0.0, 1.0, scale=100)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        conf = sliders.get("conf %")

        # 右邊畫每一類的機率長條
        panel = np.full((frame.shape[0], 260, 3), 25, np.uint8)
        if hands:
            hand = hands[0]
            draw_hand(frame, hand["pts"])
            x = extract_features(hand["pts"], hand["handedness"], bundle["rotate"],
                                 bundle["features"])
            proba = model.predict_proba(x.reshape(1, -1))[0]
            k = int(np.argmax(proba))
            answer = labels[k] if proba[k] >= conf else "?"
            for i, (lab, pr) in enumerate(zip(labels, proba)):
                y = 40 + i * 30
                color = (0, 220, 0) if i == k and proba[k] >= conf else (160, 160, 160)
                cv2.rectangle(panel, (100, y - 14), (100 + int(pr * 150), y + 4), color, -1)
                put_text(panel, f"{lab[:10]:<10}", (8, y), 0.45, bg=False)
                put_text(panel, f"{pr:.2f}", (205, y), 0.45, bg=False)
            gate = 100 + int(conf * 150)
            cv2.line(panel, (gate, 20), (gate, 40 + len(labels) * 30), (0, 0, 255), 1)
            put_text(frame, answer, (10, frame.shape[0] - 15), 1.2, (0, 255, 255), 2)
        else:
            put_text(panel, "no hand", (8, 40), 0.6, bg=False)
        put_text(frame, f"conf >= {conf:.2f}   [q] quit", (10, 25), 0.6)

        cv2.imshow("03 live classifier", np.hstack([frame, panel]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def prepare(args, pts, hand, labels, takes, users, rng):
    if args.test_user:
        test = users == args.test_user
        if not test.any():
            raise SystemExit(f"資料裡沒有 user = {args.test_user}")
        train = ~test
        how = f"換人駕駛：{args.test_user} 的資料當測試集"
    elif args.split == "take":
        train, test = split_by_take(labels, takes, args.test_size, rng)
        how = "依 take 切分"
    else:
        train, test = split_random(labels, args.test_size, rng)
        how = "隨機切分（分數可能虛高）"
    return train, test, how


def run_one(args, model_name, feature_set, data, train, test, verbose=True):
    pts, hand, labels, takes = data
    tr_pts, tr_hand, tr_lab, tr_take = augment(
        pts[train], hand[train], labels[train], takes[train], args.augment, args.seed)

    X_tr = build_features(tr_pts, tr_hand, args.rotate, feature_set)
    X_te = build_features(pts[test], hand[test], args.rotate, feature_set)
    y_tr, y_te = tr_lab, labels[test]

    model = make_classifier(model_name, args)

    cv, groups = make_cv(y_tr, tr_take, args.cv, args.split == "take", args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, groups=groups)
        model.fit(X_tr, y_tr)

    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    pred = model.predict(X_te)
    test_acc = accuracy_score(y_te, pred)
    return {"model": model, "X_tr": X_tr, "X_te": X_te, "y_te": y_te, "pred": pred,
            "train_acc": train_acc, "test_acc": test_acc,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "n_train": len(y_tr)}


def compare(args, data, train, test):
    print(f"\n比較 {len(MODELS)} 種分類器 × {len(FEATURE_SETS)} 種特徵"
          f"（augment={args.augment}, rotate={args.rotate}）\n")
    print(f"{'特徵':<8}{'分類器':<6}{'訓練':>8}{'交叉驗證':>16}{'測試':>8}{'ms/幀':>8}")
    rows = []
    for fs in FEATURE_SETS:
        for m in MODELS:
            r = run_one(args, m, fs, data, train, test)
            ms = time_per_frame(r["model"], r["X_te"][0])
            rows.append((fs, m, r, ms))
            print(f"{fs:<8}{m:<6}{r['train_acc']:>8.3f}"
                  f"{r['cv_mean']:>10.3f} ± {r['cv_std']:.3f}{r['test_acc']:>8.3f}{ms:>8.2f}")
    best = max(rows, key=lambda x: x[2]["test_acc"])
    print(f"\n測試準確率最高：特徵 {best[0]} + {best[1]}（{best[2]['test_acc']:.3f}）")
    print("看三件事：raw 特徵和正規化過的特徵差多少？訓練很高、測試很低的組合是不是在背答案？"
          "哪個組合每幀太慢、不適合即時？")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/gestures.csv")
    ap.add_argument("--model", choices=MODELS, default="svm")
    ap.add_argument("--features", choices=FEATURE_SETS, default="all")
    ap.add_argument("--rotate", action="store_true", help="特徵加上旋轉對齊（讚和倒讚會分不出來）")
    ap.add_argument("--split", choices=("take", "random"), default="take")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--test-user", help="用這個人錄的資料當測試集")
    ap.add_argument("--augment", type=int, default=0, help="每筆訓練資料多做幾筆變形")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--k", type=int, default=5, help="kNN 的 k")
    ap.add_argument("--C", type=float, default=10.0, help="SVM 的 C：越大越不容許分錯")
    ap.add_argument("--gamma", default="scale", help="RBF 的 gamma：越大分界越彎")
    ap.add_argument("--kernel", choices=("rbf", "linear"), default="rbf")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--live", action="store_true", help="開鏡頭即時看分類器（不重新訓練）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="models/gesture.joblib")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        args.gamma = float(args.gamma)
    except ValueError:
        pass

    if args.live:
        live(args)
        return

    pts, hand, labels, takes, users = load_dataset(args.data)
    classes, counts = np.unique(labels, return_counts=True)
    print(f"讀到 {len(labels)} 筆，{len(classes)} 類，{len(np.unique(takes))} 段 take")
    for c, n in zip(classes, counts):
        print(f"  {c:<12}{n:5d}")
    if counts.max() > 2 * counts.min():
        print("  注意：各類筆數差很多，少的那類比較容易被認成多的那類。")

    rng = np.random.default_rng(args.seed)
    train, test, how = prepare(args, pts, hand, labels, takes, users, rng)
    print(f"\n{how}：訓練 {train.sum()} 筆，測試 {test.sum()} 筆")
    data = (pts, hand, labels, takes)

    if args.compare:
        compare(args, data, train, test)
        return

    r = run_one(args, args.model, args.features, data, train, test)
    model = r["model"]
    print(f"\n分類器 {args.model}，特徵 {args.features}（{r['X_tr'].shape[1]} 維），"
          f"augment={args.augment}，rotate={args.rotate}")
    print(f"  訓練準確率  {r['train_acc']:.3f}   （{r['n_train']} 筆，含擴增）")
    print(f"  交叉驗證    {r['cv_mean']:.3f} ± {r['cv_std']:.3f}")
    print(f"  測試準確率  {r['test_acc']:.3f}")
    gap = r["train_acc"] - r["test_acc"]
    if r["train_acc"] < 0.8:
        print("  → 訓練就不好：欠擬合，特徵或分類器不夠力，或資料標錯")
    elif gap > 0.15:
        print("  → 訓練好、測試差：過擬合（在背答案），試試補資料、--augment，或調小 C / 調大 k")
    if r["cv_std"] > 0.1:
        print("  → 交叉驗證標準差大：表現不穩定，資料可能不夠多或不夠雜")

    print("\n每一類的表現（precision 看欄、recall 看列）")
    print(classification_report(r["y_te"], r["pred"], labels=model.classes_,
                                digits=3, zero_division=0))

    cm = confusion_matrix(r["y_te"], r["pred"], labels=model.classes_)
    print("混淆矩陣（對角線 [ ] 是答對的）")
    print_confusion(cm, list(model.classes_))
    mc = most_confused(cm, list(model.classes_))
    if mc:
        a, b, n, tot = mc
        print(f"\n最常搞混：真的是 {a}，被認成 {b}（{n}/{tot}）。"
              f"想想它們差在哪根手指、哪個角度，然後補資料或加特徵。")

    ms = time_per_frame(model, r["X_te"][0])
    print(f"\n即時辨識每幀約 {ms:.2f} ms（30 FPS 一幀只有 33 ms）")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import joblib
    joblib.dump({"model": model, "labels": [str(c) for c in model.classes_],
                 "features": args.features, "rotate": args.rotate,
                 "model_name": args.model}, args.out)
    print(f"分類器存到 {args.out}")

    plot_confusion(cm, list(model.classes_),
                   os.path.join(os.path.dirname(args.out) or ".", "confusion_matrix.png"),
                   show=not args.no_plot)


if __name__ == "__main__":
    main()
