# Codpy Section 8 電腦視覺  X 手勢辨識 

Codpy Section 8 課程程式。

| 實作 | 程式 | 做什麼 |
|---|---|---|
| 0 | `00_env_check.py` | 檢查環境，看到骨架就過關 |
| 1 | `01_hand_tracking.py` | 打開鏡頭、偵測手的關節點，用幾何規則判斷手勢 |
| 2 | `02_data_collection.py`、`03_classifier.py` | 錄資料、訓練分類器、看混淆矩陣 |
| 3 | `04_filtering.py` | 濾波去雜訊，讓答案穩下來 |

## 0. 安裝環境

需要 **Python 3.13** 和一支攝影機。在專案資料夾開終端機：

**Windows（PowerShell）**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
python 00_env_check.py
```

**macOS / Linux**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 00_env_check.py
```

看到 `ALL PASS` 就可以了。之後每次開新的終端機，只要再執行一次 Activate 那一行。

> PowerShell 不讓你執行 Activate 時，先執行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

## 1. 看見你的手

```bash
python 01_hand_tracking.py
```

畫面會畫出 21 個關節點的骨架，右上角是正規化後的手（手腕在下方，單位是「幾個手掌長」）。

| 按鍵 | 功能 |
|---|---|
| `i` | 顯示 / 隱藏關節點編號 0 ~ 20 |
| `r` | 旋轉對齊 開 / 關 |
| `p` | 在終端機印出 82 維特徵 |
| `q` | 離開（Esc 也可以） |

**params 視窗的滑桿**，拖了馬上生效：

| 滑桿 | 意思 |
|---|---|
| `det conf %` | 偵測到手的門檻（60 = 0.60） |
| `track conf %` | 低於它就算追丟、重新偵測 |
| `hands` | 最多偵測幾隻手 |
| `bend angle` | 手指關節角大於幾度算「伸直」 |
| `pinch dist %` | 拇指尖和食指尖多近算「碰到」（30 = 0.30 個手掌長） |

**任務**：修改 `01_hand_tracking.py` 的 `my_gesture_rule()`，用角度和距離寫規則，判斷數字 0 ~ 5、OK、捏，再加一個自己設計的手勢。

## 2. 分類器

### 錄資料

```bash
python 02_data_collection.py --user amy
python 02_data_collection.py --labels none,rock,paper,scissors,ok --user amy
```

| 按鍵 | 功能 |
|---|---|
| `0` ~ `9` | 選類別（對應 `--labels` 的順序） |
| `a` | 倒數 1 秒後自動錄 150 筆，這一段叫一個 take |
| 空白鍵 | 錄 1 筆 |
| `u` | 刪掉上一段 |
| `q` | 存檔離開 |

錄的訣竅：

- **每類至少錄 3 段**，段數、筆數差不多。測試集是用 take 切的，只錄一段的類別，訓練資料會很少。
- 錄的時候慢慢改變遠近、位置、角度。
- 多錄一類 `none`：放鬆、什麼都不是的手。
- 存的是原始座標（`data/gestures.csv`），之後改特徵不用重錄。

### 訓練、看混淆矩陣

```bash
python 03_classifier.py
```

終端機會印出訓練 / 交叉驗證 / 測試準確率、每一類的 precision、recall、F1 和混淆矩陣，並跳出混淆矩陣圖。分類器存在 `models/gesture.joblib`，給 `04_filtering.py` 用。

混淆矩陣的列是真實的手勢、欄是模型的預測，對角線是答對的。找出最常搞混的那一對，想想它們差在哪根手指、哪個角度，然後補資料或加特徵。

### 換參數比比看

改完重跑，看混淆矩陣變了什麼：

| 參數 | 意思 |
|---|---|
| `--model knn\|svm\|rf\|mlp` | 換分類器（預設 svm） |
| `--k 5` | kNN 看幾個鄰居 |
| `--C 10` | SVM：越大越不容許分錯 |
| `--gamma scale` | SVM RBF：越大分界越彎 |
| `--features all\|geom\|coords\|raw` | 82 維 / 只用角度距離 / 只用座標 / 原始像素（反面教材） |
| `--rotate` | 特徵加上旋轉對齊 |
| `--augment 3` | 每筆訓練資料多做 3 筆變形 |
| `--split take\|random` | 依 take 切 / 隨機切（分數會虛高） |
| `--test-user amy` | 換人測試：用 amy 錄的資料當測試集 |
| `--compare` | 一次比較所有分類器 × 特徵的組合 |
| `--live` | 開鏡頭即時看每一類的機率，用 `conf %` 滑桿調「不確定」的門檻 |

```bash
python 03_classifier.py --model knn --k 3
python 03_classifier.py --features raw
python 03_classifier.py --compare
python 03_classifier.py --live
```

## 3. 讓它穩下來

```bash
python 04_filtering.py
```

需要先跑過 `03_classifier.py`，才看得到分類結果；沒有分類器時只示範座標濾波。

畫面上橘色是原始關節點，綠色是濾波後。下方曲線是食指尖的 x 座標，最下面一條是「原始 − 濾波後」：手不動時看抖動被濾掉多少，手揮動時看落後多少。

- `jitter`：每一幀移動多少像素，手靜止時看，越小越穩
- `lag`：濾波後和原始位置差多少像素，手移動時看，越小越跟手

| 按鍵 | 功能 |
|---|---|
| `f` | 切換濾波器 |
| `n` | 加高斯雜訊，讓抖動更明顯 |
| `j` | 加跳點，偶爾整隻手飛走一幀 |
| `o` | 跳點剔除 開 / 關 |
| `l` | 模擬舊筆電：幀率忽快忽慢 |
| `s` | 答案穩定器 開 / 關 |
| `r` | 重設濾波器 |
| `q` | 離開 |

**params 視窗的滑桿**：

| 滑桿 | 意思 |
|---|---|
| `filter` | 0 none、1 移動平均、2 EMA、3 One Euro、4 中位數 |
| `N` | 移動平均 / 中位數取幾幀 |
| `alpha %` | EMA：大 → 跟手、會抖；小 → 平滑、會拖 |
| `min cutoff %` | One Euro：調小，靜止時更穩 |
| `beta x1000` | One Euro：調大，快速移動時更跟手 |
| `reject px` | 跳點剔除門檻（像素） |
| `noise px` | 按 `n` 加的雜訊大小 |
| `window` | 答案穩定器：最近幾幀投票 |
| `hold` | 答案穩定器：新答案要連續贏幾次才切換 |
| `conf %` | 答案穩定器：最高機率低於門檻就回答 `?` |

試試看：

1. 按 `n`，把 alpha 從 100 拉到 10：jitter 變小、lag 變大，這就是「穩 vs 快」。
2. 換 One Euro：先調 min cutoff 到靜止不抖，再加 beta 到揮手不拖。
3. 按 `j` 加跳點：平均擋不住，換中位數或按 `o` 剔除。
4. 按 `s` 關掉穩定器，手放在兩個手勢中間：答案會閃。

## 一些其他檔案

| 位置 | 內容 |
|---|---|
| `gesture_lib.py` | 共用工具：手部偵測、特徵、資料擴增、濾波器、答案穩定器、滑桿 |
| `assets/hand_landmarker.task` | MediaPipe 手部關節點模型 |
| `data/` | 自己錄的資料 |
| `models/` | 訓練好的分類器和混淆矩陣圖 |
