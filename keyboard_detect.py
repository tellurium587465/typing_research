import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

IMAGE_FILE = "frame_10.png"

img = cv2.imread(IMAGE_FILE)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# -----------------------------------------------
# 紫色のバックライトを抽出
# -----------------------------------------------
lower1 = np.array([125, 40, 40])
upper1 = np.array([165, 255, 255])
mask1 = cv2.inRange(hsv, lower1, upper1)

lower2 = np.array([160, 40, 40])
upper2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower2, upper2)

mask = cv2.bitwise_or(mask1, mask2)

# ノイズ除去
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

cv2.imwrite("debug_mask.png", mask)

# -----------------------------------------------
# キーボード外枠を検出（最大輪郭）
# -----------------------------------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest)
print(f"キーボード領域: x={x}, y={y}, w={w}, h={h}")

kb_mask = mask[y:y+h, x:x+w]

# -----------------------------------------------
# 投影計算
# -----------------------------------------------
def smooth(arr, win=3):
    return np.convolve(arr, np.ones(win)/win, mode="same")

def find_centers(proj, thresh):
    centers = []
    in_region = False
    start = 0
    for i, v in enumerate(proj):
        if v > thresh and not in_region:
            in_region = True
            start = i
        elif v <= thresh and in_region:
            in_region = False
            centers.append((start + i) // 2)
    if in_region:
        centers.append((start + len(proj)) // 2)
    return centers

row_proj = smooth(np.sum(kb_mask, axis=1).astype(float), 3)
col_proj = smooth(np.sum(kb_mask, axis=0).astype(float), 3)

from scipy.signal import find_peaks

row_proj = smooth(np.sum(kb_mask, axis=1).astype(float), 3)
col_proj = smooth(np.sum(kb_mask, axis=0).astype(float), 3)

# 行：scipy find_peaksで山のピークを直接検出
row_peaks, props = find_peaks(
    row_proj,
    height=row_proj.max() * 0.3,
    distance=20
)
row_centers = row_peaks.tolist()

# 5行より多い場合は高さ上位5つを採用
if len(row_centers) > 5:
    heights = [row_proj[r] for r in row_centers]
    row_centers = sorted(
        sorted(zip(heights, row_centers), reverse=True)[:5],
        key=lambda x: x[1]
    )
    row_centers = [r for _, r in row_centers]

print(f"検出行数: {len(row_centers)} → {row_centers}")

# 列：14列・左オフセットを少し追加
NUM_COLS = 14
offset = int(w * 0.01)
col_centers_calc = [int(w * (i + 0.5) / NUM_COLS) + offset for i in range(NUM_COLS)]
print(f"計算列数: {len(col_centers_calc)}")

# -----------------------------------------------
# 投影グラフ保存
# -----------------------------------------------
row_thresh_vis = row_proj.max() * 0.3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
ax1.plot(row_proj)
ax1.axhline(row_thresh_vis, color="red", linestyle="--", label="閾値")
for r in row_centers:
    ax1.axvline(r, color="green", linewidth=0.8)
ax1.set_title("行方向投影（緑=検出行）")
ax1.legend()

col_thresh = col_proj.max() * 0.15
ax2.plot(col_proj)
ax2.axhline(col_thresh, color="red", linestyle="--", label="閾値")
for c in col_centers_calc:
    ax2.axvline(c, color="green", linewidth=0.8)
ax2.set_title("列方向（計算値）")
ax2.legend()
plt.tight_layout()
plt.savefig("projection.png", dpi=150)
plt.close()

# -----------------------------------------------
# 結果の可視化
# -----------------------------------------------
vis = img.copy()
cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

for r in row_centers:
    cv2.line(vis, (x, y+r), (x+w, y+r), (0, 255, 255), 1)

for c in col_centers_calc:
    cv2.line(vis, (x+c, y), (x+c, y+h), (255, 100, 0), 1)

cv2.imwrite("keyboard_detect.png", vis)
cv2.imshow("Keyboard Detection", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("保存: keyboard_detect.png")

# -----------------------------------------------
# キーボード座標をJSONに保存
# -----------------------------------------------
import json

# QWERTYの行×列のキー配列（65%キーボード）
KEY_LAYOUT = [
    ["Esc","1","2","3","4","5","6","7","8","9","0","-","=","Backspace"],
    ["Tab","q","w","e","r","t","y","u","i","o","p","[","]","\\"],
    ["CapsLock","a","s","d","f","g","h","j","k","l",";","'","Enter",""],
    ["LShift","z","x","c","v","b","n","m",",",".","/","RShift","",""],
    ["LCtrl","LWin","LAlt"," "," "," "," "," ","RAlt","Fn","RCtrl","","",""],
]

kb_data = {
    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    "row_centers": row_centers,
    "col_centers": col_centers_calc,
    "key_layout": KEY_LAYOUT
}

with open("keyboard_grid.json", "w", encoding="utf-8") as f:
    json.dump(kb_data, f, ensure_ascii=False, indent=2)
print("グリッド保存: keyboard_grid.json")