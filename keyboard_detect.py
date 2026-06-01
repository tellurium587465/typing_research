import argparse
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from session_utils import get_session_files

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--headless", action="store_true", help="ウィンドウ表示なし（パイプライン用）")
parser.add_argument("--image", default="frame_10.png", help="キーボード検出用フレーム画像")
args = parser.parse_args()

_sf = get_session_files(args.session_id)
SESSION_ID = _sf["session_id"]
IMAGE_FILE = args.image
GRID_OUTPUT = _sf["keyboard_grid"]   # セッション別グリッドファイル

# -----------------------------------------------
# Rainy75 物理レイアウト定義（75%キーボード）
# キーピッチ: 19.05mm、全幅: 317.5mm
# -----------------------------------------------
# 各行のキー定義: (キー名, 幅の倍率)
# 標準キー幅 = 1.0u = 19.05mm
ROWS = [
    # Fn行
    [("Esc",1.0),("F1",1.0),("F2",1.0),("F3",1.0),("F4",1.0),
     ("F5",1.0),("F6",1.0),("F7",1.0),("F8",1.0),("F9",1.0),
     ("F10",1.0),("F11",1.0),("F12",1.0),("Delete",1.0),("Home",1.0)],
    # 数字行
    [("`",1.0),("1",1.0),("2",1.0),("3",1.0),("4",1.0),("5",1.0),
     ("6",1.0),("7",1.0),("8",1.0),("9",1.0),("0",1.0),("-",1.0),
     ("=",1.0),("Backspace",2.0),("End",1.0)],
    # QWERTY行
    [("Tab",1.5),("q",1.0),("w",1.0),("e",1.0),("r",1.0),("t",1.0),
     ("y",1.0),("u",1.0),("i",1.0),("o",1.0),("p",1.0),("[",1.0),
     ("]",1.0),("\\",1.5),("PgUp",1.0)],
    # ASDF行
    [("CapsLock",1.75),("a",1.0),("s",1.0),("d",1.0),("f",1.0),("g",1.0),
     ("h",1.0),("j",1.0),("k",1.0),("l",1.0),(";",1.0),("'",1.0),
     ("Enter",2.25),("PgDn",1.0)],
    # ZXCV行
    [("LShift",2.25),("z",1.0),("x",1.0),("c",1.0),("v",1.0),("b",1.0),
     ("n",1.0),("m",1.0),(",",1.0),(".",1.0),("/",1.0),
     ("RShift",1.75),("Up",1.0),("RCtrl",1.0)],
    # スペース行
    [("LCtrl",1.25),("Win",1.25),("LAlt",1.25),(" ",6.25),
     ("RAlt",1.25),("Left",1.0),("Down",1.0),("Right",1.0)],
]

# 各行のキー中心x座標を計算（単位：u）
def calc_row_centers(row):
    centers = []
    x = 0.0
    for key, width in row:
        centers.append((key, x + width/2))
        x += width
    return centers, x  # centers, 行の総幅

row_centers_u = []  # 各行の(キー名, x中心[u])リスト
row_widths = []
for row in ROWS:
    centers, total_w = calc_row_centers(row)
    row_centers_u.append(centers)
    row_widths.append(total_w)

max_width_u = max(row_widths)

# -----------------------------------------------
# 画像読み込み・キーボード領域検出
# -----------------------------------------------
img = cv2.imread(IMAGE_FILE)
h_img, w_img = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower1 = np.array([125, 20, 80])
upper1 = np.array([165, 255, 255])
mask1  = cv2.inRange(hsv, lower1, upper1)
lower2 = np.array([160, 20, 80])
upper2 = np.array([180, 255, 255])
mask2  = cv2.inRange(hsv, lower2, upper2)
mask   = cv2.bitwise_or(mask1, mask2)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
cv2.imwrite("debug_mask.png", mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("キーボード領域が検出できません")
    exit()

largest  = max(contours, key=cv2.contourArea)
bx, by, bw, bh = cv2.boundingRect(largest)

# ── ホモグラフィ補正（傾きが5°以上の場合のみ） ─────────────────────
# minAreaRect で実際の傾き角を計測し、閾値以上なら透視補正を適用
HOMOGRAPHY_ANGLE_THRESH = 5.0  # degrees

rect_min  = cv2.minAreaRect(largest)
tilt_deg  = abs(rect_min[2])                 # 0～90° の傾き
# OpenCV の minAreaRect は -90°～0° を返すので正規化
if tilt_deg > 45:
    tilt_deg = 90 - tilt_deg
print(f"キーボード傾き角: {tilt_deg:.1f}°")

M_inv = None

if tilt_deg >= HOMOGRAPHY_ANGLE_THRESH:
    # minAreaRect の4頂点を使ってホモグラフィを計算（精度が高い）
    corners = cv2.boxPoints(rect_min).astype(np.float32)
    # 時計回りに並び替え（左上・右上・右下・左下）
    s    = corners.sum(axis=1)
    diff = corners[:, 0] - corners[:, 1]
    rect = np.float32([
        corners[np.argmin(s)],
        corners[np.argmax(diff)],
        corners[np.argmax(s)],
        corners[np.argmin(diff)],
    ])
    DST_W   = int(bw)
    DST_H   = int(bh)
    pts_dst = np.float32([[0,0],[DST_W,0],[DST_W,DST_H],[0,DST_H]])
    M       = cv2.getPerspectiveTransform(rect, pts_dst)
    M_inv   = cv2.getPerspectiveTransform(pts_dst, rect)
    print(f"ホモグラフィ補正: 適用（傾き{tilt_deg:.1f}° ≥ {HOMOGRAPHY_ANGLE_THRESH}°）")
    # 補正後の画像でbboxを再設定
    bx, by, bw, bh = 0, 0, DST_W, DST_H
else:
    M     = None
    M_inv = None
    print(f"ホモグラフィ補正: スキップ（傾き{tilt_deg:.1f}° < {HOMOGRAPHY_ANGLE_THRESH}°、補正不要）")

# bboxを少し内側に補正（枠のベゼル分を除去）
BEZEL_X = 0               # 左右の余白なし
BEZEL_Y = int(bh * 0.02)  # 上下の余白（2%）
bx += BEZEL_X
by += BEZEL_Y
bw -= BEZEL_X * 2
bh -= BEZEL_Y * 2

print(f"補正後領域: x={bx}, y={by}, w={bw}, h={bh}")

# -----------------------------------------------
# 物理寸法比率でキーの中心座標を計算
# -----------------------------------------------
num_rows = len(ROWS)
key_h_px = bh / num_rows          # 1行の高さ[px]
key_w_px = bw / max_width_u       # 1u の幅[px]

def to_orig(cx, cy):
    """補正後座標→元画像座標に変換"""
    if M_inv is None:
        return int(cx), int(cy)
    pt = np.float32([[[cx, cy]]])
    r  = cv2.perspectiveTransform(pt, M_inv)
    return int(r[0][0][0]), int(r[0][0][1])

# 各キーのピクセル座標を計算
key_positions = {}

for row_idx, centers in enumerate(row_centers_u):
    cy = by + (row_idx + 0.5) * key_h_px
    for key, cx_u in centers:
        cx = bx + cx_u * key_w_px
        ox, oy = to_orig(cx, cy)
        key_positions[key] = (ox, oy)

# 行中心y座標（元画像座標）
row_y_centers = []
for i in range(num_rows):
    cy = by + (i + 0.5) * key_h_px
    _, oy = to_orig(bw//2, cy)
    row_y_centers.append(oy)

# 行ごとのキー中心x座標リスト（元画像座標）
col_centers_by_row = []
for row_idx, centers in enumerate(row_centers_u):
    cy = by + (row_idx + 0.5) * key_h_px
    cols = []
    for _, cx_u in centers:
        cx = bx + cx_u * key_w_px
        ox, _ = to_orig(cx, cy)
        cols.append(ox)
    col_centers_by_row.append(cols)

# bboxを元画像座標で再計算
if M_inv is not None:
    corners = [(0,0),(bw,0),(bw,bh),(0,bh)]
    orig_corners = [to_orig(x,y) for x,y in corners]
    xs = [p[0] for p in orig_corners]
    ys = [p[1] for p in orig_corners]
    orig_bx = min(xs); orig_by = min(ys)
    orig_bw = max(xs)-min(xs); orig_bh = max(ys)-min(ys)
else:
    orig_bx, orig_by, orig_bw, orig_bh = bx, by, bw, bh

# QWERTYキーレイアウト（文字列のみ）
KEY_LAYOUT = [[key for key, _ in row] for row in ROWS]

# -----------------------------------------------
# 可視化
# -----------------------------------------------
vis = img.copy()
cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

for key, (cx, cy) in key_positions.items():
    cv2.circle(vis, (cx, cy), 4, (255, 100, 0), -1)
    if len(key) <= 2:  # 短いキー名だけ表示
        cv2.putText(vis, key.upper(), (cx-8, cy+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.imwrite("keyboard_detect.png", vis)
print(f"検出キー数: {len(key_positions)}個")
if not args.headless:
    cv2.imshow("Keyboard Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------
# 各キーの矩形領域を計算（x_min, x_max, y_min, y_max）
# -----------------------------------------------
key_rects = {}

for row_idx, row in enumerate(ROWS):
    cy = by + (row_idx + 0.5) * key_h_px
    y_min_local = by + row_idx * key_h_px
    y_max_local = by + (row_idx + 1) * key_h_px
    x = 0.0
    for key, width in row:
        x_min_local = bx + x * key_w_px
        x_max_local = bx + (x + width) * key_w_px
        # 元画像座標に変換
        ox_min, oy_min = to_orig(x_min_local, y_min_local)
        ox_max, oy_max = to_orig(x_max_local, y_max_local)
        MARGIN = 8  # キー矩形を各方向に拡張（隣接キーのヒット率改善）
        key_rects[key] = {
            "x_min": ox_min - MARGIN, "x_max": ox_max + MARGIN,
            "y_min": oy_min - MARGIN, "y_max": oy_max + MARGIN
        }
        x += width

# -----------------------------------------------
# JSON保存
# -----------------------------------------------
kb_data = {
    "bbox": {"x": int(orig_bx), "y": int(orig_by), "w": int(orig_bw), "h": int(orig_bh)},
    "row_centers": row_y_centers,
    "col_centers": col_centers_by_row[2],
    "col_centers_by_row": col_centers_by_row,
    "key_positions": {k: list(v) for k, v in key_positions.items()},
    "key_rects": key_rects,
    "key_layout": KEY_LAYOUT,
    "key_w_px": round(key_w_px, 2),
    "key_h_px": round(key_h_px, 2),
    "homography": M_inv.tolist() if M_inv is not None else None,
}

with open(GRID_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(kb_data, f, ensure_ascii=False, indent=2)
print(f"保存: {GRID_OUTPUT}")

# 後方互換: keyboard_grid.json にも保存（integrate.py 等のフォールバック用）
with open("keyboard_grid.json", "w", encoding="utf-8") as f:
    json.dump(kb_data, f, ensure_ascii=False, indent=2)
print("保存: keyboard_grid.json（後方互換）")