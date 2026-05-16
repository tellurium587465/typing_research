import json
import numpy as np
from collections import defaultdict

from session_utils import get_session_files
_sf = get_session_files()
ONSET_FILE  = _sf["onset"]
POSE_FILE   = _sf["pose"]
GRID_FILE   = "keyboard_grid.json"
OUTPUT_FILE = _sf["integrated"]

# -----------------------------------------------
# データ読み込み
# -----------------------------------------------
with open(ONSET_FILE, encoding="utf-8") as f:
    onset_matched = json.load(f)

with open(POSE_FILE, encoding="utf-8") as f:
    pose_log = json.load(f)

with open(GRID_FILE, encoding="utf-8") as f:
    grid = json.load(f)

bbox        = grid["bbox"]
row_centers = grid["row_centers"]
col_centers = grid["col_centers"]
key_layout  = grid["key_layout"]
kb_x, kb_y = bbox["x"], bbox["y"]
kb_w, kb_h  = bbox["w"], bbox["h"]

# -----------------------------------------------
# タイムスタンプのオフセット補正
# -----------------------------------------------
# onsetはキーログ基準（録音開始からの相対時刻）
# poseは動画基準（動画開始からの相対時刻）
# recorder.pyでは録音・動画・キーロガーを同時起動しているので
# キーロガーの最初の打鍵時刻とposeの対応するフレームを使ってオフセットを推定

# onsetの最初の時刻とposeの時刻範囲を確認
onset_first = onset_matched[0]["onset_ms"]
pose_times  = [f["timestamp_ms"] for f in pose_log]
pose_first  = min(pose_times)
pose_last   = max(pose_times)

print(f"onset範囲: {onset_first}ms 〜 {onset_matched[-1]['onset_ms']}ms")
print(f"pose範囲:  {pose_first}ms 〜 {pose_last}ms")

# onsetの時刻をpose基準に変換するオフセット
# onsetはキーログ基準なので、キーログの最初の打鍵時刻を引いてpose基準に合わせる
with open(ONSET_FILE.replace("onset_matched", "keys"), encoding="utf-8") as f:
    keylog_all = json.load(f)
first_key_ts = min(e["timestamp_ms"] for e in keylog_all if not e["is_backspace"] and not e["key"].startswith("Key."))
print(f"キーログ最初の打鍵: {first_key_ts}ms")

# オフセット = onset基準時刻 - pose基準時刻の差
# onsetはキーログ基準なのでそのまま使える（recorder.pyで同時起動しているため）
# poseはフレーム番号×(1/fps)×1000で計算されているので同じ基準のはず
# → ずれがあれば手動で補正
OFFSET_MS = first_key_ts  # キーログ最初の打鍵時刻でオフセット補正

print(f"適用オフセット: {OFFSET_MS}ms")

# -----------------------------------------------
# 標準運指テーブル
# -----------------------------------------------
STANDARD_FINGER = {
    "q":"L1","w":"L2","e":"L3","r":"L4","t":"L4",
    "y":"R4","u":"R4","i":"R3","o":"R2","p":"R1",
    "a":"L1","s":"L2","d":"L3","f":"L4","g":"L4",
    "h":"R4","j":"R4","k":"R3","l":"R2",
    "z":"L1","x":"L2","c":"L3","v":"L4","b":"L4",
    "n":"R4","m":"R4",
    ",":"R3",".":"R2","/":"R1"," ":"RT",
}
FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
    "LT":"左親指","RT":"右親指"
}

# -----------------------------------------------
# pixel_to_key：ピクセル座標→キー名
# -----------------------------------------------
def pixel_to_key(px, py):
    rx = px - kb_x
    ry = py - kb_y
    if rx < 0 or rx > kb_w or ry < 0 or ry > kb_h:
        return None, None, None
    row_idx = int(np.argmin([abs(ry - r) for r in row_centers]))
    col_idx = int(np.argmin([abs(rx - c) for c in col_centers]))
    if row_idx < len(key_layout) and col_idx < len(key_layout[row_idx]):
        return row_idx, col_idx, key_layout[row_idx][col_idx]
    return None, None, None

# -----------------------------------------------
# onset時刻で骨格フレームを検索し最近傍指を特定
# -----------------------------------------------
SEARCH_WINDOW_MS = 300  # 探索範囲を広げる

results = []

for entry in onset_matched:
    onset_ms = entry["onset_ms"] - OFFSET_MS  # オフセット補正
    key      = entry["key"]
    std_finger = STANDARD_FINGER.get(key, "?")

    # onset時刻周辺のフレームを収集
    nearby = [
        f for f in pose_log
        if abs(f["timestamp_ms"] - onset_ms) <= SEARCH_WINDOW_MS
        and f["hands"]
    ]

    if not nearby:
        results.append({
            "onset_ms":      onset_ms,
            "key":           key,
            "std_finger":    std_finger,
            "actual_finger": None,
            "dist_to_key":   None,
            "key_match":     False,
            "interval_ms":   entry["interval_ms"],
            "is_error":      entry["is_error"],
            "source":        "std_only"
        })
        continue

    # 全フレームの全指先を収集し、キーボード領域内で最もキーに近い指を探す
    best = None
    best_dist = 9999

    for frame in nearby:
        ts_diff = abs(frame["timestamp_ms"] - onset_ms)
        for hand in frame["hands"]:
            label = hand["label"]
            for fname, coords in hand["fingertips"].items():
                px, py = coords["px"], coords["py"]
                rx = px - kb_x
                ry = py - kb_y
                if rx < 0 or rx > kb_w or ry < 0 or ry > kb_h:
                    continue

                # 標準運指のキーグリッド座標との距離
                row_idx, col_idx, det_key = pixel_to_key(px, py)
                if row_idx is None:
                    continue

                # 標準運指のグリッド位置との距離
                std_col = col_centers[col_idx] if col_idx < len(col_centers) else 0
                std_row = row_centers[row_idx] if row_idx < len(row_centers) else 0
                dist = np.sqrt((rx - std_col)**2 + (ry - std_row)**2)

                # タイムスタンプの近さも加味
                score = dist + ts_diff * 0.5

                if score < best_dist:
                    best_dist = score
                    best = {
                        "finger":       f"{label}_{fname}",
                        "detected_key": det_key,
                        "px": px, "py": py,
                        "dist": round(float(dist), 1)
                    }

    if best is None:
        actual_finger = None
        key_match     = False
        source        = "std_only"
        dist          = None
    else:
        actual_finger = best["finger"]
        key_match     = best["detected_key"].lower() == key.lower()
        source        = "camera"
        dist          = best["dist"]

    results.append({
        "onset_ms":      onset_ms,
        "key":           key,
        "std_finger":    std_finger,
        "std_finger_name": FINGER_NAMES.get(std_finger, "?"),
        "actual_finger": actual_finger,
        "key_match":     key_match,
        "dist_to_key":   dist,
        "interval_ms":   entry["interval_ms"],
        "is_error":      entry["is_error"],
        "source":        source
    })

# -----------------------------------------------
# 結果表示
# -----------------------------------------------
camera_results = [r for r in results if r["source"] == "camera"]
key_matches    = [r for r in camera_results if r["key_match"]]

print(f"\nカメラで指特定できた件数: {len(camera_results)}/{len(results)}件")
print(f"キー一致率: {len(key_matches)}/{len(camera_results)}件 "
      f"({len(key_matches)/max(len(camera_results),1)*100:.1f}%)")

print("\n--- サンプル（先頭20件）---")
for r in results[:20]:
    match_str = "✓" if r["key_match"] else "✗"
    src = "CAM" if r["source"] == "camera" else "STD"
    std_name = r.get('std_finger_name') or '?'
    print(f"  [{src}]{match_str} key={r['key']:4s} "
          f"標準:{std_name:8s} "
          f"検出:{str(r['actual_finger']):20s} "
          f"interval:{r['interval_ms']}ms")

# -----------------------------------------------
# キーごとの標準運指一致率
# -----------------------------------------------
print("\n=== キーごとの標準運指一致率（カメラ検出分）===")
key_stats = defaultdict(list)
for r in camera_results:
    key_stats[r["key"]].append(r)

print(f"{'キー':4s} {'標準運指':10s} {'一致率':8s} {'平均間隔ms':10s} {'件数':5s}")
print("-" * 50)
for k, entries in sorted(key_stats.items()):
    std = entries[0]["std_finger_name"]
    match_rate = sum(1 for e in entries if e["key_match"]) / len(entries)
    avg_int = np.mean([e["interval_ms"] for e in entries])
    print(f"  {k:4s} {std:10s} {match_rate*100:6.1f}%  {avg_int:8.1f}ms  {len(entries):3d}件")

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存: {OUTPUT_FILE}")