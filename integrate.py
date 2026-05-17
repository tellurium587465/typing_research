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

bbox         = grid["bbox"]
kb_x, kb_y  = bbox["x"], bbox["y"]
kb_w, kb_h  = bbox["w"], bbox["h"]
KEY_POSITIONS = grid["key_positions"]   # {"q": [190, 244], ...} グローバル座標
KEY_RECTS     = grid.get("key_rects", {})  # {"q": {x_min,x_max,y_min,y_max}, ...}
KEY_W_PX      = grid.get("key_w_px", 30.0)
KEY_H_PX      = grid.get("key_h_px", 30.0)

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

# MediaPipe label×fname → 標準コード変換
# 俯瞰カメラ: MediaPipeの Left/Right = 実際の手の左右と一致
FINGER_TO_CODE = {
    ("Left",  "pinky"):  "L1",
    ("Left",  "ring"):   "L2",
    ("Left",  "middle"): "L3",
    ("Left",  "index"):  "L4",
    ("Left",  "thumb"):  "LT",
    ("Right", "index"):  "R4",
    ("Right", "middle"): "R3",
    ("Right", "ring"):   "R2",
    ("Right", "pinky"):  "R1",
    ("Right", "thumb"):  "RT",
}

# 検出対象外の指（使わないと判明しているもの）
EXCLUDED_FINGERS = {("Right", "thumb")}  # 右親指は使わない

# -----------------------------------------------
# pixel_to_key（修正版）
# key_positions のグローバル座標で最近傍キーを返す
# -----------------------------------------------
def pixel_to_key(px, py):
    """ピクセル座標がどのキーの矩形領域に入っているか返す。
    key_rects（keyboard_detect.pyで生成した実寸矩形）を使用。
    複数キーに重なる場合は中心距離が最小のものを返す。"""
    best_key  = None
    best_dist = float("inf")
    if KEY_RECTS:
        for kname, rect in KEY_RECTS.items():
            if rect["x_min"] <= px <= rect["x_max"] and rect["y_min"] <= py <= rect["y_max"]:
                kx, ky = KEY_POSITIONS[kname]
                dist = np.sqrt((px - kx) ** 2 + (py - ky) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_key  = kname
    else:
        # key_rects がない場合は中心±半キーサイズで代替
        half_w = KEY_W_PX / 2
        half_h = KEY_H_PX / 2
        for kname, (kx, ky) in KEY_POSITIONS.items():
            if abs(px - kx) <= half_w and abs(py - ky) <= half_h:
                dist = np.sqrt((px - kx) ** 2 + (py - ky) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_key  = kname
    return best_key

# -----------------------------------------------
# タイムスタンプ基準
# onset_ms と pose の timestamp_ms はどちらも recorder.py 起動基準で同一
# 旧コードの OFFSET_MS 補正（first_key_ts を引く）は誤りだったため削除
# -----------------------------------------------
onset_first = onset_matched[0]["onset_ms"]
pose_times  = [f["timestamp_ms"] for f in pose_log]
pose_first  = min(pose_times)
pose_last   = max(pose_times)

print(f"onset範囲: {onset_first}ms 〜 {onset_matched[-1]['onset_ms']}ms")
print(f"pose範囲:  {pose_first}ms 〜 {pose_last}ms")

# -----------------------------------------------
# 打鍵ごとに pose から使用指を推定
# -----------------------------------------------
SEARCH_WINDOW_MS = 150  # 前後150ms（約±4〜5フレーム@30fps）

results = []

for entry in onset_matched:
    onset_ms   = entry["onset_ms"]
    key        = entry["key"]
    std_finger = STANDARD_FINGER.get(key, "?")
    std_key_pos = KEY_POSITIONS.get(key)

    nearby = [
        f for f in pose_log
        if abs(f["timestamp_ms"] - onset_ms) <= SEARCH_WINDOW_MS
        and f["hands"]
    ]

    if not nearby or std_key_pos is None:
        results.append({
            "onset_ms":           onset_ms,
            "key":                key,
            "std_finger":         std_finger,
            "std_finger_name":    FINGER_NAMES.get(std_finger, "?"),
            "actual_finger":      None,
            "actual_finger_name": None,
            "finger_match":       False,
            "detected_key":       None,
            "key_match":          False,
            "dist_to_key":        None,
            "interval_ms":        entry["interval_ms"],
            "is_error":           entry["is_error"],
            "source":             "std_only"
        })
        continue

    std_kx, std_ky = std_key_pos
    best       = None
    best_score = float("inf")

    for frame in nearby:
        ts_diff = abs(frame["timestamp_ms"] - onset_ms)
        for hand in frame["hands"]:
            label = hand["label"]
            for fname, coords in hand["fingertips"].items():
                # 除外指をスキップ
                if (label, fname) in EXCLUDED_FINGERS:
                    continue
                # 左親指もスペース以外は除外
                if fname == "thumb" and key != " ":
                    continue

                px, py = coords["px"], coords["py"]

                # 打鍵キーの座標から2キー分以内の指先だけを候補にする
                # bboxベースではなくキー中心距離でフィルタ（隣接手の誤混入を防ぐ）
                dist = np.sqrt((px - std_kx) ** 2 + (py - std_ky) ** 2)
                if dist > KEY_W_PX * 2.0:  # 約60px（キー2個分）を超えたら除外
                    continue

                score = dist + ts_diff * 0.5

                if score < best_score:
                    best_score = score
                    finger_code  = FINGER_TO_CODE.get((label, fname), f"{label[0]}_{fname}")
                    detected_key = pixel_to_key(px, py)
                    best = {
                        "finger_code":  finger_code,
                        "finger_label": f"{label}_{fname}",
                        "detected_key": detected_key,
                        "px": px, "py": py,
                        "dist": round(float(dist), 1),
                    }

    if best is None:
        results.append({
            "onset_ms":           onset_ms,
            "key":                key,
            "std_finger":         std_finger,
            "std_finger_name":    FINGER_NAMES.get(std_finger, "?"),
            "actual_finger":      None,
            "actual_finger_name": None,
            "finger_match":       False,
            "detected_key":       None,
            "key_match":          False,
            "dist_to_key":        None,
            "interval_ms":        entry["interval_ms"],
            "is_error":           entry["is_error"],
            "source":             "std_only"
        })
        continue

    finger_code  = best["finger_code"]
    detected_key = best["detected_key"]
    finger_match = (finger_code == std_finger)

    # key_match の判定：key_rects の実寸矩形で判定
    px, py  = best["px"], best["py"]
    if KEY_RECTS and key in KEY_RECTS:
        rect    = KEY_RECTS[key]
        in_rect = rect["x_min"] <= px <= rect["x_max"] and rect["y_min"] <= py <= rect["y_max"]
    else:
        # key_rects がない場合は中心±半キーサイズで代替
        std_kx, std_ky = KEY_POSITIONS[key]
        in_rect = (abs(px - std_kx) <= KEY_W_PX / 2 and abs(py - std_ky) <= KEY_H_PX / 2)
    name_ok   = (detected_key is not None and detected_key.lower() == key.lower())
    key_match = in_rect or name_ok

    results.append({
        "onset_ms":           onset_ms,
        "key":                key,
        "std_finger":         std_finger,
        "std_finger_name":    FINGER_NAMES.get(std_finger, "?"),
        "actual_finger":      finger_code,
        "actual_finger_name": FINGER_NAMES.get(finger_code, finger_code),
        "finger_match":       finger_match,
        "detected_key":       detected_key,
        "key_match":          key_match,
        "dist_to_key":        best["dist"],
        "interval_ms":        entry["interval_ms"],
        "is_error":           entry["is_error"],
        "source":             "camera"
    })

# -----------------------------------------------
# 結果表示
# -----------------------------------------------
camera_results = [r for r in results if r["source"] == "camera"]
key_matches    = [r for r in camera_results if r["key_match"]]
finger_matches = [r for r in camera_results if r["finger_match"]]

print(f"\nカメラで指特定できた件数: {len(camera_results)}/{len(results)}件")
print(f"キー一致率:  {len(key_matches)}/{len(camera_results)}件 "
      f"({len(key_matches)/max(len(camera_results),1)*100:.1f}%)")
print(f"運指一致率:  {len(finger_matches)}/{len(camera_results)}件 "
      f"({len(finger_matches)/max(len(camera_results),1)*100:.1f}%)")

print("\n--- サンプル（先頭20件）---")
for r in results[:20]:
    km  = "✓" if r["key_match"]    else "✗"
    fm  = "✓" if r["finger_match"] else "✗"
    src = "CAM" if r["source"] == "camera" else "STD"
    print(f"  [{src}] key={r['key']:4s} "
          f"標準:{r.get('std_finger_name','?'):8s} "
          f"検出:{str(r.get('actual_finger_name','None')):10s} "
          f"キー{km} 運指{fm} "
          f"dist={str(r.get('dist_to_key',''))+'px':8s} "
          f"interval:{r['interval_ms']}ms")

# -----------------------------------------------
# キーごとの統計
# -----------------------------------------------
print("\n=== キーごとの統計（カメラ検出分）===")
key_stats = defaultdict(list)
for r in camera_results:
    key_stats[r["key"]].append(r)

print(f"{'キー':4s} {'標準運指':10s} {'キー一致':8s} {'運指一致':8s} {'平均dist':9s} {'平均間隔':8s} {'件数':5s}")
print("-" * 65)
for k, entries in sorted(key_stats.items()):
    std      = entries[0]["std_finger_name"]
    km_rate  = sum(1 for e in entries if e["key_match"])    / len(entries)
    fm_rate  = sum(1 for e in entries if e["finger_match"]) / len(entries)
    dists    = [e["dist_to_key"] for e in entries if e["dist_to_key"] is not None]
    avg_dist = np.mean(dists) if dists else float("nan")
    avg_int  = np.mean([e["interval_ms"] for e in entries])
    print(f"  {k:4s} {std:10s} {km_rate*100:6.1f}%  {fm_rate*100:6.1f}%  "
          f"{avg_dist:7.1f}px  {avg_int:7.1f}ms  {len(entries):3d}件")

# -----------------------------------------------
# 実測運指の分布
# -----------------------------------------------
print("\n=== 実測運指の分布 ===")
finger_dist = defaultdict(int)
for r in camera_results:
    finger_dist[r["actual_finger"]] += 1

for code, count in sorted(finger_dist.items(), key=lambda x: -x[1]):
    name = FINGER_NAMES.get(code, code)
    bar  = "█" * (count // 5)
    print(f"  {code:4s} {name:8s}: {count:4d}件 {bar}")

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存: {OUTPUT_FILE}")