import json
import numpy as np
from collections import defaultdict

POSE_FILE    = "session_1778410381_pose.json"
KEYLOG_FILE  = "session_1778410381_keys.json"
GRID_FILE    = "keyboard_grid.json"
OUTPUT_FILE  = "session_1778410381_finger_key.json"

# -----------------------------------------------
# データ読み込み
# -----------------------------------------------
with open(POSE_FILE, encoding="utf-8") as f:
    pose_log = json.load(f)

with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)

with open(GRID_FILE, encoding="utf-8") as f:
    grid = json.load(f)

normal_keys = [e for e in keylog if not e["is_backspace"] and not e["key"].startswith("Key.")]
key_times   = [e["timestamp_ms"] for e in normal_keys]

# -----------------------------------------------
# グリッド情報
# -----------------------------------------------
bbox        = grid["bbox"]
row_centers = grid["row_centers"]  # キーボード内の相対y座標
col_centers = grid["col_centers"]  # キーボード内の相対x座標
key_layout  = grid["key_layout"]

kb_x, kb_y = bbox["x"], bbox["y"]
kb_w, kb_h = bbox["w"], bbox["h"]

def pixel_to_key(px, py):
    """
    画像上のピクセル座標(px, py)を
    キーボードグリッドのキー名に変換する
    """
    # キーボード領域内の相対座標
    rx = px - kb_x
    ry = py - kb_y

    # 範囲外チェック
    if rx < 0 or rx > kb_w or ry < 0 or ry > kb_h:
        return None, None, None

    # 最近傍の行・列を探す
    row_idx = int(np.argmin([abs(ry - r) for r in row_centers]))
    col_idx = int(np.argmin([abs(rx - c) for c in col_centers]))

    # キー名取得
    if row_idx < len(key_layout) and col_idx < len(key_layout[row_idx]):
        key_name = key_layout[row_idx][col_idx]
    else:
        key_name = ""

    return row_idx, col_idx, key_name

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
# キーログの各打鍵タイミングに最近傍フレームの
# 指先座標を対応付ける
# -----------------------------------------------
TOLERANCE_MS = 150
results = []

for ki, key_entry in enumerate(normal_keys):
    kt = key_entry["timestamp_ms"]
    key = key_entry["key"]

    # キーログのタイムスタンプに最も近いフレームを探す
    best_frame = None
    best_diff  = 9999
    for frame in pose_log:
        diff = abs(frame["timestamp_ms"] - kt)
        if diff < best_diff:
            best_diff  = diff
            best_frame = frame

    if best_diff > TOLERANCE_MS or best_frame is None:
        continue
    if not best_frame["hands"]:
        continue

    # 全指先の中でキーボード領域内にある指を探す
    candidates = []
    for hand in best_frame["hands"]:
        label = hand["label"]
        for fname, coords in hand["fingertips"].items():
            # pose.jsonのpx/pyをそのまま使う
            px = coords["px"]
            py = coords["py"]
            row_idx, col_idx, key_name = pixel_to_key(px, py)
            if key_name and key_name != "":
                candidates.append({
                    "finger": f"{label}_{fname}",
                    "px": px, "py": py,
                    "detected_key": key_name,
                    "row": row_idx, "col": col_idx
                })

    if not candidates:
        continue

    # キーログのキーと一致するものを優先
    matched = [c for c in candidates if c["detected_key"].lower() == key.lower()]
    chosen  = matched[0] if matched else candidates[0]

    std_finger = STANDARD_FINGER.get(key, "?")
    actual_finger_id = chosen["finger"]  # "Left_index" など
    match = chosen["detected_key"].lower() == key.lower()

    results.append({
        "timestamp_ms":   kt,
        "key":            key,
        "std_finger":     std_finger,
        "std_finger_name": FINGER_NAMES.get(std_finger, "?"),
        "detected_finger": actual_finger_id,
        "detected_key":   chosen["detected_key"],
        "key_match":      match,
        "interval_ms":    key_entry["interval_ms"],
        "is_error":       key_entry["is_error"],
    })

# -----------------------------------------------
# 結果表示
# -----------------------------------------------
print(f"マッチ件数: {len(results)} / {len(normal_keys)}件")
key_match_count = sum(1 for r in results if r["key_match"])
print(f"キー一致率: {key_match_count}/{len(results)} ({key_match_count/len(results)*100:.1f}%)")

print("\n--- サンプル（先頭20件）---")
for r in results[:20]:
    match_str = "✓" if r["key_match"] else "✗"
    print(f"  {match_str} key={r['key']:4s}  "
          f"標準:{r['std_finger_name']}  "
          f"検出:{r['detected_finger']}  "
          f"interval:{r['interval_ms']}ms")

# -----------------------------------------------
# 標準運指と検出指の乖離分析
# -----------------------------------------------
print("\n=== 標準運指 vs 検出指 乖離分析 ===")
deviation = defaultdict(list)
for r in results:
    if r["key_match"]:
        deviation[r["key"]].append({
            "std":    r["std_finger"],
            "actual": r["detected_finger"],
            "match":  r["std_finger"] in r["detected_finger"],
            "interval": r["interval_ms"]
        })

print(f"\n{'キー':4s} {'標準':10s} {'実測一致率':10s} {'平均間隔ms':10s}")
print("-" * 45)
for key, entries in sorted(deviation.items()):
    std = entries[0]["std"]
    match_rate = sum(1 for e in entries if e["match"]) / len(entries)
    avg_int = np.mean([e["interval"] for e in entries])
    print(f"  {key:4s} {FINGER_NAMES.get(std,'?'):10s} "
          f"{match_rate*100:6.1f}%    {avg_int:8.1f}ms")

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存: {OUTPUT_FILE}")