import json
import numpy as np
from collections import defaultdict

import argparse
from session_utils import get_session_files
from finger_select import select_finger, classify_quality
import os

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--use-z", action="store_true",
                    help="z深度の押下キューを使う（実験的・validate_finger.pyで検証してから）")
parser.add_argument("--reorder", action="store_true",
                    help="指先x順序でR3↔R2取り違えを補正（実験的・要検証）")
args = parser.parse_args()

_sf = get_session_files(args.session_id)
ONSET_FILE  = _sf["onset"]
POSE_FILE   = _sf["pose"]
OUTPUT_FILE = _sf["integrated"]

# セッション別グリッドを優先、なければ共通グリッドにフォールバック
_session_grid = _sf["keyboard_grid"]
GRID_FILE = _session_grid if os.path.exists(_session_grid) else "keyboard_grid.json"
print(f"キーボードグリッド: {GRID_FILE}")

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
    """ピクセル座標に最も近いキーを返す。
    ① key_rects 内に入っているキーがあれば最近傍を返す（高精度）
    ② key_rects に入っていない場合でも、最近傍キーを返す（MAX_FALLBACK_PX 以内）
    """
    MAX_FALLBACK_PX = KEY_W_PX * 1.5   # rect外でもここまでは最近傍を返す

    best_in_rect = None
    best_in_dist = float("inf")
    best_near    = None
    best_near_d  = float("inf")

    if KEY_RECTS:
        for kname, rect in KEY_RECTS.items():
            kx, ky = KEY_POSITIONS[kname]
            dist = np.sqrt((px - kx) ** 2 + (py - ky) ** 2)
            in_rect = (rect["x_min"] <= px <= rect["x_max"] and
                       rect["y_min"] <= py <= rect["y_max"])
            if in_rect and dist < best_in_dist:
                best_in_dist = dist
                best_in_rect = kname
            if dist < best_near_d:
                best_near_d = dist
                best_near   = kname
    else:
        for kname, (kx, ky) in KEY_POSITIONS.items():
            dist = np.sqrt((px - kx) ** 2 + (py - ky) ** 2)
            if dist < best_near_d:
                best_near_d = dist
                best_near   = kname

    if best_in_rect is not None:
        return best_in_rect
    if best_near_d <= MAX_FALLBACK_PX:
        return best_near
    return None

# -----------------------------------------------
# タイムスタンプ基準チェック
# onset_ms と pose の timestamp_ms はどちらも recorder.py 起動基準で同一
# -----------------------------------------------
if not onset_matched:
    print("onset_matched.json が空のため、キーログのタイムスタンプを代用します。")
    print("（音声マッチング精度はないがカメラによる指特定は試みる）")
    # キーログ直接で onset_matched 形式に変換
    import json as _json
    with open(_sf["keys"], encoding="utf-8") as _f:
        _keylog = _json.load(_f)
    _normal = [e for e in _keylog if not e.get("is_backspace") and not e["key"].startswith("Key.")]
    onset_matched = [{
        "onset_ms":    e["timestamp_ms"],
        "key_ms":      e["timestamp_ms"],
        "diff_ms":     0,
        "key":         e["key"],
        "is_error":    e.get("is_error", False),
        "interval_ms": e["interval_ms"],
    } for e in _normal]
    print(f"キーログからonset代用: {len(onset_matched)}件")

onset_first = onset_matched[0]["onset_ms"]
pose_times  = [f["timestamp_ms"] for f in pose_log]
pose_first  = min(pose_times)
pose_last   = max(pose_times)

print(f"onset範囲: {onset_first}ms 〜 {onset_matched[-1]['onset_ms']}ms")
print(f"pose範囲:  {pose_first}ms 〜 {pose_last}ms")

# -----------------------------------------------
# 打鍵ごとに pose から使用指を推定
#   多フレーム投票＋信頼度＋品質分類（finger_select.py）
# -----------------------------------------------
SEARCH_WINDOW_MS = 150  # 前後150ms（約±4〜5フレーム@30fps）

def _std_only(entry, key, std_finger):
    """カメラで指特定できなかった打鍵の結果（標準運指のみ）"""
    return {
        "onset_ms":            entry["onset_ms"],
        "key":                 key,
        "std_finger":          std_finger,
        "std_finger_name":     FINGER_NAMES.get(std_finger, "?"),
        "actual_finger":       None,
        "actual_finger_name":  None,
        "finger_match":        False,
        "detected_key":        None,
        "key_match":           False,
        "dist_to_key":         None,
        "detection_confidence": None,
        "vote_margin":          None,
        "finger_quality":       "no_detection",
        "n_candidate_frames":   0,
        "interval_ms":         entry["interval_ms"],
        "is_error":            entry["is_error"],
        "source":              "std_only",
    }

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
        results.append(_std_only(entry, key, std_finger))
        continue

    std_kx, std_ky = std_key_pos
    sel = select_finger(
        nearby, onset_ms, std_kx, std_ky, key, SEARCH_WINDOW_MS,
        KEY_W_PX, FINGER_TO_CODE, EXCLUDED_FINGERS, pixel_to_key,
        use_z=args.use_z, reorder=args.reorder)

    if sel is None:
        results.append(_std_only(entry, key, std_finger))
        continue

    finger_code  = sel["finger_code"]
    detected_key = sel["detected_key"]
    finger_match = (finger_code == std_finger)

    # key_match の判定：key_rects の実寸矩形で判定
    px, py = sel["px"], sel["py"]
    if KEY_RECTS and key in KEY_RECTS:
        rect    = KEY_RECTS[key]
        in_rect = rect["x_min"] <= px <= rect["x_max"] and rect["y_min"] <= py <= rect["y_max"]
    else:
        # key_rects がない場合は中心±半キーサイズで代替
        std_kx, std_ky = KEY_POSITIONS[key]
        in_rect = (abs(px - std_kx) <= KEY_W_PX / 2 and abs(py - std_ky) <= KEY_H_PX / 2)
    name_ok   = (detected_key is not None and detected_key.lower() == key.lower())
    key_match = in_rect or name_ok

    # 品質分類：高信頼かつ標準一致=standard / 高信頼だが不一致=nonstandard（癖）
    #           低信頼=uncertain（誤検出疑い）。"癖" と "ノイズ" を分離する。
    quality = classify_quality(sel["confidence"], sel["margin"], finger_match)

    results.append({
        "onset_ms":            onset_ms,
        "key":                 key,
        "std_finger":          std_finger,
        "std_finger_name":     FINGER_NAMES.get(std_finger, "?"),
        "actual_finger":       finger_code,
        "actual_finger_name":  FINGER_NAMES.get(finger_code, finger_code),
        "finger_match":        finger_match,
        "detected_key":        detected_key,
        "key_match":           key_match,
        "dist_to_key":         sel["dist"],
        "detection_confidence": sel["confidence"],
        "vote_margin":          sel["margin"],
        "finger_quality":       quality,
        "n_candidate_frames":   sel["n_frames"],
        "interval_ms":         entry["interval_ms"],
        "is_error":            entry["is_error"],
        "source":              "camera",
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
print(f"運指一致率(全カメラ): {len(finger_matches)}/{len(camera_results)}件 "
      f"({len(finger_matches)/max(len(camera_results),1)*100:.1f}%)")

# ── 検出品質の内訳（"本人の癖" と "誤検出" を分離）─────────────────
q_std    = [r for r in camera_results if r.get("finger_quality") == "standard"]
q_nonstd = [r for r in camera_results if r.get("finger_quality") == "nonstandard"]
q_unc    = [r for r in camera_results if r.get("finger_quality") == "uncertain"]
n_cam    = max(len(camera_results), 1)
confident = len(q_std) + len(q_nonstd)
print("\n--- 検出品質の内訳 ---")
print(f"  standard   (高信頼・標準運指)     : {len(q_std):4d}件 ({len(q_std)/n_cam*100:.1f}%)")
print(f"  nonstandard(高信頼・本人の運指の癖): {len(q_nonstd):4d}件 ({len(q_nonstd)/n_cam*100:.1f}%)")
print(f"  uncertain  (低信頼・誤検出の疑い)  : {len(q_unc):4d}件 ({len(q_unc)/n_cam*100:.1f}%)")
if confident:
    print(f"\n  妥当な運指一致率（uncertain除外）: "
          f"{len(q_std)}/{confident}件 ({len(q_std)/confident*100:.1f}%)")
    print(f"  → 残り {len(q_nonstd)/confident*100:.1f}% は誤検出ではなく "
          f"本人が標準と違う指で打っている（運指の癖）")
avg_conf = np.mean([r["detection_confidence"] for r in camera_results
                    if r.get("detection_confidence") is not None]) if camera_results else 0
print(f"  平均検出信頼度: {avg_conf:.3f}")

print("\n--- サンプル（先頭20件）---")
for r in results[:20]:
    km  = "OK" if r["key_match"]    else "NG"
    fm  = "OK" if r["finger_match"] else "NG"
    src = "CAM" if r["source"] == "camera" else "STD"
    print(f"  [{src}] key={r['key']:4s} "
          f"std:{r.get('std_finger_name','?'):8s} "
          f"det:{str(r.get('actual_finger_name','None')):10s} "
          f"key={km} fin={fm} "
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
    bar  = "#" * (count // 5)
    print(f"  {code:4s} {name:8s}: {count:4d} {bar}")

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存: {OUTPUT_FILE}")