"""
report.py  ─ 全セッションの分析結果サマリー表示

使い方:
  python report.py                         # 全セッション
  python report.py --session-id 1778934176 # 特定セッション
  python report.py --detail                # キー別詳細も表示
"""
import argparse
import json
import glob
import os
import re
from collections import defaultdict
import numpy as np

from session_utils import (get_session_files, list_all_session_ids,
                            load_meta, find_session_key_files)
from constants import PHRASE_BOUNDARY_MS, FINGER_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--detail", action="store_true", help="キー別詳細を表示")
args = parser.parse_args()

FINGER_NAMES = {
    "L1": "左小指", "L2": "左薬指", "L3": "左中指", "L4": "左人差",
    "R4": "右人差", "R3": "右中指", "R2": "右薬指", "R1": "右小指",
    "LT": "左親指", "RT": "右親指"
}

def load_integrated(session_id):
    path = f"session_{session_id}_integrated.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_keylog(session_id):
    path = f"session_{session_id}_keys.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_onset(session_id):
    path = f"session_{session_id}_onset_matched.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def analyze_session(session_id):
    data = load_integrated(session_id)
    keys = load_keylog(session_id)
    onset = load_onset(session_id)

    result = {"session_id": session_id}

    # キーログ基本統計
    PHRASE_BOUNDARY_MS = 1000  # フレーズ待機時間の閾値

    if keys:
        normal = [k for k in keys if not k.get("is_backspace") and not k["key"].startswith("Key.")]
        errors = [k for k in normal if k.get("is_error")]
        # フレーズ境界（待機時間を含む interval）を除外して統計計算
        intervals = [k["interval_ms"] for k in normal
                     if 0 < k["interval_ms"] <= PHRASE_BOUNDARY_MS]
        phrase_starts = sum(1 for k in normal if k["interval_ms"] > PHRASE_BOUNDARY_MS)
        total_ms = keys[-1]["timestamp_ms"] if keys else 0
        result["total_keys"] = len(normal)
        result["error_count"] = len(errors)
        result["error_rate"] = len(errors) / max(len(normal), 1) * 100
        result["avg_interval_ms"] = np.mean(intervals) if intervals else 0
        result["phrase_starts"] = phrase_starts
        # WPM: フレーズ境界分の待機時間を除いた純粋な打鍵時間で計算
        pure_typing_ms = total_ms - sum(
            k["interval_ms"] for k in normal if k["interval_ms"] > PHRASE_BOUNDARY_MS
        )
        result["wpm_approx"] = (len(normal) / 5) / (pure_typing_ms / 60000) if pure_typing_ms > 0 else 0
        result["duration_s"] = total_ms / 1000

    # onset マッチング
    if onset is not None:
        n_keys = result.get("total_keys", 0)
        result["onset_count"] = len(onset)
        result["onset_coverage"] = len(onset) / max(n_keys, 1) * 100

    # integrate 結果
    if data:
        total = len(data)
        cam = [r for r in data if r["source"] == "camera"]
        key_ok = [r for r in cam if r.get("key_match")]
        def is_fin(r):
            af = r.get("actual_finger") or r.get("finger_code")
            return af is not None and af == r.get("std_finger")
        fin_ok = [r for r in cam if is_fin(r)]
        dists = [r["dist_to_key"] for r in cam if r.get("dist_to_key") is not None]

        result["total_onsets"] = total
        result["camera_detected"] = len(cam)
        result["camera_rate"] = len(cam) / max(total, 1) * 100
        result["key_match_count"] = len(key_ok)
        result["key_match_rate"] = len(key_ok) / max(len(cam), 1) * 100
        result["finger_match_count"] = len(fin_ok)
        result["finger_match_rate"] = len(fin_ok) / max(len(cam), 1) * 100
        result["avg_dist_px"] = np.mean(dists) if dists else float("nan")
        result["data"] = data

    return result

# セッション一覧
if args.session_id:
    session_ids = [args.session_id]
else:
    session_ids = list_all_session_ids()

print("=" * 70)
print("  Typing Research  全セッション サマリー")
print("=" * 70)

all_results = []
for sid in session_ids:
    r = analyze_session(sid)
    all_results.append(r)

# ─── テーブル表示 ─────────────────────────────────────────────
header = f"{'Session':>12s} {'打鍵':>6s} {'誤打率':>6s} {'WPM':>5s} {'onset%':>7s} {'Cam%':>6s} {'Key%':>6s} {'Fin%':>6s}"
print(header)
print("-" * 70)

for r in all_results:
    sid   = r["session_id"]
    nk    = r.get("total_keys", "-")
    err   = f"{r['error_rate']:.1f}%" if "error_rate" in r else "-"
    wpm   = f"{r['wpm_approx']:.0f}" if "wpm_approx" in r else "-"
    ons   = f"{r['onset_coverage']:.0f}%" if "onset_coverage" in r else "-"
    cam   = f"{r['camera_rate']:.0f}%" if "camera_rate" in r else "-"
    km    = f"{r['key_match_rate']:.0f}%" if "key_match_rate" in r else "-"
    fm    = f"{r['finger_match_rate']:.0f}%" if "finger_match_rate" in r else "-"
    print(f"  {sid:>12s} {str(nk):>6s} {err:>6s} {wpm:>5s} {ons:>7s} {cam:>6s} {km:>6s} {fm:>6s}")

print()
print("凡例: onset%=音響マッチ率  Cam%=カメラ検出率  Key%=キー一致率  Fin%=運指一致率")

# ─── 詳細: 運指ミスの多いキー ──────────────────────────────────
if args.detail:
    for r in all_results:
        if "data" not in r:
            continue
        sid = r["session_id"]
        data = r["data"]
        cam = [x for x in data if x["source"] == "camera"]
        if not cam:
            continue

        print(f"\n--- session {sid}: キー別詳細 ---")
        key_stats = defaultdict(list)
        for x in cam:
            key_stats[x["key"]].append(x)

        print(f"  {'Key':4s} {'std_finger':10s} {'Key%':6s} {'Fin%':6s} {'avg_dist':9s} {'N':4s}")
        print("  " + "-" * 45)
        for k, entries in sorted(key_stats.items()):
            if len(entries) < 3:
                continue
            km = sum(1 for e in entries if e.get("key_match")) / len(entries) * 100
            def is_fin(e):
                af = e.get("actual_finger") or e.get("finger_code")
                return af is not None and af == e.get("std_finger")
            fm = sum(1 for e in entries if is_fin(e)) / len(entries) * 100
            dists = [e["dist_to_key"] for e in entries if e.get("dist_to_key") is not None]
            avg_d = np.mean(dists) if dists else float("nan")
            sf = entries[0].get("std_finger_name", "?")
            flag = " <-- 改善余地あり" if km < 50 else ""
            print(f"  {k:4s} {sf:10s} {km:5.0f}% {fm:5.0f}% {avg_d:7.1f}px {len(entries):4d}{flag}")

# ─── 運指分布サマリー ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  運指使用分布（全セッション合計）")
print("=" * 70)
finger_total = defaultdict(int)
finger_correct = defaultdict(int)
for r in all_results:
    if "data" not in r:
        continue
    for x in r["data"]:
        if x["source"] != "camera":
            continue
        af = x.get("actual_finger")
        if af:
            finger_total[af] += 1
            def is_fin(e):
                af2 = e.get("actual_finger") or e.get("finger_code")
                return af2 is not None and af2 == e.get("std_finger")
            if is_fin(x):
                finger_correct[af] += 1

for code in ["L1","L2","L3","L4","R4","R3","R2","R1"]:
    total = finger_total.get(code, 0)
    if total == 0:
        continue
    correct = finger_correct.get(code, 0)
    rate = correct / total * 100
    bar_len = total // 10
    bar = "#" * bar_len
    name = FINGER_NAMES.get(code, code)
    print(f"  {code} {name:6s}: {total:5d}回 | 運指一致 {rate:5.1f}% | {bar}")

print()
