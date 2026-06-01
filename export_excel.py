import argparse
import json
import glob
import os
import pandas as pd
from collections import defaultdict

from session_utils import get_session_files

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None,
                    help="特定セッションのみ出力（省略時は全keylogファイルを処理）")
args = parser.parse_args()

VOWELS = set("aiueo")

def key_type(k):
    return "母音" if k in VOWELS else "子音"

def vowel_pattern(k1, k2, k3):
    v = lambda k: "V" if k in VOWELS else "C"
    return f"{v(k1)}{v(k2)}{v(k3)}"

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
    "LT":"左親指","RT":"右親指"
}

STANDARD_FINGER = {
    "q":"L1","w":"L2","e":"L3","r":"L4","t":"L4",
    "y":"R4","u":"R4","i":"R3","o":"R2","p":"R1",
    "a":"L1","s":"L2","d":"L3","f":"L4","g":"L4",
    "h":"R4","j":"R4","k":"R3","l":"R2",
    "z":"L1","x":"L2","c":"L3","v":"L4","b":"L4",
    "n":"R4","m":"R4",
    ",":"R3",".":"R2","/":"R1"," ":"RT",
}

def same_hand(f1, f2):
    return f1[0] == f2[0]

if args.session_id:
    # セッション指定: session_*_keys.json を使う
    _sf = get_session_files(args.session_id)
    keylog_files = [_sf["keys"]] if os.path.exists(_sf["keys"]) else []
else:
    # 全ファイルモード: keylog_*.json + session_*_keys.json
    with_errors = glob.glob("keylog_*_with_errors.json")
    base_ids = [f.replace("_with_errors.json","").replace("keylog_","") for f in with_errors]
    keylog_files = with_errors + [
        f for f in glob.glob("keylog_*.json")
        if "with_errors" not in f
        and f.replace("keylog_","").replace(".json","") not in base_ids
    ]
    # session_*_keys.json もマージ
    keylog_files += glob.glob("session_*_keys.json")

print(f"検出したキーログファイル: {len(keylog_files)}件")
for f in keylog_files:
    print(f"  {f}")

all_trigrams_combined = []

import re as _re
for fname in keylog_files:
    # session_*_keys.json 形式と keylog_*.json 形式に両対応
    _m = _re.search(r"session_(\d+)_keys", fname)
    if _m:
        session_id = _m.group(1)
    else:
        session_id = (fname.replace("keylog_","")
                          .replace("_with_errors.json","")
                          .replace(".json",""))

    with open(fname, encoding="utf-8") as f:
        keylog = json.load(f)

    # ── フレーズ境界閾値 ─────────────────────────────────────────
    # 寿司打などでは次のお題が出るまでの待機時間が interval_ms に混入する。
    # この閾値を超えた interval_ms はフレーズ先頭キーとみなし、
    # タイミング統計から除外する（データは保持、フラグで管理）。
    PHRASE_BOUNDARY_MS = 1000  # ms

    normal_keys = [
        e for e in keylog
        if not e.get("is_backspace", False)
        and not e["key"].startswith("Key.")
        and e["key"] in STANDARD_FINGER
    ]

    phrase_starts = 0
    for e in normal_keys:
        e["std_finger"]      = STANDARD_FINGER[e["key"]]
        e["std_finger_name"] = FINGER_NAMES.get(e["std_finger"], "?")
        e["session_id"]      = session_id
        e["boin"]            = e["key"] in VOWELS
        # フレーズ先頭フラグ（待機時間を含む interval を持つキー）
        e["is_phrase_start"] = e["interval_ms"] > PHRASE_BOUNDARY_MS
        if e["is_phrase_start"]:
            phrase_starts += 1

    if phrase_starts > 0:
        print(f"  フレーズ境界を検出: {phrase_starts}件 "
              f"(interval > {PHRASE_BOUNDARY_MS}ms) → タイミング統計から除外")

    LEFT  = {"L1","L2","L3","L4"}
    RIGHT = {"R1","R2","R3","R4"}
    def hand(f):
        if f in LEFT:  return "L"
        if f in RIGHT: return "R"
        return "?"

    trigrams = []
    for i in range(len(normal_keys) - 2):
        a, b, c = normal_keys[i], normal_keys[i+1], normal_keys[i+2]
        fa, fb, fc = a["std_finger"], b["std_finger"], c["std_finger"]

        # フレーズ境界をまたぐ3gramは timing が汚染されるのでフラグを立てる
        # （b か c が is_phrase_start なら interval が不正）
        spans_boundary = b.get("is_phrase_start", False) or c.get("is_phrase_start", False)

        tg = {
            "session_id":        session_id,
            "finger_1":          fa,
            "finger_2":          fb,
            "finger_3":          fc,
            "finger_1_name":     FINGER_NAMES.get(fa, fa),
            "finger_2_name":     FINGER_NAMES.get(fb, fb),
            "finger_3_name":     FINGER_NAMES.get(fc, fc),
            "key_1":             a["key"],
            "key_2":             b["key"],
            "key_3":             c["key"],
            "key_1_type":        key_type(a["key"]),
            "key_2_type":        key_type(b["key"]),
            "key_3_type":        key_type(c["key"]),
            "vowel_count":       sum(1 for k in [a["key"], b["key"], c["key"]] if k in VOWELS),
            "vowel_pattern":     vowel_pattern(a["key"], b["key"], c["key"]),
            "hand_pattern":      f"{hand(fa)}{hand(fb)}{hand(fc)}",
            "interval_1_2_ms":   b["interval_ms"],
            "interval_2_3_ms":   c["interval_ms"],
            "avg_interval_ms":   (b["interval_ms"] + c["interval_ms"]) / 2,
            "has_error":         a.get("is_error", False) or b.get("is_error", False) or c.get("is_error", False),
            "spans_boundary":    spans_boundary,   # フレーズ境界をまたぐか
            "same_finger":       fa == fb or fb == fc,
            "same_hand_all":     same_hand(fa, fb) and same_hand(fb, fc),
            "alternating":       not same_hand(fa, fb) and not same_hand(fb, fc),
            "pattern_type":      ("同指連打" if fa==fb or fb==fc
                                  else "左右交互" if not same_hand(fa,fb) and not same_hand(fb,fc)
                                  else "同手3連" if same_hand(fa,fb) and same_hand(fb,fc)
                                  else "混合"),
        }
        trigrams.append(tg)

    # ── 集計はフレーズ境界をまたがないものだけで行う ─────────────
    trigrams_valid = [tg for tg in trigrams if not tg["spans_boundary"]]
    n_excluded = len(trigrams) - len(trigrams_valid)
    if n_excluded > 0:
        print(f"  タイミング集計から除外した3gram: {n_excluded}件 "
              f"（境界またぎ）、残り {len(trigrams_valid)}件で集計")

    stats = defaultdict(lambda: {"count":0,"total_interval":0,"errors":0,
                                  "same_finger":False,"same_hand":False,
                                  "alternating":False,"pattern_type":""})
    for tg in trigrams_valid:
        key = (tg["finger_1"], tg["finger_2"], tg["finger_3"])
        stats[key]["count"]          += 1
        stats[key]["total_interval"] += tg["avg_interval_ms"]
        stats[key]["errors"]         += int(tg["has_error"])
        stats[key]["same_finger"]    = tg["same_finger"]
        stats[key]["same_hand"]      = tg["same_hand_all"]
        stats[key]["alternating"]    = tg["alternating"]
        stats[key]["pattern_type"]   = tg["pattern_type"]

    stat_rows = []
    for (f1,f2,f3), s in stats.items():
        avg = s["total_interval"] / s["count"]
        stat_rows.append({
            "finger_1": f1, "finger_2": f2, "finger_3": f3,
            "finger_1_name": FINGER_NAMES.get(f1, f1),
            "finger_2_name": FINGER_NAMES.get(f2, f2),
            "finger_3_name": FINGER_NAMES.get(f3, f3),
            "count": s["count"],
            "avg_interval_ms": round(avg, 1),
            "score": round(s["count"] * avg, 1),
            "error_rate": round(s["errors"] / s["count"], 3),
            "same_finger": s["same_finger"],
            "same_hand": s["same_hand"],
            "alternating": s["alternating"],
            "pattern_type": s["pattern_type"],
        })

    df_keys  = pd.DataFrame(normal_keys)
    df_tg    = pd.DataFrame(trigrams)        # 全データ（spans_boundary 列あり）
    df_tg_v  = pd.DataFrame(trigrams_valid)  # 有効データのみ

    df_stats = pd.DataFrame(stat_rows).sort_values("score", ascending=False)

    # サマリーは有効3gramのみで計算（フレーズ境界除外済み）
    summary = df_tg_v.groupby("pattern_type").agg(
        count=("avg_interval_ms","count"),
        avg_interval=("avg_interval_ms","mean"),
        std_interval=("avg_interval_ms","std"),
    ).reset_index() if not df_tg_v.empty else pd.DataFrame()

    vowel_summary = df_tg_v.groupby("vowel_pattern").agg(
        count=("avg_interval_ms","count"),
        avg_interval=("avg_interval_ms","mean"),
        std_interval=("avg_interval_ms","std"),
    ).reset_index() if not df_tg_v.empty else pd.DataFrame()

    output_file = f"typing_analysis_{session_id}.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_keys.to_excel(writer, sheet_name="打鍵ログ", index=False)
        df_tg.to_excel(writer, sheet_name="3gram_全データ", index=False)
        if not df_tg_v.empty:
            df_tg_v.to_excel(writer, sheet_name="3gram_有効のみ", index=False)
        df_stats.to_excel(writer, sheet_name="3gram_集計", index=False)
        if not summary.empty:
            summary.to_excel(writer, sheet_name="パターン別サマリ", index=False)
        if not vowel_summary.empty:
            vowel_summary.to_excel(writer, sheet_name="母音パターン別サマリ", index=False)

    # JASP用CSV（フレーズ境界除外済みデータを出力）
    csv_file = f"jasp_trigram_{session_id}.csv"
    df_tg_v.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"保存: {output_file}（打鍵:{len(normal_keys)}件 "
          f"3gram全:{len(trigrams)}件 有効:{len(trigrams_valid)}件）")
    print(f"JASP用CSV（境界除外済み）: {csv_file}")

    all_trigrams_combined.append(df_tg_v)

# 全セッション統合CSV
if all_trigrams_combined:
    df_all = pd.concat(all_trigrams_combined, ignore_index=True)
    combined_csv = "jasp_trigram_ALL.csv"
    df_all.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    print(f"\n全セッション統合CSV: {combined_csv}（{len(df_all)}件）")
    print(f"hand_patternの分布:")
    print(df_all["hand_pattern"].value_counts().sort_index())