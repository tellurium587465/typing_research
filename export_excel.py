import json
import glob
import pandas as pd
from collections import defaultdict

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

keylog_files = glob.glob("keylog_*_with_errors.json") + glob.glob("keylog_*.json")
# with_errorsがあれば通常版は除外
with_errors = glob.glob("keylog_*_with_errors.json")
base_ids = [f.replace("_with_errors.json","").replace("keylog_","") for f in with_errors]
keylog_files = with_errors + [
    f for f in glob.glob("keylog_*.json")
    if "with_errors" not in f
    and f.replace("keylog_","").replace(".json","") not in base_ids
]

print(f"検出したキーログファイル: {len(keylog_files)}件")
for f in keylog_files:
    print(f"  {f}")

all_trigrams_combined = []

for fname in keylog_files:
    session_id = (fname.replace("keylog_","")
                      .replace("_with_errors.json","")
                      .replace(".json",""))

    with open(fname, encoding="utf-8") as f:
        keylog = json.load(f)

    normal_keys = [
        e for e in keylog
        if not e.get("is_backspace", False)
        and not e["key"].startswith("Key.")
        and e["key"] in STANDARD_FINGER
    ]

    for e in normal_keys:
        e["std_finger"]      = STANDARD_FINGER[e["key"]]
        e["std_finger_name"] = FINGER_NAMES.get(e["std_finger"], "?")
        e["session_id"]      = session_id
        e["boin"]            = e["key"] in VOWELS

    trigrams = []
    for i in range(len(normal_keys) - 2):
        a, b, c = normal_keys[i], normal_keys[i+1], normal_keys[i+2]
        fa, fb, fc = a["std_finger"], b["std_finger"], c["std_finger"]
        LEFT  = {"L1","L2","L3","L4"}
        RIGHT = {"R1","R2","R3","R4"}
        def hand(f):
            if f in LEFT:  return "L"
            if f in RIGHT: return "R"
            return "?"

        tg = {
            "session_id":      session_id,
            "finger_1":        fa,
            "finger_2":        fb,
            "finger_3":        fc,
            "finger_1_name":   FINGER_NAMES.get(fa, fa),
            "finger_2_name":   FINGER_NAMES.get(fb, fb),
            "finger_3_name":   FINGER_NAMES.get(fc, fc),
            "key_1":           a["key"],
            "key_2":           b["key"],
            "key_3":           c["key"],
            "key_1_type":      key_type(a["key"]),
            "key_2_type":      key_type(b["key"]),
            "key_3_type":      key_type(c["key"]),
            "vowel_count":     sum(1 for k in [a["key"], b["key"], c["key"]] if k in VOWELS),
            "vowel_pattern":   vowel_pattern(a["key"], b["key"], c["key"]),
            "hand_pattern":    f"{hand(fa)}{hand(fb)}{hand(fc)}",
            "interval_1_2_ms": b["interval_ms"],
            "interval_2_3_ms": c["interval_ms"],
            "avg_interval_ms": (b["interval_ms"] + c["interval_ms"]) / 2,
            "has_error":       a.get("is_error", False) or b.get("is_error", False) or c.get("is_error", False),
            "same_finger":     fa == fb or fb == fc,
            "same_hand_all":   same_hand(fa, fb) and same_hand(fb, fc),
            "alternating":     not same_hand(fa, fb) and not same_hand(fb, fc),
            "pattern_type":    ("同指連打" if fa==fb or fb==fc
                                else "左右交互" if not same_hand(fa,fb) and not same_hand(fb,fc)
                                else "同手3連" if same_hand(fa,fb) and same_hand(fb,fc)
                                else "混合"),
        }
        trigrams.append(tg)

    stats = defaultdict(lambda: {"count":0,"total_interval":0,"errors":0,
                                  "same_finger":False,"same_hand":False,
                                  "alternating":False,"pattern_type":""})
    for tg in trigrams:
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
    df_tg    = pd.DataFrame(trigrams)
    df_stats = pd.DataFrame(stat_rows).sort_values("score", ascending=False)
    summary  = df_tg.groupby("pattern_type").agg(
        count=("avg_interval_ms","count"),
        avg_interval=("avg_interval_ms","mean"),
        std_interval=("avg_interval_ms","std"),
    ).reset_index()
    vowel_summary = df_tg.groupby("vowel_pattern").agg(
        count=("avg_interval_ms","count"),
        avg_interval=("avg_interval_ms","mean"),
        std_interval=("avg_interval_ms","std"),
    ).reset_index()

    output_file = f"typing_analysis_{session_id}.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_keys.to_excel(writer, sheet_name="打鍵ログ", index=False)
        df_tg.to_excel(writer, sheet_name="3gram_全データ", index=False)
        df_stats.to_excel(writer, sheet_name="3gram_集計", index=False)
        summary.to_excel(writer, sheet_name="パターン別サマリ", index=False)
        vowel_summary.to_excel(writer, sheet_name="母音パターン別サマリ", index=False)

    # JASP用CSVを別途出力
    csv_file = f"jasp_trigram_{session_id}.csv"
    df_tg.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"保存: {output_file}（打鍵:{len(normal_keys)}件 3gram:{len(trigrams)}件）")
    print(f"JASP用CSV: {csv_file}")

    all_trigrams_combined.append(df_tg)

# 全セッション統合CSV
if all_trigrams_combined:
    df_all = pd.concat(all_trigrams_combined, ignore_index=True)
    combined_csv = "jasp_trigram_ALL.csv"
    df_all.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    print(f"\n全セッション統合CSV: {combined_csv}（{len(df_all)}件）")
    print(f"hand_patternの分布:")
    print(df_all["hand_pattern"].value_counts().sort_index())