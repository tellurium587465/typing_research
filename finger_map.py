import json
import numpy as np
from collections import defaultdict

KEYLOG_FILE = "keylog_1778545201.json"

# -----------------------------------------------
# QWERTY 標準運指テーブル
# -----------------------------------------------
# 指ID定義：
#   L1=左小指 L2=左薬指 L3=左中指 L4=左人差し指
#   R4=右人差し指 R3=右中指 R2=右薬指 R1=右小指
#   LT=左親指 RT=右親指（スペース）

STANDARD_FINGER = {
    # 数字行
    "1": "L1", "2": "L2", "3": "L3", "4": "L4", "5": "L4",
    "6": "R4", "7": "R4", "8": "R3", "9": "R2", "0": "R1",

    # 上段
    "q": "L1", "w": "L2", "e": "L3", "r": "L4", "t": "L4",
    "y": "R4", "u": "R4", "i": "R3", "o": "R2", "p": "R1",

    # 中段（ホームポジション）
    "a": "L1", "s": "L2", "d": "L3", "f": "L4", "g": "L4",
    "h": "R4", "j": "R4", "k": "R3", "l": "R2",

    # 下段
    "z": "L1", "x": "L2", "c": "L3", "v": "L4", "b": "L4",
    "n": "R4", "m": "R4",

    # 記号
    ",": "R3", ".": "R2", "/": "R1",
    ";": "R1", "'": "R1",
    "-": "R1", "=": "R1",
    "[": "R1", "]": "R1",

    # スペース・特殊
    " ": "RT",
}

FINGER_NAMES = {
    "L1": "左小指", "L2": "左薬指", "L3": "左中指", "L4": "左人差し指",
    "R4": "右人差し指", "R3": "右中指", "R2": "右薬指", "R1": "右小指",
    "LT": "左親指", "RT": "右親指"
}

# 同じ手かどうか
def same_hand(f1, f2):
    return f1[0] == f2[0]

# 指番号（数字が大きいほど内側）
FINGER_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4,
                "R4": 4, "R3": 3, "R2": 2, "R1": 1,
                "LT": 0, "RT": 0}

# -----------------------------------------------
# キーログ読み込み＋標準運指を付与
# -----------------------------------------------
with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)

# 通常キーのみ（バックスペース・特殊キー除外）
normal_keys = [
    e for e in keylog
    if not e["is_backspace"]
    and not e["key"].startswith("Key.")
    and e["key"] in STANDARD_FINGER
]

# 標準運指を付与
for e in normal_keys:
    e["std_finger"] = STANDARD_FINGER[e["key"]]

print(f"有効打鍵数: {len(normal_keys)}件")

# -----------------------------------------------
# 3-gram 生成（標準運指ベース）
# -----------------------------------------------
trigrams = []
for i in range(len(normal_keys) - 2):
    a, b, c = normal_keys[i], normal_keys[i+1], normal_keys[i+2]
    tg = {
        "keys":    (a["key"], b["key"], c["key"]),
        "fingers": (a["std_finger"], b["std_finger"], c["std_finger"]),
        "intervals": (b["interval_ms"], c["interval_ms"]),
        "avg_interval": (b["interval_ms"] + c["interval_ms"]) / 2,
        "has_error": a["is_error"] or b["is_error"] or c["is_error"],
        # パターン分類
        "same_finger_repeat": a["std_finger"] == b["std_finger"] or
                              b["std_finger"] == c["std_finger"],
        "same_hand_all":      same_hand(a["std_finger"], b["std_finger"]) and
                              same_hand(b["std_finger"], c["std_finger"]),
        "alternating":        not same_hand(a["std_finger"], b["std_finger"]) and
                              not same_hand(b["std_finger"], c["std_finger"]),
    }
    trigrams.append(tg)

print(f"3-gram総数: {len(trigrams)}件")

# -----------------------------------------------
# 干渉スコア算出（出現頻度 × 平均打鍵間隔）
# -----------------------------------------------
tg_stats = defaultdict(lambda: {
    "count": 0, "total_interval": 0, "errors": 0,
    "same_finger": False, "same_hand": False, "alternating": False
})

for tg in trigrams:
    key = tg["fingers"]
    tg_stats[key]["count"] += 1
    tg_stats[key]["total_interval"] += tg["avg_interval"]
    tg_stats[key]["errors"] += int(tg["has_error"])
    tg_stats[key]["same_finger"] = tg["same_finger_repeat"]
    tg_stats[key]["same_hand"]   = tg["same_hand_all"]
    tg_stats[key]["alternating"] = tg["alternating"]

results = []
for fingers, stats in tg_stats.items():
    avg_int = stats["total_interval"] / stats["count"]
    score   = stats["count"] * avg_int
    results.append({
        "fingers":      fingers,
        "count":        stats["count"],
        "avg_interval": round(avg_int, 1),
        "score":        round(score, 1),
        "error_rate":   round(stats["errors"] / stats["count"], 3),
        "same_finger":  stats["same_finger"],
        "same_hand":    stats["same_hand"],
        "alternating":  stats["alternating"],
    })

results.sort(key=lambda x: -x["score"])

# -----------------------------------------------
# 結果表示
# -----------------------------------------------
print("\n" + "="*60)
print("【ボトルネック上位10件】スコア＝出現頻度×平均打鍵間隔")
print("="*60)
for r in results[:10]:
    f = [FINGER_NAMES.get(x, x) for x in r["fingers"]]
    pattern = []
    if r["same_finger"]:  pattern.append("同指連打")
    if r["same_hand"]:    pattern.append("同手3連")
    if r["alternating"]:  pattern.append("左右交互")
    print(f"  {' → '.join(f)}")
    print(f"    スコア:{r['score']:8.1f}  "
          f"出現:{r['count']:3d}回  "
          f"平均間隔:{r['avg_interval']:6.1f}ms  "
          f"ミス率:{r['error_rate']:.1%}  "
          f"{' '.join(pattern)}")

print("\n" + "="*60)
print("【同指連打パターン】最も最適化余地が大きい")
print("="*60)
same_finger = [r for r in results if r["same_finger"]]
for r in same_finger[:5]:
    f = [FINGER_NAMES.get(x, x) for x in r["fingers"]]
    print(f"  {' → '.join(f)}")
    print(f"    スコア:{r['score']:8.1f}  出現:{r['count']:3d}回  "
          f"平均間隔:{r['avg_interval']:6.1f}ms  ミス率:{r['error_rate']:.1%}")

print("\n" + "="*60)
print("【左右交互パターン】理想的な高速打鍵")
print("="*60)
alt = [r for r in results if r["alternating"]]
alt_fast = sorted(alt, key=lambda x: x["avg_interval"])
for r in alt_fast[:5]:
    f = [FINGER_NAMES.get(x, x) for x in r["fingers"]]
    print(f"  {' → '.join(f)}")
    print(f"    スコア:{r['score']:8.1f}  出現:{r['count']:3d}回  "
          f"平均間隔:{r['avg_interval']:6.1f}ms  ミス率:{r['error_rate']:.1%}")

# -----------------------------------------------
# JSONに保存
# -----------------------------------------------
output = {
    "session": KEYLOG_FILE,
    "total_keystrokes": len(normal_keys),
    "total_trigrams": len(trigrams),
    "trigram_stats": [
        {**r, "fingers": list(r["fingers"])}
        for r in results
    ]
}
with open("trigram_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n保存完了: trigram_analysis.json")