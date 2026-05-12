import json

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
    "LT":"左親指","RT":"右親指"
}

with open("trigram_analysis.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"総打鍵数: {data['total_keystrokes']}")
print(f"総3-gram数: {data['total_trigrams']}")

print("\n" + "="*60)
print("【ボトルネック上位10件】")
print("="*60)
for r in data["trigram_stats"][:10]:
    fingers = [FINGER_NAMES.get(f, f) for f in r["fingers"]]
    pattern = []
    if r["same_finger"]: pattern.append("同指連打")
    if r["same_hand"]:   pattern.append("同手3連")
    if r["alternating"]: pattern.append("左右交互")
    print(f"  {' → '.join(fingers)}")
    print(f"    スコア:{r['score']:8.1f}  出現:{r['count']:3d}回  "
          f"平均:{r['avg_interval']:6.1f}ms  "
          f"ミス率:{r['error_rate']:.1%}  {' '.join(pattern)}")

print("\n" + "="*60)
print("【同指連打パターン上位5件】")
print("="*60)
same_finger = [r for r in data["trigram_stats"] if r["same_finger"]]
for r in same_finger[:5]:
    fingers = [FINGER_NAMES.get(f, f) for f in r["fingers"]]
    print(f"  {' → '.join(fingers)}")
    print(f"    スコア:{r['score']:8.1f}  出現:{r['count']:3d}回  "
          f"平均:{r['avg_interval']:6.1f}ms  ミス率:{r['error_rate']:.1%}")

print("\n" + "="*60)
print("【左右交互パターン（速い順）上位5件】")
print("="*60)
alt = [r for r in data["trigram_stats"] if r["alternating"]]
alt_fast = sorted(alt, key=lambda x: x["avg_interval"])
for r in alt_fast[:5]:
    fingers = [FINGER_NAMES.get(f, f) for f in r["fingers"]]
    print(f"  {' → '.join(fingers)}")
    print(f"    スコア:{r['score']:8.1f}  出現:{r['count']:3d}回  "
          f"平均:{r['avg_interval']:6.1f}ms  ミス率:{r['error_rate']:.1%}")