import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams["font.family"] = "MS Gothic"  # 日本語フォント

STANDARD_FINGER = {
    "q":"L1","w":"L2","e":"L3","r":"L4","t":"L4",
    "y":"R4","u":"R4","i":"R3","o":"R2","p":"R1",
    "a":"L1","s":"L2","d":"L3","f":"L4","g":"L4",
    "h":"R4","j":"R4","k":"R3","l":"R2",
    "z":"L1","x":"L2","c":"L3","v":"L4","b":"L4",
    "n":"R4","m":"R4",
    ",":"R3",".":"R2","/":"R1"," ":"RT",
}
FINGER_ORDER = ["L1","L2","L3","L4","R4","R3","R2","R1"]
FINGER_LABELS = {
    "L1":"左\n小指","L2":"左\n薬指","L3":"左\n中指","L4":"左\n人差し指",
    "R4":"右\n人差し指","R3":"右\n中指","R2":"右\n薬指","R1":"右\n小指"
}

# クリーニング済みファイルを使用
clean_files = glob.glob("keylog_*_clean.json")
if not clean_files:
    clean_files = glob.glob("keylog_*_with_errors.json") + [
        f for f in glob.glob("keylog_*.json") if "with_errors" not in f
    ]
print(f"使用ファイル: {clean_files}")

# 全セッション統合
all_keys = []
for fname in clean_files:
    with open(fname, encoding="utf-8") as f:
        data = json.load(f)
    normal = [
        e for e in data
        if not e.get("is_backspace", False)
        and not e["key"].startswith("Key.")
        and e["key"] in STANDARD_FINGER
    ]
    all_keys.extend(normal)

print(f"総打鍵数: {len(all_keys)}件")

# -----------------------------------------------
# ヒートマップ①：指×指の平均打鍵間隔（2-gram）
# -----------------------------------------------
pair_intervals = defaultdict(list)
for i in range(len(all_keys) - 1):
    fa = STANDARD_FINGER.get(all_keys[i]["key"])
    fb = STANDARD_FINGER.get(all_keys[i+1]["key"])
    if fa in FINGER_ORDER and fb in FINGER_ORDER:
        pair_intervals[(fa, fb)].append(all_keys[i+1]["interval_ms"])

# 行列作成
n = len(FINGER_ORDER)
matrix = np.full((n, n), np.nan)
for i, fa in enumerate(FINGER_ORDER):
    for j, fb in enumerate(FINGER_ORDER):
        vals = pair_intervals.get((fa, fb), [])
        if len(vals) >= 3:
            matrix[i][j] = np.mean(vals)

fig, ax = plt.subplots(figsize=(9, 7))
labels = [FINGER_LABELS[f] for f in FINGER_ORDER]
im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
               vmin=np.nanpercentile(matrix, 10),
               vmax=np.nanpercentile(matrix, 90))

ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("次の指（finger_t+1）", fontsize=11)
ax.set_ylabel("前の指（finger_t）", fontsize=11)
ax.set_title("指×指 平均打鍵間隔ヒートマップ（ms）\n赤=遅い ボトルネック / 緑=速い", fontsize=12)

for i in range(n):
    for j in range(n):
        if not np.isnan(matrix[i][j]):
            ax.text(j, i, f"{matrix[i][j]:.0f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[i][j] > np.nanpercentile(matrix, 70) else "black")

plt.colorbar(im, ax=ax, label="平均打鍵間隔 (ms)")
plt.tight_layout()
plt.savefig("heatmap_finger.png", dpi=150)
plt.show()
print("保存: heatmap_finger.png")

# -----------------------------------------------
# ヒートマップ②：母音パターン別平均間隔
# -----------------------------------------------
VOWELS = set("aiueo")
vowel_pattern_data = defaultdict(list)

for i in range(len(all_keys) - 2):
    a, b, c = all_keys[i], all_keys[i+1], all_keys[i+2]
    vp = "".join("V" if k["key"] in VOWELS else "C" for k in [a, b, c])
    avg_int = (b["interval_ms"] + c["interval_ms"]) / 2
    vowel_pattern_data[vp].append(avg_int)

patterns = ["CCC","CCV","CVC","CVV","VCC","VCV","VVC","VVV"]
means  = [np.mean(vowel_pattern_data[p]) if vowel_pattern_data[p] else np.nan for p in patterns]
counts = [len(vowel_pattern_data[p]) for p in patterns]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#d73027" if m > np.nanmean(means) else "#4575b4" for m in means]
bars = ax.bar(patterns, [m if not np.isnan(m) else 0 for m in means], color=colors)
for bar, count, mean in zip(bars, counts, means):
    if not np.isnan(mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={count}\n{mean:.0f}ms", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("母音パターン（V=母音 C=子音）", fontsize=11)
ax.set_ylabel("平均打鍵間隔 (ms)", fontsize=11)
ax.set_title("母音パターン別 平均打鍵間隔\n赤=平均以上（遅い） 青=平均以下（速い）", fontsize=12)
ax.axhline(np.nanmean(means), color="gray", linestyle="--", linewidth=1, label=f"全体平均 {np.nanmean(means):.0f}ms")
ax.legend()
plt.tight_layout()
plt.savefig("heatmap_vowel.png", dpi=150)
plt.show()
print("保存: heatmap_vowel.png")

# -----------------------------------------------
# ヒートマップ③：左手・右手・両手パターン別
# -----------------------------------------------
LEFT  = {"L1","L2","L3","L4"}
RIGHT = {"R1","R2","R3","R4"}

def hand_pattern(fa, fb, fc):
    def h(f):
        if f in LEFT:  return "L"
        if f in RIGHT: return "R"
        return "?"
    return f"{h(fa)}{h(fb)}{h(fc)}"

hand_data = defaultdict(list)
for i in range(len(all_keys) - 2):
    a, b, c = all_keys[i], all_keys[i+1], all_keys[i+2]
    fa = STANDARD_FINGER.get(a["key"])
    fb = STANDARD_FINGER.get(b["key"])
    fc = STANDARD_FINGER.get(c["key"])
    if fa and fb and fc:
        hp = hand_pattern(fa, fb, fc)
        avg_int = (b["interval_ms"] + c["interval_ms"]) / 2
        hand_data[hp].append(avg_int)

hand_patterns = ["LLL","LLR","LRL","LRR","RLL","RLR","RRL","RRR"]
hand_labels   = ["左左左\n（左手3連）","左左右","左右左\n（左右交互）","左右右",
                 "右左左","右左右\n（右左交互）","右右左","右右右\n（右手3連）"]
hand_means  = [np.mean(hand_data[p]) if hand_data[p] else np.nan for p in hand_patterns]
hand_counts = [len(hand_data[p]) for p in hand_patterns]

fig, ax = plt.subplots(figsize=(12, 5))
overall = np.nanmean(hand_means)
colors  = ["#d73027" if m > overall else "#4575b4" for m in hand_means]
bars    = ax.bar(hand_labels, [m if not np.isnan(m) else 0 for m in hand_means], color=colors)
for bar, count, mean in zip(bars, hand_counts, hand_means):
    if not np.isnan(mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={count}\n{mean:.0f}ms", ha="center", va="bottom", fontsize=9)
ax.axhline(overall, color="gray", linestyle="--", linewidth=1, label=f"全体平均 {overall:.0f}ms")
ax.set_ylabel("平均打鍵間隔 (ms)", fontsize=11)
ax.set_title("手のパターン別 平均打鍵間隔\n赤=平均以上（遅い） 青=平均以下（速い）", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("heatmap_hand.png", dpi=150)
plt.show()
print("保存: heatmap_hand.png")

# -----------------------------------------------
# ヒートマップ④：左手内・右手内の指連続パターン
# -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, hand, fingers, title in [
    (axes[0], "左手", ["L1","L2","L3","L4"], "左手内 指×指 平均打鍵間隔"),
    (axes[1], "右手", ["R4","R3","R2","R1"], "右手内 指×指 平均打鍵間隔"),
]:
    n = len(fingers)
    mat = np.full((n, n), np.nan)
    for i, fa in enumerate(fingers):
        for j, fb in enumerate(fingers):
            vals = pair_intervals.get((fa, fb), [])
            if len(vals) >= 2:
                mat[i][j] = np.mean(vals)

    labels = [FINGER_LABELS[f] for f in fingers]
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=np.nanpercentile(mat[~np.isnan(mat)], 10) if not np.all(np.isnan(mat)) else 0,
                   vmax=np.nanpercentile(mat[~np.isnan(mat)], 90) if not np.all(np.isnan(mat)) else 200)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("次の指", fontsize=10)
    ax.set_ylabel("前の指", fontsize=10)
    ax.set_title(title, fontsize=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i][j]):
                ax.text(j, i, f"{mat[i][j]:.0f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if mat[i][j] > np.nanpercentile(mat[~np.isnan(mat)], 70) else "black")
    plt.colorbar(im, ax=ax, label="ms")

plt.suptitle("左手・右手内の指連続パターン 平均打鍵間隔\n赤=遅い / 緑=速い", fontsize=13)
plt.tight_layout()
plt.savefig("heatmap_hand_detail.png", dpi=150)
plt.show()
print("保存: heatmap_hand_detail.png")