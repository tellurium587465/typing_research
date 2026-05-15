import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from collections import defaultdict

matplotlib.rcParams["font.family"] = "MS Gothic"

# -----------------------------------------------
# データ読み込み
# -----------------------------------------------
keylog_files = glob.glob("keylog_*_clean.json")
if not keylog_files:
    keylog_files = glob.glob("keylog_*_with_errors.json") + [
        f for f in glob.glob("keylog_*.json")
        if "with_errors" not in f and "clean" not in f
    ]

all_keylog = []
for fname in keylog_files:
    with open(fname, encoding="utf-8") as f:
        all_keylog.extend(json.load(f))

normal = [
    e for e in all_keylog
    if not e.get("is_backspace", False)
    and not e["key"].startswith("Key.")
    and len(e["key"]) == 1
]

# 3-gram生成
trigrams = []
for i in range(len(normal) - 2):
    a, b, c = normal[i], normal[i+1], normal[i+2]
    trigrams.append({
        "k1": a["key"], "k2": b["key"], "k3": c["key"],
        "avg_ms": (b["interval_ms"] + c["interval_ms"]) / 2
    })

# -----------------------------------------------
# ① 散布図：出現頻度 vs 平均打鍵間隔
# -----------------------------------------------
from nlp_analysis import roman_to_hira

tg_stats = defaultdict(list)
for tg in trigrams:
    key = (tg["k1"], tg["k2"], tg["k3"])
    tg_stats[key].append(tg["avg_ms"])

scatter_data = []
for (k1, k2, k3), vals in tg_stats.items():
    if len(vals) >= 3:
        roman = k1 + k2 + k3
        scatter_data.append({
            "roman": roman,
            "hira":  roman_to_hira(roman),
            "count": len(vals),
            "avg":   np.mean(vals),
            "score": len(vals) * np.mean(vals)
        })

counts = np.array([d["count"] for d in scatter_data])
avgs   = np.array([d["avg"]   for d in scatter_data])
scores = np.array([d["score"] for d in scatter_data])

fig, ax = plt.subplots(figsize=(12, 8))

# 散布図（色=スコア）
sc = ax.scatter(counts, avgs,
                c=scores, cmap="RdYlGn_r",
                s=80, alpha=0.8, edgecolors="white", linewidth=0.5)

plt.colorbar(sc, ax=ax, label="スコア（頻度×間隔）")

# 平均線
ax.axhline(np.mean(avgs), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(np.mean(counts), color="gray", linestyle="--", linewidth=1, alpha=0.7)

# 象限ラベル
ax.text(0.97, 0.97, "高頻度×高間隔\n【最優先改善】",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#d73027",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.03, 0.97, "低頻度×高間隔\n【改善候補】",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color="#fc8d59",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.97, 0.03, "高頻度×低間隔\n【理想パターン】",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, color="#4575b4",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 上位ラベル表示（スコア上位10件）
top10 = sorted(scatter_data, key=lambda x: -x["score"])[:10]
for d in top10:
    ax.annotate(d["hira"],
                xy=(d["count"], d["avg"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="#d73027")

ax.set_xlabel("出現頻度（回）", fontsize=12)
ax.set_ylabel("平均打鍵間隔（ms）", fontsize=12)
ax.set_title("文字3-gram：出現頻度 vs 平均打鍵間隔\n右上=ボトルネック / 左下=高速パターン", fontsize=13)
plt.tight_layout()
plt.savefig("scatter_trigram.png", dpi=150)
plt.show()
print("保存: scatter_trigram.png")

# -----------------------------------------------
# ② キーボードヒートマップ（3-gramの関係を矢印で表示）
# -----------------------------------------------
# QWERTYキーボードの物理座標
KEY_POS = {
    "q":(0,3),"w":(1,3),"e":(2,3),"r":(3,3),"t":(4,3),
    "y":(5,3),"u":(6,3),"i":(7,3),"o":(8,3),"p":(9,3),
    "a":(0.3,2),"s":(1.3,2),"d":(2.3,2),"f":(3.3,2),"g":(4.3,2),
    "h":(5.3,2),"j":(6.3,2),"k":(7.3,2),"l":(8.3,2),
    "z":(0.7,1),"x":(1.7,1),"c":(2.7,1),"v":(3.7,1),"b":(4.7,1),
    "n":(5.7,1),"m":(6.7,1),",":(7.7,1),".":(8.7,1),
    " ":(4.5,0),
}
KEY_SIZE = 0.85

# キーごとの平均打鍵間隔（2打目として登場したときの間隔）
key2_intervals = defaultdict(list)
for tg in trigrams:
    key2_intervals[tg["k2"]].append(tg["avg_ms"])

key_avg = {k: np.mean(v) for k, v in key2_intervals.items() if len(v) >= 3}
if key_avg:
    v_min = min(key_avg.values())
    v_max = max(key_avg.values())
else:
    v_min, v_max = 80, 150

cmap = plt.cm.RdYlGn_r

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 4.0)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("キーボードヒートマップ\n色=平均打鍵間隔（赤=遅い/緑=速い）　矢印=ボトルネック3-gramの流れ",
             fontsize=12)

# キーを描画
for key, (x, y) in KEY_POS.items():
    avg = key_avg.get(key, None)
    if avg is not None:
        norm_val = (avg - v_min) / (v_max - v_min + 1e-9)
        color = cmap(norm_val)
    else:
        color = "#cccccc"

    rect = mpatches.FancyBboxPatch(
        (x - KEY_SIZE/2, y - KEY_SIZE/2), KEY_SIZE, KEY_SIZE,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.5
    )
    ax.add_patch(rect)

    label = key.upper() if key != " " else "SPC"
    ax.text(x, y + 0.05, label, ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="white" if avg and norm_val > 0.5 else "black")
    if avg:
        ax.text(x, y - 0.22, f"{avg:.0f}", ha="center", va="center",
                fontsize=6.5, color="white" if norm_val > 0.5 else "#333333")

# ボトルネック上位5件を矢印で表示
top5 = sorted(scatter_data, key=lambda x: -x["score"])[:5]
colors_arrow = ["#8B0000","#cc0000","#e34a33","#fc8d59","#fdbb84"]
for idx, d in enumerate(top5):
    k1, k2, k3 = d["roman"][0], d["roman"][1], d["roman"][2]
    # 複数文字の場合は最初の文字を使用
    keys = [k1, k2, k3]
    positions = [KEY_POS.get(k) for k in keys]
    if all(positions):
        for j in range(len(positions)-1):
            x1, y1 = positions[j]
            x2, y2 = positions[j+1]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors_arrow[idx],
                    lw=2.0, alpha=0.8
                )
            )
        # ラベル
        mx = np.mean([p[0] for p in positions])
        my = np.mean([p[1] for p in positions]) + 0.5
        ax.text(mx, my, f"#{idx+1} {d['hira']}",
                ha="center", fontsize=8,
                color=colors_arrow[idx],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

# カラーバー
sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=v_min, vmax=v_max))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="平均打鍵間隔 (ms)",
             orientation="horizontal", fraction=0.03, pad=0.02)

plt.tight_layout()
plt.savefig("keyboard_heatmap_char.png", dpi=150)
plt.show()
print("保存: keyboard_heatmap_char.png")