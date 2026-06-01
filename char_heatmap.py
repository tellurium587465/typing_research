import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from collections import defaultdict
from nlp_analysis import roman_to_hira

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

trigrams = []
for i in range(len(normal) - 2):
    a, b, c = normal[i], normal[i+1], normal[i+2]
    trigrams.append({
        "k1": a["key"], "k2": b["key"], "k3": c["key"],
        "avg_ms": (b["interval_ms"] + c["interval_ms"]) / 2
    })

# -----------------------------------------------
# ① 散布図
# -----------------------------------------------
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
sc = ax.scatter(counts, avgs, c=scores, cmap="RdYlGn_r",
                s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
plt.colorbar(sc, ax=ax, label="スコア（頻度×間隔）")
ax.axhline(np.mean(avgs), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axvline(np.mean(counts), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(0.97, 0.97, "高頻度×高間隔\n【最優先改善】", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color="#d73027",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.03, 0.97, "低頻度×高間隔\n【改善候補】", transform=ax.transAxes,
        ha="left", va="top", fontsize=10, color="#fc8d59",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text(0.97, 0.03, "高頻度×低間隔\n【理想パターン】", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10, color="#4575b4",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
for d in sorted(scatter_data, key=lambda x: -x["score"])[:10]:
    ax.annotate(d["hira"], xy=(d["count"], d["avg"]),
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
# ② キーボードヒートマップ（改善版）
# -----------------------------------------------
# QWERTYキーボードの物理座標（数字行追加）
KEY_ROWS = [
    # 数字行
    [("1",0),("2",1),("3",2),("4",3),("5",4),("6",5),("7",6),("8",7),("9",8),("0",9)],
    # 上段
    [("q",0.5),("w",1.5),("e",2.5),("r",3.5),("t",4.5),("y",5.5),("u",6.5),("i",7.5),("o",8.5),("p",9.5)],
    # 中段
    [("a",0.8),("s",1.8),("d",2.8),("f",3.8),("g",4.8),("h",5.8),("j",6.8),("k",7.8),("l",8.8)],
    # 下段
    [("z",1.2),("x",2.2),("c",3.2),("v",4.2),("b",5.2),("n",6.2),("m",7.2),(",",8.2),(".",9.2)],
]
KEY_Y = {0: 3, 1: 2, 2: 1, 3: 0}  # 行→y座標

KEY_POS = {}
for row_idx, row in enumerate(KEY_ROWS):
    for key, x in row:
        KEY_POS[key] = (x, KEY_Y[row_idx])

# スペース
KEY_POS[" "] = (4.5, -1)

# ホームポジションキー
HOME_KEYS = {"a", "s", "d", "f", "j", "k", "l"}

# キーごとの平均打鍵間隔
key2_intervals = defaultdict(list)
for tg in trigrams:
    key2_intervals[tg["k2"]].append(tg["avg_ms"])

key_avg = {k: np.mean(v) for k, v in key2_intervals.items() if len(v) >= 3}
all_vals = list(key_avg.values())
v_min = np.percentile(all_vals, 10) if all_vals else 80
v_max = np.percentile(all_vals, 90) if all_vals else 150

cmap = plt.cm.RdYlGn_r
KEY_SIZE = 0.85

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.8, 4.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("キーボードヒートマップ（改善版）\n色=平均打鍵間隔（赤=遅い/緑=速い）"
             "　矢印=ボトルネック3-gramの流れ　■=ホームポジション", fontsize=12)

for key, (x, y) in KEY_POS.items():
    avg = key_avg.get(key)
    if avg is not None:
        norm_val = (avg - v_min) / (v_max - v_min + 1e-9)
        norm_val = np.clip(norm_val, 0, 1)
        color = cmap(norm_val)
        alpha = 1.0
    else:
        color = "#e8e8e8"
        norm_val = 0.5
        alpha = 0.5

    # ホームポジションは枠を太く
    edge_color = "#FF6600" if key in HOME_KEYS else "white"
    edge_width  = 3.0     if key in HOME_KEYS else 1.5

    rect = mpatches.FancyBboxPatch(
        (x - KEY_SIZE/2, y - KEY_SIZE/2), KEY_SIZE, KEY_SIZE,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edge_color,
        linewidth=edge_width, alpha=alpha
    )
    ax.add_patch(rect)

    label = key.upper() if key not in (" ", ",", ".") else key
    text_color = "white" if (avg and norm_val > 0.6) else "#333333"
    ax.text(x, y + 0.1, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color)
    if avg:
        ax.text(x, y - 0.22, f"{avg:.0f}ms", ha="center", va="center",
                fontsize=6.5, color=text_color)

# スペースキー（横長）
avg_spc = key_avg.get(" ")
spc_color = cmap(np.clip((avg_spc - v_min)/(v_max - v_min + 1e-9), 0, 1)) if avg_spc else "#e8e8e8"
rect_spc = mpatches.FancyBboxPatch((2.5, -1.4), 4.0, 0.85,
    boxstyle="round,pad=0.05", facecolor=spc_color,
    edgecolor="white", linewidth=1.5)
ax.add_patch(rect_spc)
ax.text(4.5, -1.0, "SPACE", ha="center", va="center", fontsize=9, fontweight="bold")
if avg_spc:
    ax.text(4.5, -1.3, f"{avg_spc:.0f}ms", ha="center", va="center", fontsize=6.5)

# ボトルネック上位5件を矢印で表示
top5 = sorted(scatter_data, key=lambda x: -x["score"])[:5]
arrow_colors = ["#8B0000","#cc0000","#e34a33","#fc8d59","#fdbb84"]
for idx, d in enumerate(top5):
    keys = [d["roman"][0], d["roman"][1], d["roman"][2]]
    positions = [KEY_POS.get(k) for k in keys]
    if all(p is not None for p in positions):
        for j in range(len(positions)-1):
            x1, y1 = positions[j]
            x2, y2 = positions[j+1]
            ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>",
                    color=arrow_colors[idx], lw=2.0, alpha=0.85))
        mx = np.mean([p[0] for p in positions])
        my = max(p[1] for p in positions) + 0.6
        ax.text(mx, my, f"#{idx+1} {d['hira']}({d['avg']:.0f}ms)",
                ha="center", fontsize=8, color=arrow_colors[idx],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# カラーバー
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="平均打鍵間隔 (ms)",
             orientation="horizontal", fraction=0.03, pad=0.05)

# 凡例
legend_elements = [
    mpatches.Patch(facecolor="white", edgecolor="#FF6600", linewidth=3, label="ホームポジション"),
    mpatches.Patch(facecolor="#e8e8e8", label="データなし（出現頻度低）"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("keyboard_heatmap_char.png", dpi=150)
plt.show()
print("保存: keyboard_heatmap_char.png")