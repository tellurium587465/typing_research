"""
weakness.py  ─ キーごとの「苦手スコア」と練習優先度

苦手スコア = (1 - key_match_rate) × 0.40
           + (1 - fin_match_rate)  × 0.30
           + norm(avg_dist)        × 0.20
           + error_rate            × 0.10

使い方:
  python weakness.py              # 全セッション統合
  python weakness.py --session-id 1778934176
  python weakness.py --top 8      # 上位N件を表示
"""
import argparse
import json
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from session_utils import get_session_files, find_session_integrated_files, list_all_session_ids
from constants import FINGER_NAMES, STANDARD_FINGER
from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--top", type=int, default=8, help="苦手キー上位N件")
args = parser.parse_args()

# ── データ収集 ─────────────────────────────────────────────────
if args.session_id:
    sf_s = get_session_files(args.session_id)
    int_files = [sf_s["integrated"]] if os.path.exists(sf_s["integrated"]) else []
else:
    int_files = find_session_integrated_files()

KEY_STATS = defaultdict(lambda: {
    "cam": 0, "key_ok": 0, "fin_ok": 0,
    "dists": [], "errors": 0, "std_finger": "?"
})

def is_fin(r):
    af = r.get("actual_finger")
    return af is not None and af == r.get("std_finger")

for fpath in int_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        if r["source"] != "camera":
            continue
        k = r["key"]
        if len(k) != 1 or not k.isalpha():
            continue  # 記号・スペースは除外
        KEY_STATS[k]["cam"]       += 1
        KEY_STATS[k]["key_ok"]    += int(r.get("key_match", False))
        KEY_STATS[k]["fin_ok"]    += int(is_fin(r))
        KEY_STATS[k]["errors"]    += int(r.get("is_error", False))
        KEY_STATS[k]["std_finger"] = r.get("std_finger", "?")
        if r.get("dist_to_key") is not None:
            KEY_STATS[k]["dists"].append(r["dist_to_key"])

# ── スコア計算 ─────────────────────────────────────────────────
MIN_SAMPLES = 5  # サンプル数が少ないキーは除外

results = []
for key, st in KEY_STATS.items():
    n = st["cam"]
    if n < MIN_SAMPLES:
        continue
    key_rate = st["key_ok"] / n
    fin_rate = st["fin_ok"] / n
    avg_dist = np.mean(st["dists"]) if st["dists"] else 0.0
    err_rate = st["errors"] / n
    results.append({
        "key":      key,
        "n":        n,
        "key_rate": key_rate,
        "fin_rate": fin_rate,
        "avg_dist": avg_dist,
        "err_rate": err_rate,
        "std":      st["std_finger"],
    })

if not results:
    print("データが不足しています")
    sys.exit(0)

# avg_dist を 0〜1 に正規化
all_dists = [r["avg_dist"] for r in results]
dmin, dmax = min(all_dists), max(all_dists)
for r in results:
    r["dist_norm"] = (r["avg_dist"] - dmin) / max(dmax - dmin, 1e-9)

# 苦手スコア（高いほど練習が必要）
for r in results:
    r["weakness"] = (
        (1 - r["key_rate"]) * 0.40 +
        (1 - r["fin_rate"]) * 0.30 +
        r["dist_norm"]      * 0.20 +
        r["err_rate"]       * 0.10
    )

results.sort(key=lambda x: -x["weakness"])

# ── テキスト出力 ───────────────────────────────────────────────
FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差",
    "R4":"右人差","R3":"右中指","R2":"右薬指","R1":"右小指",
    "LT":"左親指","RT":"右親指",
}

print("=" * 70)
print(f"  キーごとの苦手スコア（全{len(results)}キー、サンプル数>={MIN_SAMPLES}）")
print("=" * 70)
print(f"  {'Key':4s} {'担当指':8s} {'苦手スコア':>10s} {'Key%':>6s} {'Fin%':>6s} {'dist':>7s} {'N':>5s}")
print("  " + "-" * 56)
for i, r in enumerate(results):
    star = " ★" if i < args.top else "  "
    fin  = FINGER_NAMES.get(r["std"], r["std"])
    print(f"{star} {r['key']:4s} {fin:8s}  {r['weakness']:.3f}      "
          f"{r['key_rate']*100:5.0f}%  {r['fin_rate']*100:5.0f}%  "
          f"{r['avg_dist']:5.1f}px  {r['n']:4d}")

print()
print(f"  練習優先キー TOP{args.top}: ", end="")
print(" ".join(f"[{r['key'].upper()}]" for r in results[:args.top]))

# ── ヒートマップ可視化 ────────────────────────────────────────
KEYBOARD_ROWS = [
    ["q","w","e","r","t","y","u","i","o","p"],
    ["a","s","d","f","g","h","j","k","l"],
    ["z","x","c","v","b","n","m"],
]
ROW_OFFSETS = [0.0, 0.3, 0.6]  # 各行の左オフセット（キー幅単位）

score_map = {r["key"]: r["weakness"] for r in results}
n_map     = {r["key"]: r["n"] for r in results}

# グリーン→イエロー→レッドのカラーマップ
cmap = LinearSegmentedColormap.from_list(
    "weakness", ["#22bb44", "#ffdd00", "#ee2222"])

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 3.3)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

for row_i, (row, offset) in enumerate(zip(KEYBOARD_ROWS, ROW_OFFSETS)):
    for col_i, key in enumerate(row):
        x = col_i + offset
        y = 2 - row_i
        score  = score_map.get(key, np.nan)
        color  = cmap(score) if not np.isnan(score) else (0.3, 0.3, 0.3, 1.0)
        n      = n_map.get(key, 0)

        rect = plt.Rectangle((x - 0.45, y - 0.45), 0.90, 0.90,
                              facecolor=color, edgecolor="white",
                              linewidth=1.0, zorder=2)
        ax.add_patch(rect)

        # キー名
        ax.text(x, y + 0.12, key.upper(), ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=3)
        # スコア
        if not np.isnan(score):
            ax.text(x, y - 0.18, f"{score:.2f}", ha="center", va="center",
                    fontsize=7, color="white", alpha=0.9, zorder=3)

# カラーバー
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                    fraction=0.04, pad=0.02, aspect=40)
cbar.set_label("苦手スコア  (赤=練習が必要)", color="white", fontsize=10)
cbar.ax.xaxis.set_tick_params(color="white")
plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")
cbar.outline.set_edgecolor("white")

title_sfx = f"session {args.session_id}" if args.session_id else "全セッション合計"
ax.set_title(f"キーごとの苦手ヒートマップ（{title_sfx}）",
             color="white", fontsize=13, fontweight="bold", pad=10)

from session_utils import output_path as _op
plt.savefig(_op("weakness_heatmap.png"), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\n保存: {_op('weakness_heatmap.png')}") 
