"""
trend.py  ─ セッション間の成長グラフ

使い方:
  python trend.py              # PNG を保存して表示
  python trend.py --no-show   # PNG 保存のみ（ヘッドレス）
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
import matplotlib.gridspec as gridspec
from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

PHRASE_BOUNDARY_MS = 1000
FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差",
    "R4":"右人差","R3":"右中指","R2":"右薬指","R1":"右小指",
}
STANDARD_FINGER = {
    "q":"L1","w":"L2","e":"L3","r":"L4","t":"L4",
    "y":"R4","u":"R4","i":"R3","o":"R2","p":"R1",
    "a":"L1","s":"L2","d":"L3","f":"L4","g":"L4",
    "h":"R4","j":"R4","k":"R3","l":"R2",
    "z":"L1","x":"L2","c":"L3","v":"L4","b":"L4",
    "n":"R4","m":"R4",",":"R3",".":"R2","/":"R1"," ":"RT",
}

# ── データ収集 ─────────────────────────────────────────────────
session_ids = sorted(set(
    re.search(r"session_(\d+)_keys", f).group(1)
    for f in glob.glob("session_*_keys.json")
))

records = []
for sid in session_ids:
    rec = {"session_id": sid}

    # ── キーログ統計 ──────────────────────────────────────────
    kf = f"session_{sid}_keys.json"
    if not os.path.exists(kf):
        continue
    with open(kf, encoding="utf-8") as f:
        keys = json.load(f)
    normal = [k for k in keys
              if not k.get("is_backspace") and not k["key"].startswith("Key.")]
    errors = sum(1 for k in normal if k.get("is_error"))
    ivs = [k["interval_ms"] for k in normal
           if 0 < k["interval_ms"] <= PHRASE_BOUNDARY_MS]
    pure_ms = (keys[-1]["timestamp_ms"] - keys[0]["timestamp_ms"]) - \
              sum(k["interval_ms"] for k in normal if k["interval_ms"] > PHRASE_BOUNDARY_MS)

    rec["n_keys"]      = len(normal)
    rec["error_rate"]  = errors / max(len(normal), 1) * 100
    rec["wpm"]         = (len(normal) / 5) / (pure_ms / 60000) if pure_ms > 0 else 0
    rec["avg_iv"]      = np.mean(ivs) if ivs else 0
    rec["median_iv"]   = np.median(ivs) if ivs else 0

    # ── integrate 統計 ────────────────────────────────────────
    inf = f"session_{sid}_integrated.json"
    if os.path.exists(inf):
        with open(inf, encoding="utf-8") as f:
            data = json.load(f)
        cam = [r for r in data if r["source"] == "camera"]
        key_ok = sum(1 for r in cam if r.get("key_match"))
        def is_fin(r):
            af = r.get("actual_finger")
            return af is not None and af == r.get("std_finger")
        fin_ok = sum(1 for r in cam if is_fin(r))
        dists  = [r["dist_to_key"] for r in cam if r.get("dist_to_key") is not None]

        rec["cam_rate"]    = len(cam) / max(len(data), 1) * 100
        rec["key_match"]   = key_ok / max(len(cam), 1) * 100
        rec["fin_match"]   = fin_ok / max(len(cam), 1) * 100
        rec["avg_dist"]    = np.mean(dists) if dists else float("nan")

        # 指ごとの一致率
        finger_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in cam:
            af = r.get("actual_finger")
            if af:
                finger_stats[af]["total"] += 1
                if is_fin(r):
                    finger_stats[af]["correct"] += 1
        rec["finger_stats"] = finger_stats

    records.append(rec)

if len(records) < 2:
    print("セッションが2件以上必要です")
    sys.exit(0)

# ── プロット ───────────────────────────────────────────────────
labels = [f"S{i+1}\n({r['n_keys']}打)" for i, r in enumerate(records)]
x      = np.arange(len(records))
colors = plt.cm.tab10.colors

fig = plt.figure(figsize=(16, 12))
fig.suptitle("タイピング研究 セッション間の成長記録", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── ① WPM ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
wpms = [r["wpm"] for r in records]
ax1.plot(x, wpms, "o-", color=colors[0], linewidth=2, markersize=7)
ax1.fill_between(x, wpms, alpha=0.15, color=colors[0])
for xi, v in zip(x, wpms):
    ax1.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=9)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_title("WPM（フレーズ待機除外）", fontweight="bold")
ax1.set_ylabel("Words Per Minute")
ax1.grid(axis="y", alpha=0.3)

# ── ② 平均打鍵間隔 ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
avg_ivs = [r["avg_iv"] for r in records]
med_ivs = [r["median_iv"] for r in records]
ax2.plot(x, avg_ivs, "o-", color=colors[1], linewidth=2, markersize=7, label="平均")
ax2.plot(x, med_ivs, "s--", color=colors[1], linewidth=1.5, markersize=6, alpha=0.6, label="中央値")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_title("打鍵間隔（ms）", fontweight="bold")
ax2.set_ylabel("間隔 (ms)")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

# ── ③ キー一致率・運指一致率 ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
w = 0.35
key_m = [r.get("key_match", 0) for r in records]
fin_m = [r.get("fin_match", 0) for r in records]
bars1 = ax3.bar(x - w/2, key_m, w, label="キー位置一致率", color=colors[2], alpha=0.8)
bars2 = ax3.bar(x + w/2, fin_m, w, label="運指一致率",     color=colors[3], alpha=0.8)
for bar in [*bars1, *bars2]:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
             ha="center", va="bottom", fontsize=7.5)
ax3.set_ylim(0, 110)
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8)
ax3.set_title("位置・運指の一致率", fontweight="bold")
ax3.set_ylabel("一致率 (%)")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)

# ── ④ ミス率 ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
errs = [r["error_rate"] for r in records]
bar_colors = [colors[4] if e > 1.0 else colors[0] for e in errs]
bars = ax4.bar(x, errs, color=bar_colors, alpha=0.8)
for bar, v in zip(bars, errs):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=9)
ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=8)
ax4.set_title("ミス率（バックスペース率）", fontweight="bold")
ax4.set_ylabel("ミス率 (%)")
ax4.grid(axis="y", alpha=0.3)

# ── ⑤ 指ごとの運指一致率（全セッション合計） ─────────────────
ax5 = fig.add_subplot(gs[2, :])
finger_order = ["L1","L2","L3","L4","R4","R3","R2","R1"]
session_finger_data = []
for r in records:
    fs = r.get("finger_stats", {})
    rates = []
    for code in finger_order:
        st = fs.get(code, {"total": 0, "correct": 0})
        rates.append(st["correct"] / max(st["total"], 1) * 100 if st["total"] > 0 else float("nan"))
    session_finger_data.append(rates)

bar_w = 0.15
for si, (r, rates) in enumerate(zip(records, session_finger_data)):
    xi = np.arange(len(finger_order)) + si * bar_w
    valid = [v if not np.isnan(v) else 0 for v in rates]
    ax5.bar(xi, valid, bar_w, label=f"S{si+1}", alpha=0.8, color=colors[si % len(colors)])

center = np.arange(len(finger_order)) + bar_w * (len(records) - 1) / 2
ax5.set_xticks(center)
ax5.set_xticklabels([f"{FINGER_NAMES.get(c, c)}\n({c})" for c in finger_order], fontsize=9)
ax5.set_ylim(0, 110)
ax5.set_title("指ごとの運指一致率（セッション別）", fontweight="bold")
ax5.set_ylabel("運指一致率 (%)")
ax5.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="80%基準")
ax5.legend(fontsize=8, loc="lower right")
ax5.grid(axis="y", alpha=0.3)

plt.savefig("trend.png", dpi=150, bbox_inches="tight")
print("保存: trend.png")

# ── テキストサマリー ──────────────────────────────────────────
print()
print("=" * 65)
print(f"  成長トレンドサマリー（全{len(records)}セッション）")
print("=" * 65)
print(f"{'':4s} {'打鍵数':>6s} {'WPM':>6s} {'avg_iv':>7s} {'ミス率':>6s} {'Key%':>6s} {'Fin%':>6s}")
print("-" * 65)
for i, r in enumerate(records):
    print(f"  S{i+1}  {r['n_keys']:6d}  {r['wpm']:6.0f}  "
          f"{r['avg_iv']:7.0f}ms  {r['error_rate']:5.1f}%  "
          f"{r.get('key_match', 0):5.0f}%  {r.get('fin_match', 0):5.0f}%")

# 最初と最後を比較
if len(records) >= 2:
    first, last = records[0], records[-1]
    print()
    print("  【最初 → 最新 の変化】")
    def delta(key, fmt=".0f", unit=""):
        a, b = first.get(key, 0), last.get(key, 0)
        sign = "+" if b >= a else ""
        return f"{a:{fmt}}{unit} → {b:{fmt}}{unit}  ({sign}{b-a:{fmt}}{unit})"
    print(f"  WPM     : {delta('wpm', '.0f')}")
    print(f"  打鍵間隔: {delta('avg_iv', '.0f', 'ms')}")
    print(f"  ミス率  : {delta('error_rate', '.1f', '%')}")
    print(f"  運指一致: {delta('fin_match', '.0f', '%')}")
