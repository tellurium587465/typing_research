"""
arpeggio_analysis.py  ─ アルペジオ打鍵の検出とペア分類

アルペジオ（意図的な高速連打）と偶発的な急加速を統計的に区別する。

分類基準:
  CV (変動係数 = σ/μ) と 平均間隔 で 4 種類に分類:

  ① arpeggio   : 速い + 一貫 (mean < FAST_MS  and CV < CV_ARPEGGIO)
  ② fast       : 速い + やや変動 (mean < FAST_MS  and CV < CV_VARIABLE)
  ③ variable   : 分散大 (CV >= CV_VARIABLE)  ← 急加速リスクが最も高い
  ④ normal     : 上記以外

出力:
  arpeggio_map.json  ─ パイプラインの error_analysis が参照する

使い方:
  python arpeggio_analysis.py              # 全セッション統合
  python arpeggio_analysis.py --min-n 3   # 最小サンプル数を下げる
  python arpeggio_analysis.py --show      # ペア一覧を表示して終了
"""
import argparse, json, glob, os, re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from session_utils import (get_session_files, list_all_session_ids,
                           find_session_key_files, output_path)
from constants import PHRASE_BOUNDARY_MS, OUTPUT_DIR
from plot_utils import setup_jp_font
setup_jp_font()

# ── パラメータ ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min-n",       type=int,   default=5,
                    help="最小サンプル数（少ないと信頼性低）")
parser.add_argument("--fast-ms",     type=float, default=100,
                    help="「速い」と判定する平均間隔の閾値 (ms)")
parser.add_argument("--cv-arpeggio", type=float, default=0.35,
                    help="アルペジオと判定するCV上限")
parser.add_argument("--cv-variable", type=float, default=0.60,
                    help="「変動大」と判定するCV下限（急加速リスク）")
parser.add_argument("--phrase-ms",   type=int,   default=1000)
parser.add_argument("--show",        action="store_true",
                    help="ペア一覧を表示")
parser.add_argument("--out",         default=None,
                    help="保存先（省略時は output/arpeggio_map.json）")
args = parser.parse_args()

# ── 全セッションのキーペア統計を収集 ──────────────────────────
pair_ivs = defaultdict(list)   # (prev_key, curr_key) → [interval_ms, ...]

for kf in find_session_key_files():
    with open(kf, encoding="utf-8") as f:
        keys = json.load(f)
    normal = [k for k in keys
              if not k.get("is_backspace")
              and not k["key"].startswith("Key.")]
    for i in range(1, len(normal)):
        a, b = normal[i-1], normal[i]
        if 0 < b["interval_ms"] <= args.phrase_ms:
            pair_ivs[(a["key"], b["key"])].append(b["interval_ms"])

print(f"ペア総数: {len(pair_ivs)}  総サンプル: {sum(len(v) for v in pair_ivs.values())}")

# ── ペア分類 ───────────────────────────────────────────────────
arpeggio_map = {}

for (pk, ck), ivs in pair_ivs.items():
    if len(ivs) < args.min_n:
        continue
    mean = np.mean(ivs)
    std  = np.std(ivs, ddof=1)
    cv   = std / mean if mean > 0 else 999
    q10  = np.percentile(ivs, 10)
    q90  = np.percentile(ivs, 90)

    # 分類
    if mean < args.fast_ms and cv < args.cv_arpeggio:
        pair_type = "arpeggio"   # 速い + 一貫 → 意図的なアルペジオ
    elif mean < args.fast_ms and cv < args.cv_variable:
        pair_type = "fast"       # 速い + やや変動 → 練習中のアルペジオ候補
    elif cv >= args.cv_variable:
        pair_type = "variable"   # 分散大 → 急加速リスク
    else:
        pair_type = "normal"

    arpeggio_map[f"{pk}{ck}"] = {
        "prev_key": pk, "curr_key": ck,
        "mean":   round(float(mean), 1),
        "std":    round(float(std),  1),
        "cv":     round(float(cv),   3),
        "q10":    round(float(q10),  1),
        "q90":    round(float(q90),  1),
        "n":      len(ivs),
        "type":   pair_type,
    }

# ── 保存 ──────────────────────────────────────────────────────
out_path = args.out if args.out else output_path("arpeggio_map.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(arpeggio_map, f, ensure_ascii=False, indent=2)
print(f"保存: {out_path}  ({len(arpeggio_map)}ペア)")

# ── サマリー ──────────────────────────────────────────────────
by_type = defaultdict(list)
for v in arpeggio_map.values():
    by_type[v["type"]].append(v)

print()
print("=" * 60)
print("  ペア分類サマリー")
print("=" * 60)
for t in ["arpeggio", "fast", "variable", "normal"]:
    entries = by_type[t]
    if not entries:
        continue
    means = [e["mean"] for e in entries]
    print(f"  {t:10s}: {len(entries):4d}ペア  "
          f"avg_mean={np.mean(means):.0f}ms  "
          f"avg_cv={np.mean([e['cv'] for e in entries]):.2f}")

# アルペジオペアを表示
if by_type["arpeggio"]:
    print()
    print("  【アルペジオペア一覧】（速い + 一貫）")
    print(f"  {'ペア':6s} {'mean':>8s} {'std':>6s} {'CV':>6s} {'N':>5s}")
    print("  " + "-" * 35)
    for e in sorted(by_type["arpeggio"], key=lambda x: x["mean"]):
        print(f"  {e['prev_key']}->{e['curr_key']}  "
              f"{e['mean']:7.0f}ms  {e['std']:5.0f}ms  "
              f"{e['cv']:5.2f}  {e['n']:4d}")

# 変動大ペアを表示（急加速リスク）
if by_type["variable"]:
    print()
    print("  【変動大ペア】（急加速リスク）")
    print(f"  {'ペア':6s} {'mean':>8s} {'std':>6s} {'CV':>6s} {'N':>5s}")
    print("  " + "-" * 35)
    for e in sorted(by_type["variable"], key=lambda x: -x["cv"])[:10]:
        print(f"  {e['prev_key']}->{e['curr_key']}  "
              f"{e['mean']:7.0f}ms  {e['std']:5.0f}ms  "
              f"{e['cv']:5.2f}  {e['n']:4d}")

# ── 可視化 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("アルペジオ打鍵パターン分析", fontsize=13, fontweight="bold")

# 左: 全ペアの mean vs CV の散布図
type_colors = {
    "arpeggio": "#F44336", "fast": "#FF9800",
    "variable": "#9C27B0", "normal": "#2196F3"
}
ax = axes[0]
for t, col in type_colors.items():
    pts = [(e["mean"], e["cv"]) for e in by_type[t] if e["n"] >= args.min_n]
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=col, label=t, alpha=0.7, s=30)

# アルペジオ境界を描画
ax.axvline(args.fast_ms,      color="gray", linestyle="--", linewidth=1)
ax.axhline(args.cv_arpeggio,  color="red",  linestyle=":",  linewidth=1, label=f"CV={args.cv_arpeggio}")
ax.axhline(args.cv_variable,  color="purple",linestyle=":", linewidth=1, label=f"CV={args.cv_variable}")

# アルペジオペアにラベル
for e in by_type["arpeggio"]:
    ax.annotate(f"{e['prev_key']}{e['curr_key']}",
                (e["mean"], e["cv"]), fontsize=7,
                xytext=(3, 3), textcoords="offset points")

ax.set_xlabel("平均間隔 (ms)")
ax.set_ylabel("変動係数 CV = σ/μ")
ax.set_title("ペア分類マップ\n（赤=アルペジオ、紫=変動大）", fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(0, min(300, max(e["mean"] for e in arpeggio_map.values()) * 1.1))
ax.set_ylim(0, min(2.5, max(e["cv"] for e in arpeggio_map.values()) * 1.1))
ax.grid(alpha=0.3)

# 右: アルペジオペアの分布（バイオリンプロット）
ax2 = axes[1]
top_arp = sorted(by_type["arpeggio"], key=lambda x: x["mean"])[:8]
if top_arp:
    data_for_violin = []
    labels_v = []
    for e in top_arp:
        pk, ck = e["prev_key"], e["curr_key"]
        ivs = pair_ivs.get((pk, ck), [])
        if ivs:
            data_for_violin.append(ivs)
            labels_v.append(f"{pk}->{ck}")
    if data_for_violin:
        vp = ax2.violinplot(data_for_violin,
                            showmedians=True, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor("#F44336")
            pc.set_alpha(0.6)
        vp["cmedians"].set_color("white")
        ax2.set_xticks(range(1, len(labels_v) + 1))
        ax2.set_xticklabels(labels_v, rotation=30, fontsize=9)
        ax2.set_title("アルペジオペアの間隔分布", fontweight="bold")
        ax2.set_ylabel("interval (ms)")
        ax2.grid(alpha=0.3)

plt.tight_layout()
from session_utils import output_path as _op
plt.savefig(_op("arpeggio_analysis.png"), dpi=150, bbox_inches="tight")
print(f"\n保存: {_op('arpeggio_analysis.png')}") 
