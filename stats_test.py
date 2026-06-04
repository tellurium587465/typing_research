"""
stats_test.py  ─ 3gramパターン間の打鍵速度を統計検定

比較対象:
  左右交互 vs 同手3連 vs 同指連打

検定方法:
  1. Shapiro-Wilk 正規性検定
  2. Mann-Whitney U 検定（対比較ペアごと）
  3. Kruskal-Wallis 検定（全グループ同時）
  4. Cohen's d 効果量

使い方:
  python stats_test.py                    # jasp_trigram_ALL.csv を使用
  python stats_test.py --csv my_data.csv
  python stats_test.py --save results.csv # 結果をCSV保存

出力: ターミナル表示 + stats_test_result.png（分布図）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="jasp_trigram_ALL.csv")
parser.add_argument("--save", default=None, help="検定結果をCSV保存するパス")
parser.add_argument("--min-n", type=int, default=30, help="最小サンプル数")
parser.add_argument("--max-iv", type=int, default=1000, help="外れ値除外閾値(ms)")
args = parser.parse_args()

if not os.path.exists(args.csv):
    print(f"CSVが見つかりません: {args.csv}")
    print("先に export_excel.py を実行してください")
    sys.exit(1)

# ── データ読み込み ─────────────────────────────────────────────
df = pd.read_csv(args.csv)
print(f"読み込み: {args.csv}  ({len(df)}件)")

# フレーズ境界またぎを除外（既にCSVから除外済みのはずだが念のため）
if "spans_boundary" in df.columns:
    df = df[~df["spans_boundary"]]
    print(f"フレーズ境界除外後: {len(df)}件")

# 外れ値除外（interval が異常に大きい行）
df = df[df["avg_interval_ms"] <= args.max_iv]
df = df[df["avg_interval_ms"] > 0]
print(f"外れ値除外後({args.max_iv}ms以下): {len(df)}件")

# ── グループ定義 ───────────────────────────────────────────────
GROUP_COL   = "pattern_type"
TARGET_COL  = "avg_interval_ms"
GROUPS_OF_INTEREST = ["左右交互", "同手3連", "同指連打"]

available = df[GROUP_COL].unique().tolist()
groups = {g: df[df[GROUP_COL] == g][TARGET_COL].values
          for g in GROUPS_OF_INTEREST if g in available}

print()
print("=" * 65)
print("  グループ別記述統計")
print("=" * 65)
desc_rows = []
for name, vals in groups.items():
    row = {
        "パターン":   name,
        "N":         len(vals),
        "平均(ms)":  round(np.mean(vals), 1),
        "中央値(ms)": round(np.median(vals), 1),
        "SD(ms)":    round(np.std(vals, ddof=1), 1),
        "最小":      round(np.min(vals), 1),
        "最大":      round(np.max(vals), 1),
    }
    desc_rows.append(row)
    print(f"  {name:8s}: N={len(vals):5d}  平均={np.mean(vals):.1f}ms  "
          f"中央値={np.median(vals):.1f}ms  SD={np.std(vals,ddof=1):.1f}ms")

# ── 正規性検定（Shapiro-Wilk、N<=5000 でのみ実施） ─────────────
print()
print("=" * 65)
print("  正規性検定（Shapiro-Wilk）")
print("=" * 65)
normality = {}
for name, vals in groups.items():
    sample = vals if len(vals) <= 5000 else vals[np.random.choice(len(vals), 5000, replace=False)]
    stat, p = stats.shapiro(sample)
    normality[name] = p
    verdict = "正規分布とみなせる" if p > 0.05 else "非正規（ノンパラ検定推奨）"
    print(f"  {name:8s}: W={stat:.4f}  p={p:.4e}  → {verdict}")

use_nonparam = any(p <= 0.05 for p in normality.values())

# ── Kruskal-Wallis 検定（全グループ） ─────────────────────────
print()
print("=" * 65)
print("  Kruskal-Wallis 検定（全グループ同時）")
print("=" * 65)
group_arrays = [v for v in groups.values() if len(v) >= args.min_n]
if len(group_arrays) >= 2:
    kw_stat, kw_p = stats.kruskal(*group_arrays)
    print(f"  H = {kw_stat:.4f}  p = {kw_p:.4e}")
    if kw_p < 0.001:
        print("  → グループ間に有意差あり（p < .001）")
    elif kw_p < 0.05:
        print("  → グループ間に有意差あり（p < .05）")
    else:
        print("  → グループ間に有意差なし（p ≥ .05）")

# ── Mann-Whitney U 検定（対比較） ─────────────────────────────
from itertools import combinations

print()
print("=" * 65)
print("  Mann-Whitney U 検定（対比較）＋ Cohen's d 効果量")
print("=" * 65)
print(f"  {'比較':22s} {'U':>10s} {'p値':>10s} {'有意':>6s} {'d':>7s} {'効果量'}  {'diff(ms)'}")
print("  " + "-" * 75)

test_results = []
group_names = list(groups.keys())
n_comparisons = len(list(combinations(group_names, 2)))

for g1, g2 in combinations(group_names, 2):
    v1, v2 = groups[g1], groups[g2]
    if len(v1) < args.min_n or len(v2) < args.min_n:
        continue

    # Mann-Whitney U
    u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided")

    # Bonferroni 補正
    p_adj = min(p_val * n_comparisons, 1.0)

    # Cohen's d（プールSD）
    pool_sd = np.sqrt((np.std(v1, ddof=1)**2 + np.std(v2, ddof=1)**2) / 2)
    d = (np.mean(v1) - np.mean(v2)) / max(pool_sd, 1e-9)

    # 効果量の分類
    abs_d = abs(d)
    if abs_d >= 0.8:   effect = "大"
    elif abs_d >= 0.5: effect = "中"
    elif abs_d >= 0.2: effect = "小"
    else:              effect = "無視できる"

    sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "n.s."))
    diff_ms = np.mean(v1) - np.mean(v2)

    label = f"{g1} vs {g2}"
    print(f"  {label:22s} {u_stat:10.0f} {p_adj:10.4e} {sig:>6s} {d:+7.3f} ({effect:8s})  {diff_ms:+.1f}ms")

    test_results.append({
        "比較": label,
        "U統計量": round(u_stat, 1),
        "p値(補正前)": round(p_val, 6),
        "p値(Bonferroni補正)": round(p_adj, 6),
        "有意": sig,
        "Cohen_d": round(d, 4),
        "効果量": effect,
        "平均差_ms": round(diff_ms, 2),
        f"{g1}_mean": round(np.mean(v1), 2),
        f"{g1}_n": len(v1),
        f"{g2}_mean": round(np.mean(v2), 2),
        f"{g2}_n": len(v2),
    })

# ── 結果保存 ───────────────────────────────────────────────────
if args.save and test_results:
    pd.DataFrame(test_results).to_csv(args.save, index=False, encoding="utf-8-sig")
    print(f"\n検定結果CSV保存: {args.save}")

# ── 可視化 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3gramパターン別 打鍵間隔の分布", fontsize=13, fontweight="bold")

colors_g = {"左右交互": "#2196F3", "同手3連": "#FF9800", "同指連打": "#F44336"}
labels_en = {"左右交互": "Alternating", "同手3連": "Same-Hand", "同指連打": "Same-Finger"}

# 左: バイオリンプロット
ax = axes[0]
data_list = [(name, vals) for name, vals in groups.items() if len(vals) >= args.min_n]
positions = range(1, len(data_list) + 1)
vparts = ax.violinplot([v for _, v in data_list], positions=positions,
                        showmedians=True, showextrema=False)
for i, (pc, (name, _)) in enumerate(zip(vparts["bodies"], data_list)):
    pc.set_facecolor(colors_g.get(name, "#888888"))
    pc.set_alpha(0.75)
vparts["cmedians"].set_color("white")
vparts["cmedians"].set_linewidth(2)
ax.set_xticks(positions)
ax.set_xticklabels([n for n, _ in data_list], fontsize=10)
ax.set_ylabel("打鍵間隔 (ms)")
ax.set_title("分布（バイオリンプロット）")
ax.grid(axis="y", alpha=0.3)

# 右: 平均 ± SD 棒グラフ
ax2 = axes[1]
names  = [n for n, _ in data_list]
means  = [np.mean(v) for _, v in data_list]
sds    = [np.std(v, ddof=1) for _, v in data_list]
bar_colors = [colors_g.get(n, "#888888") for n in names]
bars = ax2.bar(positions, means, yerr=sds, capsize=6,
               color=bar_colors, alpha=0.8, error_kw={"linewidth": 1.5})
for pos, m, sd in zip(positions, means, sds):
    ax2.text(pos, m + sd + 3, f"{m:.0f}ms", ha="center", va="bottom", fontsize=10)
ax2.set_xticks(positions)
ax2.set_xticklabels(names, fontsize=10)
ax2.set_ylabel("平均打鍵間隔 (ms)")
ax2.set_title("平均 ± SD")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
from session_utils import output_path as _op
plt.savefig(_op("stats_test_result.png"), dpi=150, bbox_inches="tight")
print(f"\n保存: {_op('stats_test_result.png')}") 

# ── 解釈コメント ──────────────────────────────────────────────
print()
print("=" * 65)
print("  解釈（研究発表用メモ）")
print("=" * 65)
for name, vals in groups.items():
    print(f"  {name:8s}: 平均 {np.mean(vals):.1f}ms  中央値 {np.median(vals):.1f}ms  "
          f"N={len(vals)}")
if test_results:
    best = min(test_results, key=lambda r: abs(r["Cohen_d"]))
    worst = max(test_results, key=lambda r: abs(r["Cohen_d"]))
    print(f"\n  最大の効果量: {worst['比較']}  d={worst['Cohen_d']:+.3f} ({worst['効果量']})")
    print(f"  最小の効果量: {best['比較']}  d={best['Cohen_d']:+.3f} ({best['効果量']})")
