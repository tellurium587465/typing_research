"""
fatigue_analysis.py  ─ セッション内の疲労・時間帯影響分析（④）

「長時間打っていると指が疲れて遅くなるか」を
セッションを時間帯ブロックに分割して比較する。

使い方:
  python fatigue_analysis.py                  # 全セッション
  python fatigue_analysis.py --session-id 1778934176
  python fatigue_analysis.py --blocks 5       # 何分割するか
"""
import argparse, json, glob, os, re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from session_utils import (get_session_files, find_session_key_files,
                           list_all_session_ids, session_id_from_path, output_path)
from constants import PHRASE_BOUNDARY_MS, FINGER_NAMES
from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--blocks",    type=int,   default=5, help="時間ブロック数")
parser.add_argument("--phrase-ms", type=int,   default=1000)
args = parser.parse_args()

if args.session_id:
    sf_s = get_session_files(args.session_id)
    key_files = [sf_s["keys"]] if os.path.exists(sf_s["keys"]) else []
else:
    key_files = find_session_key_files()

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
}

print("=" * 60)
print("  疲労・時間帯影響分析")
print("=" * 60)

fig, axes = plt.subplots(len(key_files), 2,
                          figsize=(13, 4 * len(key_files)),
                          squeeze=False)
fig.suptitle("セッション内の時間帯別パフォーマンス", fontsize=13, fontweight="bold")

for row_i, kf in enumerate(key_files):
    if not os.path.exists(kf):
        continue
    sid = session_id_from_path(kf)

    with open(kf, encoding="utf-8") as f:
        keys = json.load(f)

    # 有効打鍵（BSなし・フレーズ境界なし）
    normal = [k for k in keys
              if not k.get("is_backspace")
              and not k["key"].startswith("Key.")
              and 0 < k["interval_ms"] <= args.phrase_ms]

    if len(normal) < args.blocks * 5:
        print(f"  session {sid}: データ不足（{len(normal)}打鍵）")
        continue

    # 時系列の先頭・末尾
    t0 = normal[0]["timestamp_ms"]
    t1 = normal[-1]["timestamp_ms"]
    duration_s = (t1 - t0) / 1000

    # N ブロックに分割
    block_size = len(normal) // args.blocks
    blocks = [normal[i * block_size:(i + 1) * block_size]
              for i in range(args.blocks)]

    block_stats = []
    for bi, blk in enumerate(blocks):
        ivs    = [k["interval_ms"] for k in blk]
        errors = sum(1 for k in blk if k.get("is_error"))
        t_mid  = (blk[0]["timestamp_ms"] + blk[-1]["timestamp_ms"]) / 2
        elapsed_s = (t_mid - t0) / 1000
        block_stats.append({
            "block":      bi + 1,
            "elapsed_s":  elapsed_s,
            "mean_iv":    np.mean(ivs),
            "median_iv":  np.median(ivs),
            "std_iv":     np.std(ivs, ddof=1),
            "n_errors":   errors,
            "error_rate": errors / max(len(blk), 1) * 100,
            "n":          len(blk),
        })

    print(f"\n  session {sid} ({duration_s:.0f}秒、{len(normal)}打鍵)")
    print(f"  {'ブロック':>6s} {'経過時間':>8s} {'平均iv':>8s} {'中央iv':>8s} {'σ':>6s} {'エラー':>6s}")
    print("  " + "-" * 50)
    for st in block_stats:
        trend_mark = ""
        if st["block"] > 1:
            prev_med = block_stats[st["block"] - 2]["median_iv"]
            if st["median_iv"] > prev_med * 1.05:
                trend_mark = "↑遅"
            elif st["median_iv"] < prev_med * 0.95:
                trend_mark = "↓速"
        print(f"  {st['block']:6d}  {st['elapsed_s']:7.0f}s  "
              f"{st['mean_iv']:7.0f}ms  {st['median_iv']:7.0f}ms  "
              f"{st['std_iv']:5.0f}ms  {st['n_errors']:5d}  {trend_mark}")

    # 疲労の有無を判定（後半が前半より遅いか）
    if len(block_stats) >= 2:
        first_half  = np.mean([s["median_iv"] for s in block_stats[:len(block_stats)//2]])
        second_half = np.mean([s["median_iv"] for s in block_stats[len(block_stats)//2:]])
        change_pct  = (second_half - first_half) / first_half * 100
        verdict = f"後半 {change_pct:+.1f}%"
        if change_pct > 5:
            verdict += " → 疲労の可能性あり（遅くなっている）"
        elif change_pct < -5:
            verdict += " → ウォームアップ後に加速している"
        else:
            verdict += " → 大きな変化なし"
        print(f"  前半平均={first_half:.0f}ms → 後半平均={second_half:.0f}ms  ({verdict})")

    # グラフ
    ax_iv  = axes[row_i][0]
    ax_err = axes[row_i][1]
    xs     = [s["elapsed_s"] for s in block_stats]
    medivs = [s["median_iv"] for s in block_stats]

    ax_iv.plot(xs, medivs, "o-", color="#2196F3", linewidth=2, markersize=7)
    ax_iv.fill_between(
        xs,
        [s["median_iv"] - s["std_iv"] for s in block_stats],
        [s["median_iv"] + s["std_iv"] for s in block_stats],
        alpha=0.2, color="#2196F3"
    )
    ax_iv.axhline(np.mean(medivs), color="gray", linestyle="--",
                  linewidth=1, label="全体平均")
    for xi, yi in zip(xs, medivs):
        ax_iv.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)
    ax_iv.set_title(f"session {sid[-6:]}: 打鍵間隔の推移", fontweight="bold")
    ax_iv.set_xlabel("経過時間 (s)")
    ax_iv.set_ylabel("中央値 interval (ms)")
    ax_iv.legend(fontsize=8)
    ax_iv.grid(alpha=0.3)

    err_rates = [s["error_rate"] for s in block_stats]
    ax_err.bar(range(len(block_stats)), err_rates,
               color=["#F44336" if e > 0 else "#4CAF50" for e in err_rates],
               alpha=0.8)
    ax_err.set_xticks(range(len(block_stats)))
    ax_err.set_xticklabels([f"B{s['block']}" for s in block_stats])
    ax_err.set_title(f"session {sid[-6:]}: ブロック別エラー率", fontweight="bold")
    ax_err.set_ylabel("エラー率 (%)")
    ax_err.grid(axis="y", alpha=0.3)

plt.tight_layout()
from session_utils import output_path as _op
plt.savefig(_op("fatigue_analysis.png"), dpi=150, bbox_inches="tight")
print(f"\n保存: {_op('fatigue_analysis.png')}") 
