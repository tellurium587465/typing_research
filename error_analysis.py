"""
error_analysis.py  ─ アルペジオ考慮型エラー誘発パターン分析

アルペジオ（意図的な高速連打）と偶発的な急加速を区別して
「本当に危険なミス」を特定する。

判定フロー:
  1. 直前ペアが arpeggio_map でアルペジオ認定されているか？
     → YES: アルペジオ内エラー（速さ自体は問題でない）
     → NO:  Z-score でペア固有の外れ値かチェック

  2. ペア固有 Z-score = (interval - pair_mean) / pair_std
     → Z < -2 かつ pair.type == "variable" : 危険な急加速
     → Z < -2 かつ pair.type == "normal"   : 偶発的な急加速（中程度のリスク）
     → -2 <= Z : 統計的に正常範囲

使い方:
  python error_analysis.py                    # 全セッション
  python error_analysis.py --session-id 1778934176
  python error_analysis.py --arpeggio-map arpeggio_map.json
"""
import argparse, json, glob, os, re, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--session-id",    default=None)
parser.add_argument("--phrase-ms",     type=int,   default=1000)
parser.add_argument("--arpeggio-map",  default="arpeggio_map.json")
parser.add_argument("--context",       type=int,   default=4)
args = parser.parse_args()

# ── アルペジオマップの読み込み ─────────────────────────────────
arpeggio_map = {}
if os.path.exists(args.arpeggio_map):
    with open(args.arpeggio_map, encoding="utf-8") as f:
        arpeggio_map = json.load(f)
    print(f"アルペジオマップ: {len(arpeggio_map)}ペア読み込み")
else:
    print("※ arpeggio_map.json なし。先に python arpeggio_analysis.py を実行してください")
    print("  グローバル baseline のみで分析します")

def get_pair_info(prev_key, curr_key):
    """ペア情報を取得（なければ None）"""
    key = f"{prev_key}{curr_key}"
    return arpeggio_map.get(key)

def classify_speed(interval_ms, pair_info, global_mean):
    """
    間隔を分類して (label, z_score, explanation) を返す。

    label:
      "arpeggio_normal"  : アルペジオペアの正常範囲
      "arpeggio_extreme" : アルペジオペアでも異常に速い（Z < -3）
      "rush_variable"    : 変動大ペアでの急加速 → 最危険
      "rush_normal"      : 通常ペアでの急加速 → 中リスク
      "rush_unknown"     : ペア不明での急加速（globalで判断）
      "normal"           : 正常範囲
    """
    if pair_info:
        pm, ps = pair_info["mean"], pair_info["std"]
        z = (interval_ms - pm) / max(ps, 1)
        ptype = pair_info["type"]

        if ptype == "arpeggio":
            if z < -3:
                return "arpeggio_extreme", z, f"アルペジオ内だが異常速 (Z={z:.1f})"
            return "arpeggio_normal", z, f"アルペジオ正常範囲 (Z={z:.1f})"
        elif z < -2:
            if ptype == "variable":
                return "rush_variable", z, f"変動大ペアでの急加速 (Z={z:.1f}) ← 最危険"
            else:
                return "rush_normal",   z, f"通常ペアでの急加速 (Z={z:.1f})"
        else:
            return "normal", z, f"正常範囲 (Z={z:.1f})"
    else:
        # ペア不明：グローバル基準
        ratio = interval_ms / global_mean if global_mean > 0 else 1
        if ratio < 0.4:
            return "rush_unknown", None, f"ペア不明・global比{ratio:.0%}"
        return "normal", None, "ペア不明・通常範囲"

# ── データ読み込み ─────────────────────────────────────────────
if args.session_id:
    key_files = [f"session_{args.session_id}_keys.json"]
else:
    key_files = sorted(glob.glob("session_*_keys.json"))

def load_integrated(sid):
    p = f"session_{sid}_integrated.json"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {r["onset_ms"]: r for r in data if r.get("actual_finger")}

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
}

ALL_ERRORS    = []
ALL_BASELINES = []
SESSION_SUMMARIES = []

for kf in key_files:
    if not os.path.exists(kf):
        continue
    sid = re.search(r"session_(\d+)_keys", kf).group(1)
    with open(kf, encoding="utf-8") as f:
        keylog = json.load(f)

    integ = load_integrated(sid)

    normal = [k for k in keylog
              if not k.get("is_backspace")
              and not k["key"].startswith("Key.")
              and 0 < k["interval_ms"] <= args.phrase_ms]

    baselines = [k["interval_ms"] for k in normal if not k.get("is_error")]
    ALL_BASELINES.extend(baselines)
    global_mean   = np.mean(baselines)   if baselines else 1
    global_median = np.median(baselines) if baselines else 1

    errors = [k for k in keylog if k.get("is_error")]
    SESSION_SUMMARIES.append({
        "sid": sid, "total": len(normal), "n_errors": len(errors),
        "error_rate": len(errors) / max(len(normal), 1) * 100,
        "global_mean": global_mean,
    })

    all_keys = keylog
    valid_idx = [i for i, k in enumerate(all_keys) if not k["key"].startswith("Key.")]

    for ki, entry in enumerate(all_keys):
        if not entry.get("is_error"):
            continue

        iv = entry["interval_ms"]

        # コンテキスト取得
        vi = next((j for j, idx in enumerate(valid_idx) if idx == ki), None)
        if vi is None:
            continue

        ctx_before = [all_keys[valid_idx[j]] for j in range(max(0, vi-args.context), vi)]
        ctx_after  = [all_keys[valid_idx[j]] for j in
                      range(vi+1, min(len(valid_idx), vi+args.context+2))
                      if not all_keys[valid_idx[j]].get("is_backspace")]

        prev_key = ctx_before[-1]["key"] if ctx_before else None

        # ペア情報とスピード分類
        pair_info = get_pair_info(prev_key, entry["key"]) if prev_key else None
        label, z_score, explanation = classify_speed(iv, pair_info, global_mean)

        # 指情報
        actual_finger = None
        if integ:
            ts = entry["timestamp_ms"]
            actual_finger = next(
                (r.get("actual_finger") for oms, r in integ.items() if abs(oms - ts) < 100),
                None
            )

        # 回復コスト
        recovery_ivs = [k["interval_ms"] for k in ctx_after[:4]
                        if 0 < k["interval_ms"] <= args.phrase_ms]

        ALL_ERRORS.append({
            "sid": sid,
            "key": entry["key"],
            "interval_ms":   iv,
            "global_mean":   global_mean,
            "prev_key":      prev_key,
            "pair_info":     pair_info,
            "label":         label,
            "z_score":       z_score,
            "explanation":   explanation,
            "actual_finger": actual_finger,
            "ctx_before":    ctx_before,
            "ctx_after":     ctx_after,
            "recovery_ivs":  recovery_ivs,
            "recovery_mean": np.mean(recovery_ivs) if recovery_ivs else None,
        })

# ── 出力 ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("  エラー誘発パターン分析（アルペジオ考慮版）")
print("=" * 65)

print(f"\n  {'Session':>12s}  {'打鍵':>6s}  {'エラー':>6s}  {'エラー率':>8s}  {'avg_iv':>7s}")
print("  " + "-" * 50)
for s in SESSION_SUMMARIES:
    print(f"  {s['sid']:>12s}  {s['total']:6d}  {s['n_errors']:6d}  "
          f"{s['error_rate']:7.1f}%  {s['global_mean']:6.0f}ms")

total_err = len(ALL_ERRORS)
print(f"\n  合計エラー: {total_err}件")

if not ALL_ERRORS:
    print("\n  エラーデータなし")
    sys.exit(0)

# ── 各エラーの詳細分析 ────────────────────────────────────────
print()
print("=" * 65)
print("  各エラーの詳細分類")
print("=" * 65)

LABEL_ICONS = {
    "arpeggio_normal":  "[ARPEGGIO] アルペジオ正常",
    "arpeggio_extreme": "[EXTREME ] アルペジオ内異常速",
    "rush_variable":    "[DANGER  ] 変動大ペア急加速（最危険）",
    "rush_normal":      "[WARNING ] 通常ペア急加速",
    "rush_unknown":     "[CAUTION ] 不明ペア急加速",
    "normal":           "[OK      ] 正常範囲",
}

for e in ALL_ERRORS:
    icon = LABEL_ICONS.get(e["label"], e["label"])
    pi   = e["pair_info"]

    print(f"\n  {icon}")
    print(f"  key=[{e['key']}]  interval={e['interval_ms']}ms"
          f"  (global_mean={e['global_mean']:.0f}ms)")

    if pi:
        print(f"  ペア [{e['prev_key']}→{e['key']}]: "
              f"mean={pi['mean']:.0f}ms  std={pi['std']:.0f}ms  "
              f"CV={pi['cv']:.2f}  type={pi['type']}  N={pi['n']}")
        if e["z_score"] is not None:
            print(f"  Z-score={e['z_score']:.2f}  → {e['explanation']}")
    else:
        print(f"  ペア不明  → {e['explanation']}")

    print(f"  前後: ", end="")
    for c in e["ctx_before"][-3:]:
        print(f"[{c['key']}:{c['interval_ms']}ms]", end=" ")
    print(f"→ [ERR:{e['key']}:{e['interval_ms']}ms] →", end=" ")
    for c in e["ctx_after"][:3]:
        print(f"[{c['key']}:{c['interval_ms']}ms]", end=" ")
    print()

    if e["actual_finger"]:
        fname = FINGER_NAMES.get(e["actual_finger"], e["actual_finger"])
        print(f"  使用指: {fname} ({e['actual_finger']})")

    if e["recovery_mean"]:
        ratio = e["recovery_mean"] / e["global_mean"] * 100
        print(f"  回復コスト: {e['recovery_mean']:.0f}ms (baseline比 {ratio:.0f}%)")

# ── ラベル別集計 ───────────────────────────────────────────────
print()
print("=" * 65)
print("  エラー種別サマリー")
print("=" * 65)

label_counts = defaultdict(int)
for e in ALL_ERRORS:
    label_counts[e["label"]] += 1

for label, icon in LABEL_ICONS.items():
    cnt = label_counts.get(label, 0)
    if cnt > 0:
        pct = cnt / total_err * 100
        print(f"  {icon:30s}: {cnt:3d}件 ({pct:.0f}%)")

print()
print("  【解釈】")
arp_n  = label_counts.get("arpeggio_normal", 0)
rush_n = label_counts.get("rush_variable", 0) + label_counts.get("rush_normal", 0)
unk_n  = label_counts.get("rush_unknown", 0)

if arp_n > 0:
    print(f"  アルペジオ内エラー {arp_n}件: 速さ自体は問題ではない。")
    print(f"    → キーの選択（どの文字を打つか）の判断ミスが原因と考えられる。")
if rush_n > 0:
    print(f"  急加速エラー {rush_n}件: 意図せず速くなりすぎた可能性。")
    print(f"    → そのペアの練習または意識的なペース維持が有効。")
if unk_n > 0:
    print(f"  不明ペアエラー {unk_n}件: データが少ないペア。")
    print(f"    → セッションを重ねると分類精度が上がります。")

# ── 可視化 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.suptitle("エラー分析（アルペジオ考慮版）", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# 左: ペア分布 with エラーポイント
ax1 = fig.add_subplot(gs[0])
if ALL_BASELINES and arpeggio_map:
    # 全ペアの分布
    means = [v["mean"] for v in arpeggio_map.values()]
    cvs   = [v["cv"]   for v in arpeggio_map.values()]
    types = [v["type"] for v in arpeggio_map.values()]
    type_colors = {"arpeggio":"#F44336","fast":"#FF9800","variable":"#9C27B0","normal":"#BBBBBB"}
    for t, col in type_colors.items():
        idxs = [i for i, tp in enumerate(types) if tp == t]
        if idxs:
            ax1.scatter([means[i] for i in idxs], [cvs[i] for i in idxs],
                        c=col, alpha=0.5, s=20, label=t)

    # エラーポイントを重ねて表示
    for e in ALL_ERRORS:
        pi = e["pair_info"]
        if pi:
            marker = "X" if "rush" in e["label"] else "o"
            ec_color = "red" if "rush" in e["label"] else "blue"
            ax1.scatter(pi["mean"], pi["cv"], c=ec_color, s=120,
                        marker=marker, zorder=5, edgecolors="black", linewidth=1.5)
            ax1.annotate(f"{e['prev_key']}->{e['key']}\n({e['interval_ms']}ms)",
                         (pi["mean"], pi["cv"]), fontsize=7.5,
                         xytext=(5, 5), textcoords="offset points",
                         color=ec_color)

    ax1.axvline(100,  color="gray",   linestyle="--", linewidth=1)
    ax1.axhline(0.35, color="red",    linestyle=":",  linewidth=1)
    ax1.axhline(0.60, color="purple", linestyle=":",  linewidth=1)
    ax1.set_xlabel("平均間隔 (ms)")
    ax1.set_ylabel("CV = σ/μ")
    ax1.set_title("ペア分類マップ\nX=危険エラー  o=アルペジオエラー", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

# 右: エラー種別の円グラフ
ax2 = fig.add_subplot(gs[1])
label_display = {
    "arpeggio_normal":  "アルペジオ\n正常",
    "arpeggio_extreme": "アルペジオ\n異常速",
    "rush_variable":    "急加速\n(変動大)",
    "rush_normal":      "急加速\n(通常)",
    "rush_unknown":     "急加速\n(不明)",
    "normal":           "正常範囲",
}
pie_colors = {
    "arpeggio_normal":  "#2196F3",
    "arpeggio_extreme": "#FF9800",
    "rush_variable":    "#F44336",
    "rush_normal":      "#FF5722",
    "rush_unknown":     "#FFC107",
    "normal":           "#4CAF50",
}
pie_data = [(label_display.get(l, l), cnt, pie_colors.get(l, "#888888"))
            for l, cnt in label_counts.items() if cnt > 0]
if pie_data:
    lbls = [p[0] for p in pie_data]
    vals = [p[1] for p in pie_data]
    cols = [p[2] for p in pie_data]
    wedges, texts, autotexts = ax2.pie(
        vals, labels=lbls, colors=cols,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    ax2.set_title("エラー種別の内訳", fontweight="bold")

plt.savefig("error_analysis.png", dpi=150, bbox_inches="tight")
print("\n保存: error_analysis.png")
