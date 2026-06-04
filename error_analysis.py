"""
error_analysis.py  ─ エラー誘発パターン分析（⑥ メイン）

「なぜミスが起きるか」を打鍵リズム・指遷移・前後文脈から分析する。

主要な分析:
  1. エラー直前のリズム崩れ  ─ 急加速した直後にミスが集中するか
  2. 危険なキー遷移          ─ どの2鍵の連続でエラーが起きやすいか
  3. 指遷移別エラー率        ─ どの指の組み合わせが危険か
  4. エラーの回復コスト      ─ バックスペース後に速度が戻るまでの打鍵数
  5. エラーのクラスタリング  ─ ミスは連続して起きやすいか
  6. 急加速検出              ─ 「危険速度帯」の閾値を推定

使い方:
  python error_analysis.py                    # 全セッション統合
  python error_analysis.py --session-id 1778934176
  python error_analysis.py --context 5        # エラー前後N打鍵を表示
  python error_analysis.py --speed-ratio 0.5  # 急加速の閾値（平均の何倍）
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
parser.add_argument("--session-id",   default=None)
parser.add_argument("--context",      type=int,   default=4,
                    help="エラー前後に表示する打鍵数")
parser.add_argument("--phrase-ms",    type=int,   default=1000)
parser.add_argument("--speed-ratio",  type=float, default=0.6,
                    help="急加速と判定する閾値（baseline の何倍以下）")
args = parser.parse_args()

PHRASE_MS    = args.phrase_ms
SPEED_RATIO  = args.speed_ratio  # interval < baseline * SPEED_RATIO → 急加速

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
}

# ── データ読み込み ─────────────────────────────────────────────
if args.session_id:
    key_files = [f"session_{args.session_id}_keys.json"]
else:
    key_files = sorted(glob.glob("session_*_keys.json"))

# integrated data があれば指情報も使う
def load_integrated(sid):
    p = f"session_{sid}_integrated.json"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # onset_ms → actual_finger のマップ
    return {r["onset_ms"]: r for r in data if r.get("actual_finger")}

# ── セッションごとの分析 ────────────────────────────────────────
ALL_ERRORS        = []   # エラーの詳細
ALL_BASELINES     = []   # baseline interval
ALL_RECOVERY      = []   # 回復コスト（バックスペース後のN打鍵の平均interval）

SESSION_SUMMARIES = []

for kf in key_files:
    if not os.path.exists(kf):
        continue
    sid = re.search(r"session_(\d+)_keys", kf).group(1)
    with open(kf, encoding="utf-8") as f:
        keylog = json.load(f)

    integ = load_integrated(sid)

    # フレーズ境界を除いた通常打鍵
    valid = [k for k in keylog if 0 < k["interval_ms"] <= PHRASE_MS
             and not k["key"].startswith("Key.")]
    baselines = [k["interval_ms"] for k in valid if not k.get("is_error")]
    baseline_mean   = np.mean(baselines)   if baselines else 1
    baseline_median = np.median(baselines) if baselines else 1

    errors = [k for k in keylog if k.get("is_error")]
    ALL_BASELINES.extend(baselines)

    summary = {
        "sid": sid,
        "total_keys":     len(valid),
        "n_errors":       len(errors),
        "error_rate":     len(errors) / max(len(valid), 1) * 100,
        "baseline_mean":  baseline_mean,
        "baseline_median": baseline_median,
    }
    SESSION_SUMMARIES.append(summary)

    # エラーごとの文脈分析
    all_keys = keylog  # 全打鍵（BSも含む）
    for ki, entry in enumerate(all_keys):
        if not entry.get("is_error"):
            continue

        iv = entry["interval_ms"]

        # 前後の打鍵コンテキスト（BSを除く有効打鍵）
        valid_idx = [i for i, k in enumerate(all_keys) if not k["key"].startswith("Key.")]
        vi = None
        for vi_cand, idx in enumerate(valid_idx):
            if idx == ki:
                vi = vi_cand
                break
        if vi is None:
            continue

        ctx_before = [all_keys[valid_idx[j]] for j in range(max(0, vi-args.context), vi)]
        ctx_after  = [all_keys[valid_idx[j]] for j in range(vi+1, min(len(valid_idx), vi+args.context+1))
                      if not all_keys[valid_idx[j]].get("is_backspace")]

        # 急加速フラグ
        is_rushing = (0 < iv <= PHRASE_MS) and (iv < baseline_mean * SPEED_RATIO)

        # 直前のキーとの遷移
        prev_key = ctx_before[-1]["key"] if ctx_before else None
        prev_iv  = ctx_before[-1]["interval_ms"] if ctx_before else None

        # 指情報
        actual_finger = None
        # onset_ms でマッチング（±100ms）
        if integ:
            ts = entry["timestamp_ms"]
            for oms, r in integ.items():
                if abs(oms - ts) < 100:
                    actual_finger = r.get("actual_finger")
                    break

        # 回復コスト: バックスペース後の打鍵の間隔が元に戻るまで
        recovery_ivs = []
        for k in ctx_after[:4]:
            if 0 < k["interval_ms"] <= PHRASE_MS:
                recovery_ivs.append(k["interval_ms"])

        error_rec = {
            "sid":            sid,
            "key":            entry["key"],
            "interval_ms":    iv,
            "baseline_mean":  baseline_mean,
            "ratio_to_base":  iv / baseline_mean if baseline_mean > 0 else 1,
            "is_rushing":     is_rushing,
            "prev_key":       prev_key,
            "prev_iv":        prev_iv,
            "actual_finger":  actual_finger,
            "ctx_before":     ctx_before,
            "ctx_after":      ctx_after,
            "recovery_ivs":   recovery_ivs,
            "recovery_mean":  np.mean(recovery_ivs) if recovery_ivs else None,
        }
        ALL_ERRORS.append(error_rec)

# ── 出力 ──────────────────────────────────────────────────────
print("=" * 65)
print("  エラー誘発パターン分析")
print("=" * 65)

# セッションサマリー
print(f"\n  セッション別エラー集計")
print(f"  {'Session':>12s}  {'打鍵':>6s}  {'エラー':>6s}  {'エラー率':>8s}  {'avg_iv':>7s}")
print("  " + "-" * 50)
total_err = 0
for s in SESSION_SUMMARIES:
    print(f"  {s['sid']:>12s}  {s['total_keys']:6d}  {s['n_errors']:6d}  "
          f"{s['error_rate']:7.1f}%  {s['baseline_mean']:6.0f}ms")
    total_err += s["n_errors"]

print(f"\n  合計エラー数: {total_err}件  / 全{sum(s['total_keys'] for s in SESSION_SUMMARIES)}打鍵")

if not ALL_ERRORS:
    print("\n  エラーデータなし（今後のセッションで蓄積されます）")
    sys.exit(0)

# ── ① リズム崩れ分析 ─────────────────────────────────────────
print()
print("=" * 65)
print("  ① エラー直前のリズム崩れ（急加速の検出）")
print(f"     急加速 = interval < baseline_mean × {SPEED_RATIO:.1f}")
print("=" * 65)

rushing = [e for e in ALL_ERRORS if e["is_rushing"]]
calm    = [e for e in ALL_ERRORS if not e["is_rushing"]]

print(f"\n  急加速中のエラー: {len(rushing)}/{total_err}件 ({len(rushing)/total_err*100:.0f}%)")
print(f"  通常速度のエラー: {len(calm)}/{total_err}件 ({len(calm)/total_err*100:.0f}%)")

for e in ALL_ERRORS:
    rush_mark = "[急加速!]" if e["is_rushing"] else "[通常]  "
    ratio     = e["ratio_to_base"]
    print(f"\n  {rush_mark}  key=[{e['key']:8s}]  "
          f"interval={e['interval_ms']:4d}ms  "
          f"(baseline={e['baseline_mean']:.0f}ms, ratio={ratio:.2f})")
    print(f"    前後コンテキスト: ", end="")
    for c in e["ctx_before"]:
        print(f"[{c['key']}:{c['interval_ms']}ms]", end=" ")
    print(f"→ [ERR:{e['key']}:{e['interval_ms']}ms] →", end=" ")
    for c in e["ctx_after"][:3]:
        print(f"[{c['key']}:{c['interval_ms']}ms]", end=" ")
    print()
    if e["is_rushing"] and e["ctx_before"]:
        # 急加速の直前N打鍵でintervalが下がってきているか確認
        ivs_trend = [c["interval_ms"] for c in e["ctx_before"]
                     if 0 < c["interval_ms"] <= PHRASE_MS]
        if len(ivs_trend) >= 2:
            slope = ivs_trend[-1] - ivs_trend[0]
            trend = "加速傾向" if slope < -10 else ("減速傾向" if slope > 10 else "安定")
            print(f"    直前のリズム: {ivs_trend} → {trend} (変化={slope:+.0f}ms)")

# ── ② 危険なキー遷移 ─────────────────────────────────────────
print()
print("=" * 65)
print("  ② 危険なキー遷移（エラーが起きた前→現キーの組み合わせ）")
print("=" * 65)

if ALL_ERRORS:
    for e in ALL_ERRORS:
        if e["prev_key"]:
            print(f"  [{e['prev_key'].upper()}] → [{e['key'].upper()}]  "
                  f"interval={e['interval_ms']}ms  rushing={e['is_rushing']}")
    print("\n  ※ データが少ないため、セッションを重ねると統計が充実します")

# ── ③ 指遷移別エラー率 ────────────────────────────────────────
print()
print("=" * 65)
print("  ③ 指遷移別エラー情報（カメラ検出データ）")
print("=" * 65)

finger_errors = [e for e in ALL_ERRORS if e.get("actual_finger")]
if finger_errors:
    for e in finger_errors:
        fname = FINGER_NAMES.get(e["actual_finger"], e["actual_finger"])
        print(f"  エラーキー=[{e['key']}]  使用指={fname}  rushing={e['is_rushing']}")
else:
    print("  指データなし（integrated.json が必要）")

# ── ④ 回復コスト ─────────────────────────────────────────────
print()
print("=" * 65)
print("  ④ エラーからの回復コスト")
print("     バックスペース後、打鍵リズムが元に戻るまでの分析")
print("=" * 65)

for e in ALL_ERRORS:
    bs_iv = None
    # バックスペース自体の間隔
    ki_in_all = None
    for i, k in enumerate(ALL_ERRORS):
        if k is e:
            ki_in_all = i
            break

    print(f"\n  エラー: key=[{e['key']}]  baseline={e['baseline_mean']:.0f}ms")

    if e["recovery_ivs"]:
        rec = e["recovery_ivs"]
        slowdown = np.mean(rec) / e["baseline_mean"] * 100
        print(f"  回復後{len(rec)}打鍵の平均interval: {np.mean(rec):.0f}ms "
              f"(baseline比 {slowdown:.0f}%)")
        print(f"  回復interval列: {[round(v) for v in rec]}")
        if slowdown > 120:
            print(f"  → ミス後のリズム乱れあり（baseline比+{slowdown-100:.0f}%遅い）")
        elif slowdown < 90:
            print(f"  → ミス後は反応が早い傾向")
        else:
            print(f"  → ほぼ即回復")

# ── ⑤ クラスタリング ─────────────────────────────────────────
print()
print("=" * 65)
print("  ⑤ エラーのクラスタリング分析")
print("=" * 65)

# 全セッションでエラーのタイムスタンプ間隔を見る
error_timestamps = [(e["sid"], k["timestamp_ms"])
                    for e in ALL_ERRORS
                    for kf2 in key_files
                    for k in ([e_k for e_k in (json.load(open(kf2, encoding="utf-8"))
                                               if os.path.exists(kf2) else [])
                                if e_k.get("is_error")])
                    ]

print(f"  現在のデータでは{total_err}件のエラーのみ。")
print(f"  クラスタリング分析には最低10件以上のエラーが必要です。")
print(f"  セッションを重ねると分析が充実します。")

# ── ⑥ 急加速の「危険閾値」推定 ───────────────────────────────
print()
print("=" * 65)
print("  ⑥ 急加速の「危険速度帯」推定")
print("=" * 65)

if ALL_BASELINES:
    bm = np.mean(ALL_BASELINES)
    bmd = np.median(ALL_BASELINES)
    print(f"\n  全セッション baseline: 平均={bm:.0f}ms  中央値={bmd:.0f}ms")
    print(f"\n  エラー時の interval（vs baseline）:")
    for e in ALL_ERRORS:
        pct = e["interval_ms"] / bm * 100
        bar = "#" * int(pct / 5)
        print(f"    [{e['key']}]  {e['interval_ms']:4d}ms  = baseline の {pct:.0f}%  {bar}")

    if ALL_ERRORS:
        error_ratios = [e["interval_ms"] / bm for e in ALL_ERRORS
                        if 0 < e["interval_ms"] <= PHRASE_MS]
        if error_ratios:
            mean_ratio = np.mean(error_ratios)
            print(f"\n  推定危険閾値: baseline の {mean_ratio*100:.0f}% 以下 "
                  f"(= {bm * mean_ratio:.0f}ms 以下) で打ち続けるとミスが増える可能性")
            print(f"  → 現在のあなたには「{bm * mean_ratio:.0f}ms ({60000 / (bm * mean_ratio / 5):.0f} WPM相当）以下」が危険速度帯")

# ── 可視化 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.suptitle("エラー誘発パターン分析", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 左上: セッション別エラー率
ax1 = fig.add_subplot(gs[0, 0])
sids  = [s["sid"][-4:] for s in SESSION_SUMMARIES]
rates = [s["error_rate"] for s in SESSION_SUMMARIES]
colors = ["#F44336" if r > 0 else "#4CAF50" for r in rates]
bars = ax1.bar(range(len(sids)), rates, color=colors, alpha=0.8)
for bar, v in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.1f}%", ha="center", fontsize=9)
ax1.set_xticks(range(len(sids))); ax1.set_xticklabels(sids, fontsize=9)
ax1.set_title("セッション別エラー率", fontweight="bold")
ax1.set_ylabel("エラー率 (%)")
ax1.grid(axis="y", alpha=0.3)

# 右上: エラー時のintervalをbaseline分布と比較
ax2 = fig.add_subplot(gs[0, 1])
if ALL_BASELINES:
    ax2.hist(ALL_BASELINES, bins=40, alpha=0.6, color="#2196F3",
             label=f"baseline (N={len(ALL_BASELINES)})", density=True)
    for e in ALL_ERRORS:
        ax2.axvline(e["interval_ms"], color="#F44336", linewidth=2.5,
                    label=f"エラー [{e['key']}] ({e['interval_ms']}ms)")
    ax2.axvline(np.mean(ALL_BASELINES) * SPEED_RATIO, color="orange",
                linestyle="--", linewidth=1.5, label=f"急加速閾値 ({SPEED_RATIO:.0%})")
    ax2.set_title("エラー時のinterval vs baseline分布", fontweight="bold")
    ax2.set_xlabel("打鍵間隔 (ms)")
    ax2.set_ylabel("密度")
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, min(np.percentile(ALL_BASELINES, 95) * 1.5, 600))
    ax2.grid(alpha=0.3)

# 左下: エラー前後のinterval推移
ax3 = fig.add_subplot(gs[1, 0])
if ALL_ERRORS:
    for ei, e in enumerate(ALL_ERRORS):
        ctx = e["ctx_before"][-args.context:]
        ivs = [c["interval_ms"] for c in ctx if 0 < c["interval_ms"] <= PHRASE_MS]
        ivs.append(e["interval_ms"])  # エラー地点
        rec = [v for v in e["recovery_ivs"][:args.context] if 0 < v <= PHRASE_MS]
        ivs.extend(rec)

        x_err = len([c for c in ctx if 0 < c["interval_ms"] <= PHRASE_MS])
        xs = list(range(len(ivs)))
        color = "#F44336" if e["is_rushing"] else "#FF9800"
        ax3.plot(xs, ivs, "o-", color=color, alpha=0.8,
                 label=f"[{e['key']}] (rushing={e['is_rushing']})")
        ax3.axvline(x_err, color="red", linestyle="--", linewidth=1, alpha=0.5)

    if ALL_BASELINES:
        ax3.axhline(np.mean(ALL_BASELINES), color="blue",
                    linestyle=":", linewidth=1, label="baseline平均")
        ax3.axhline(np.mean(ALL_BASELINES) * SPEED_RATIO, color="orange",
                    linestyle=":", linewidth=1, label="急加速閾値")

    ax3.set_title("エラー前後のinterval推移\n(縦破線=エラー地点)", fontweight="bold")
    ax3.set_xlabel("打鍵インデックス（エラー地点基準）")
    ax3.set_ylabel("interval (ms)")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

# 右下: 回復コスト
ax4 = fig.add_subplot(gs[1, 1])
recovery_data = [(e["key"], e["recovery_mean"], e["baseline_mean"])
                 for e in ALL_ERRORS if e["recovery_mean"] is not None]
if recovery_data:
    labels_r = [f"[{k}]" for k, _, _ in recovery_data]
    ratios_r  = [rm/bm*100 for _, rm, bm in recovery_data]
    colors_r  = ["#F44336" if r > 120 else "#4CAF50" for r in ratios_r]
    bars_r = ax4.bar(range(len(labels_r)), ratios_r, color=colors_r, alpha=0.8)
    ax4.axhline(100, color="blue", linestyle="--", linewidth=1, label="baseline=100%")
    for bar, v in zip(bars_r, ratios_r):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 1,
                 f"{v:.0f}%", ha="center", fontsize=10)
    ax4.set_xticks(range(len(labels_r))); ax4.set_xticklabels(labels_r)
    ax4.set_title("エラー後の回復コスト\n(baseline=100%)", fontweight="bold")
    ax4.set_ylabel("回復後interval / baseline (%)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
else:
    ax4.text(0.5, 0.5, "回復データなし", ha="center", va="center", transform=ax4.transAxes)

plt.savefig("error_analysis.png", dpi=150, bbox_inches="tight")
print("\n\n保存: error_analysis.png")
print("\n注意: 現在のエラーデータは少ないため、セッションを重ねるほど")
print("      より信頼性の高い統計が得られます。")
