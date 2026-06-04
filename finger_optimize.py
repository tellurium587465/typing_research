"""
finger_optimize.py  ─ 距離を考慮した実践的な運指最適化

「物理的に届く指の中で、どの切り替えが一番速くなるか」を答える。

フィルタ二重構造:
  1. 距離フィルタ  : ホームから --reach キー幅以内の指のみ対象
  2. 信頼性フィルタ: あるキーで全検出の --min-pct % 未満の指は
                    カメラ誤検出と判断して除外する

  例: Iキーは R3=93%, R4=5%, R2=2% → min-pct=15 なら R3 のみ有効

使い方:
  python finger_optimize.py                    # 全セッション統合
  python finger_optimize.py --session-id 1778934176
  python finger_optimize.py --reach 2.5        # リーチ閾値を広げる
  python finger_optimize.py --min-pct 10       # 信頼性閾値を下げる
  python finger_optimize.py --min-n 5
"""
import argparse, json, glob, os, re, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from session_utils import get_session_files, find_session_integrated_files, list_all_session_ids
from constants import FINGER_NAMES, FINGER_ORDER, PHRASE_BOUNDARY_MS
from plot_utils import setup_jp_font
setup_jp_font()

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None)
parser.add_argument("--min-n",   type=int,   default=8,
                    help="最小サンプル数（絶対値）")
parser.add_argument("--min-pct", type=float, default=15.0,
                    help="最小検出割合 %（これ未満は誤検出として除外）")
parser.add_argument("--phrase-ms", type=int, default=1000)
parser.add_argument("--reach",  type=float, default=2.2,
                    help="指のホームからキーまでの最大距離（キー幅単位）")
args = parser.parse_args()

FINGER_NAMES = {
    "L1":"左小指","L2":"左薬指","L3":"左中指","L4":"左人差し指",
    "R4":"右人差し指","R3":"右中指","R2":"右薬指","R1":"右小指",
}
FINGER_ORDER = ["L1","L2","L3","L4","R4","R3","R2","R1"]

# ── キーボードグリッドの読み込み ────────────────────────────────
GRID_FILE = "keyboard_grid.json"
if not os.path.exists(GRID_FILE):
    print(f"キーボードグリッドが見つかりません: {GRID_FILE}")
    sys.exit(1)

with open(GRID_FILE) as f:
    grid = json.load(f)

KEY_POS  = grid["key_positions"]   # {"a": [x, y], ...}
KEY_W_PX = grid["key_w_px"]
KEY_H_PX = grid["key_h_px"]

# 各指のホームキー（タッチタイピング基準）
FINGER_HOME = {
    "L1": "a", "L2": "s", "L3": "d", "L4": "f",
    "R4": "j", "R3": "k", "R2": "l", "R1": "p",   # R1 はセミコロンが近いが p で近似
}

def key_distance(key_a: str, key_b: str) -> float:
    """2キー間の距離をキー幅単位で返す。座標不明なら inf。"""
    pa = KEY_POS.get(key_a)
    pb = KEY_POS.get(key_b)
    if pa is None or pb is None:
        return float("inf")
    dx = (pa[0] - pb[0]) / KEY_W_PX
    dy = (pa[1] - pb[1]) / KEY_H_PX
    return (dx**2 + dy**2) ** 0.5

def finger_reach(finger: str, key: str) -> float:
    """指のホームからキーまでの距離（キー幅単位）"""
    home = FINGER_HOME.get(finger)
    if home is None:
        return float("inf")
    return key_distance(home, key)

def is_practical(finger: str, key: str, threshold: float) -> bool:
    """この指でこのキーを打つのが物理的に実用的かどうか"""
    return finger_reach(finger, key) <= threshold

# ── データ読み込み ─────────────────────────────────────────────
if args.session_id:
    sf_s = get_session_files(args.session_id)
    int_files = [sf_s["integrated"]] if os.path.exists(sf_s["integrated"]) else []
else:
    int_files = find_session_integrated_files()

all_strokes = []
for fpath in int_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    sid = session_id_from_path(fpath)
    cam = [r for r in data
           if r["source"] == "camera"
           and r.get("actual_finger")
           and 0 < r["interval_ms"] <= args.phrase_ms
           and len(r["key"]) == 1
           and r["key"].isalpha()]
    for r in cam:
        r["_session"] = sid
    all_strokes.extend(cam)

print(f"分析対象: {len(all_strokes)}打鍵")

# ── 連続打鍵ペア ───────────────────────────────────────────────
pairs = []
for i in range(1, len(all_strokes)):
    prev, curr = all_strokes[i-1], all_strokes[i]
    if prev["_session"] != curr["_session"]:
        continue
    if curr["interval_ms"] > args.phrase_ms:
        continue
    pairs.append({
        "prev_finger": prev["actual_finger"],
        "curr_finger": curr["actual_finger"],
        "prev_key":    prev["key"],
        "curr_key":    curr["key"],
        "interval_ms": curr["interval_ms"],
        "same_finger": prev["actual_finger"] == curr["actual_finger"],
        "same_hand":   prev["actual_finger"][0] == curr["actual_finger"][0],
    })

# ── ① 同指連打コスト ──────────────────────────────────────────
same_ivs = [p["interval_ms"] for p in pairs if p["same_finger"]]
diff_ivs = [p["interval_ms"] for p in pairs if not p["same_finger"]]
alt_ivs  = [p["interval_ms"] for p in pairs if not p["same_hand"]]
sh_ivs   = [p["interval_ms"] for p in pairs if p["same_hand"] and not p["same_finger"]]

print()
print("=" * 60)
print("  ① 同指連打の干渉コスト")
print("=" * 60)
base = np.median(alt_ivs) if alt_ivs else 1
for label, ivs in [("左右交互（基準）", alt_ivs),
                   ("同手・異指",       sh_ivs),
                   ("同指連打",         same_ivs)]:
    if not ivs:
        continue
    md   = np.median(ivs)
    diff = md - base
    tag  = f"+{diff:.0f}ms 遅い" if diff > 2 else ("同等" if abs(diff) <= 2 else f"{diff:.0f}ms")
    print(f"  {label:12s}  N={len(ivs):5d}  中央値={md:5.0f}ms  ({tag})")

# 指別コスト
print()
print("  指別の同指連打コスト（多い順）")
print("  " + "-" * 50)
fcosts = []
fsame = defaultdict(list); fother = defaultdict(list)
for p in pairs:
    f = p["curr_finger"]
    (fsame if p["same_finger"] else fother)[f].append(p["interval_ms"])
for f in FINGER_ORDER:
    s, o = fsame.get(f,[]), fother.get(f,[])
    if len(s) < 3 or len(o) < 3: continue
    cost = np.median(s) - np.median(o)
    fcosts.append((f, cost, len(s)))
fcosts.sort(key=lambda x: -x[1])
for f, cost, ns in fcosts:
    name = FINGER_NAMES.get(f, f)
    bar  = "#" * int(max(cost,0)/5)
    print(f"  {f}  {name:12s}  +{cost:5.0f}ms  (同指{ns:3d}回)  {bar}")

# ── ② 指遷移コスト行列 ────────────────────────────────────────
print()
print("=" * 60)
print("  ② 指遷移コスト行列（中央値 ms）")
print("     行=前の指, 列=今の指  [対角]=同指連打")
print("=" * 60)

trans = defaultdict(list)
for p in pairs:
    trans[(p["prev_finger"], p["curr_finger"])].append(p["interval_ms"])

active = [f for f in FINGER_ORDER
          if any(len(trans.get((f,g),[])) >= 2 or
                 len(trans.get((g,f),[])) >= 2
                 for g in FINGER_ORDER)]

hdr = "         " + "".join(f"{f:>7s}" for f in active)
print("  " + hdr)
print("  " + "-" * len(hdr))
matrix = {}
for pf in active:
    row = f"  {pf:7s}"
    for cf in active:
        ivs = trans.get((pf,cf),[])
        if len(ivs) >= 2:
            v = np.median(ivs)
            matrix[(pf,cf)] = v
            cell = f"[{v:4.0f}]" if pf==cf else f"{v:5.0f}ms"
        else:
            cell = "     -"
        row += f" {cell}"
    print(row)

# ── ③ キーごとの指別速度（距離フィルタ付き） ─────────────────
print()
print("=" * 60)
print(f"  ③ キーごとの指別速度分析")
print(f"     フィルタ: 距離<={args.reach:.1f}u  かつ  全検出の{args.min_pct:.0f}%以上")
print(f"     ※ 少数派検出（{args.min_pct:.0f}%未満）はカメラ誤検出として除外")
print("=" * 60)

key_finger_ivs = defaultdict(lambda: defaultdict(list))
for s in all_strokes:
    key_finger_ivs[s["key"]][s["actual_finger"]].append(s["interval_ms"])

# 各キーについて実用的な代替指の速度を比較
practical_recs = []

for key in sorted(key_finger_ivs.keys()):
    # 全検出数（このキーの合計）
    total_n = sum(len(ivs) for ivs in key_finger_ivs[key].values())

    # フィルタ:
    #   1. 距離フィルタ（ホームから reach 以内）
    #   2. 信頼性フィルタ（全検出の min_pct % 以上 かつ 絶対数 min_n 以上）
    finger_data = {}
    for f, ivs in key_finger_ivs[key].items():
        dist = finger_reach(f, key)
        pct  = len(ivs) / total_n * 100
        if (len(ivs) >= args.min_n
                and pct >= args.min_pct
                and dist <= args.reach):
            finger_data[f] = (ivs, dist, pct)

    if len(finger_data) < 2:
        continue

    stats = {
        f: {
            "med":  np.median(ivs),
            "mean": np.mean(ivs),
            "n":    len(ivs),
            "dist": dist,
            "pct":  pct,
        }
        for f, (ivs, dist, pct) in finger_data.items()
    }

    best_f    = min(stats, key=lambda f: stats[f]["med"])
    worst_f   = max(stats, key=lambda f: stats[f]["med"])
    most_used = max(stats, key=lambda f: stats[f]["n"])

    gain = stats[worst_f]["med"] - stats[best_f]["med"]

    if gain < 5:   # 5ms 未満は誤差範囲
        continue

    practical_recs.append({
        "key":       key,
        "best_f":    best_f,
        "worst_f":   worst_f,
        "most_used": most_used,
        "gain_ms":   gain,
        "stats":     stats,
    })

    print(f"  [{key.upper()}] ", end="")
    for f, st in sorted(stats.items(), key=lambda x: x[1]["med"]):
        best_mark = "★" if f == best_f else " "
        print(f"{best_mark}{f}:{st['med']:.0f}ms(N={st['n']},{st['pct']:.0f}%,d={st['dist']:.1f}u)",
              end="  ")
    print(f"→ 差:{gain:.0f}ms")

# ── ④ 切り替え推奨ランキング（実用的なもののみ） ──────────────
print()
print("=" * 60)
print("  ④ 切り替え推奨ランキング（距離フィルタ適用済み）")
print("=" * 60)

switch_recs = []
for r in practical_recs:
    most  = r["most_used"]
    best  = r["best_f"]
    if most == best:
        continue
    st_most = r["stats"][most]
    st_best = r["stats"][best]
    gain    = st_most["med"] - st_best["med"]
    freq    = st_most["n"]
    switch_recs.append({
        "key":          r["key"],
        "from_f":       most,
        "to_f":         best,
        "gain_ms":      gain,
        "freq":         freq,
        "total_gain":   gain * freq,
        "from_med":     st_most["med"],
        "to_med":       st_best["med"],
        "from_dist":    st_most["dist"],
        "to_dist":      st_best["dist"],
        "from_pct":     st_most["pct"],
        "to_pct":       st_best["pct"],
    })

switch_recs.sort(key=lambda x: -x["total_gain"])

if switch_recs:
    print(f"\n  {'#':>2s}  Key  現在の指（距離/割合）         推奨の指（距離/割合）         改善   頻度   合計改善")
    print("  " + "-" * 82)
    for i, r in enumerate(switch_recs, 1):
        fn  = FINGER_NAMES.get(r["from_f"], r["from_f"])
        tn  = FINGER_NAMES.get(r["to_f"],   r["to_f"])
        fd  = r["from_dist"]; fpct = r["from_pct"]
        td  = r["to_dist"];   tpct = r["to_pct"]
        print(f"  {i:2d}  [{r['key'].upper()}]  "
              f"{fn:12s}({fd:.1f}u,{fpct:.0f}%) → {tn:12s}({td:.1f}u,{tpct:.0f}%)  "
              f"{r['gain_ms']:+5.0f}ms  {r['freq']:4d}回  {r['total_gain']:6.0f}ms")
        print(f"         {r['from_med']:.0f}ms → {r['to_med']:.0f}ms")
else:
    print("  実用的な切り替え推奨なし（誤検出フィルタ後）")

# ── ⑤ 同指連打ホットスポット ─────────────────────────────────
print()
print("=" * 60)
print("  ⑤ 同指連打ホットスポット（避けるべきキーペア）")
print("     この2キーを続けて同じ指で打つと大きく遅くなる")
print("=" * 60)

hotspot = defaultdict(list)
nonhot  = defaultdict(list)
for p in pairs:
    pk = (p["prev_key"], p["curr_key"])
    if p["same_finger"]:
        hotspot[(pk[0], pk[1], p["curr_finger"])].append(p["interval_ms"])
    else:
        nonhot[pk].append(p["interval_ms"])

spots = []
for (pk, ck, finger), ivs in hotspot.items():
    if len(ivs) < 2:
        continue
    same_med = np.median(ivs)
    ref_ivs  = nonhot.get((pk, ck), [])
    ref_med  = np.median(ref_ivs) if len(ref_ivs) >= 2 else None
    delay    = (same_med - ref_med) if ref_med else same_med * 0.6
    # 解決策: もう一方の手 or 別の指で打てるか？
    alt_key = ck  # ckを別の指で打てるか
    alt_fingers = [f for f in FINGER_ORDER
                   if f != finger and is_practical(f, alt_key, args.reach + 0.3)]
    spots.append({
        "pair": f"{pk.upper()}→{ck.upper()}", "finger": finger,
        "same_med": same_med, "ref_med": ref_med,
        "delay": delay, "n": len(ivs),
        "alt_fingers": alt_fingers,
    })

spots.sort(key=lambda x: -x["delay"])
print(f"  {'キーペア':8s}  {'指':12s}  {'同指':>8s}  {'異指':>8s}  {'遅延':>7s}  N  代替案")
print("  " + "-" * 70)
for h in spots[:10]:
    fn  = FINGER_NAMES.get(h["finger"], h["finger"])
    ref = f"{h['ref_med']:.0f}ms" if h["ref_med"] else "   N/A"
    alts = ",".join(h["alt_fingers"]) if h["alt_fingers"] else "なし"
    print(f"  {h['pair']:8s}  {fn:12s}  {h['same_med']:7.0f}ms  {ref:>8s}  "
          f"{h['delay']:+6.0f}ms  {h['n']:2d}  → 代替:{alts}")

# ── グラフ ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("運指最適化（距離フィルタ適用）", fontsize=13, fontweight="bold")

# 左: 切り替え推奨の効果
ax = axes[0]
if switch_recs:
    top = switch_recs[:8]
    labels = []
    for r in top:
        fd = r["from_dist"]; td = r["to_dist"]
        labels.append(f"[{r['key'].upper()}]\n{r['from_f']}({fd:.1f}u)\n→{r['to_f']}({td:.1f}u)")
    x = np.arange(len(top))
    gains  = [r["gain_ms"] for r in top]
    totals = [r["total_gain"]/100 for r in top]
    b1 = ax.bar(x - 0.2, gains,  0.38, label="1打あたり改善(ms)", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + 0.2, totals, 0.38, label="合計改善(×100ms)",  color="#FF9800", alpha=0.85)
    for bar, v in zip(b1, gains):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f"+{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title("切り替え推奨TOP8\n（物理的に届く指のみ）", fontsize=10)
    ax.set_ylabel("改善量")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
else:
    ax.text(0.5, 0.5, "実用的な推奨なし", ha="center", va="center", transform=ax.transAxes)

# 右: 各指のホームからキーへの距離マップ（よく使うキーのみ）
ax2 = axes[1]
common_keys = [s["key"] for s in all_strokes]
from collections import Counter
top_keys = [k for k, _ in Counter(common_keys).most_common(15)
            if k.isalpha() and KEY_POS.get(k)]

if top_keys:
    n_keys = len(top_keys)
    n_fing = len(active)
    mat = np.full((n_fing, n_keys), np.nan)
    for fi, fing in enumerate(active):
        for ki, key in enumerate(top_keys):
            d = finger_reach(fing, key)
            mat[fi, ki] = d

    im = ax2.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=4, aspect="auto")
    ax2.set_xticks(range(n_keys))
    ax2.set_xticklabels([k.upper() for k in top_keys], fontsize=9)
    ax2.set_yticks(range(n_fing))
    ax2.set_yticklabels(active, fontsize=9)
    ax2.set_title(f"指-キー間距離（u単位）\n赤={args.reach:.1f}u超=実用外  緑=近い", fontsize=9)
    for fi in range(n_fing):
        for ki in range(n_keys):
            d = mat[fi, ki]
            if not np.isnan(d):
                color = "white" if d > 2.5 else "black"
                ax2.text(ki, fi, f"{d:.1f}", ha="center", va="center",
                         fontsize=7, color=color)
    plt.colorbar(im, ax=ax2, label="距離（キー幅単位）")

    # リーチ閾値のラインを表示
    ax2.axhline(-0.5, color="white", linewidth=0.5)

plt.tight_layout()
from session_utils import output_path as _op
plt.savefig(_op("finger_optimize.png"), dpi=150, bbox_inches="tight")
print(f"\n保存: {_op('finger_optimize.png')}") 
