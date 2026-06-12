"""
consolidated_analysis.py ─ 全セッション横断・統合分析（標準ライブラリのみ）

これまで stats_test.py / weakness.py / trend.py / fatigue_analysis.py /
arpeggio_analysis.py に分散していた「キーログ・トリグラムだけで計算できる」
分析を 1 本にまとめ、全 clean セッションをプールして一括で出す統合版。

設計方針:
  - numpy / pandas / scipy / matplotlib を一切使わない（Python 標準ライブラリのみ）
    → どの環境でも `python consolidated_analysis.py` だけで走る。
      （カメラ・MediaPipe を使う運指分析とは独立して、データ層だけで完結する）
  - Mann-Whitney U / Kruskal-Wallis / Cohen's d / chi-square p値 を自前実装
  - フレーズ境界（interval > 1000ms）は除外済みの *_clean データを入力にする

入力（スクリプトと同じフォルダから自動収集）:
  jasp_trigram_*_clean.csv   … トリグラム（3連打）パターン別の打鍵間隔
  keylog_*_clean.json        … 打鍵 1 件ごとの interval / error

出力:
  ターミナル表示
  consolidated_report.txt        … 全結果のテキストレポート
  jasp_trigram_ALL_clean.csv     … 全セッションをプールしたトリグラム CSV
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations

# Windows のコンソール（cp932）でも日本語を文字化けさせない
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PHRASE_BOUNDARY_MS = 1000
VOWELS = set("aiueo")
STANDARD_FINGER = {
    "q": "L1", "w": "L2", "e": "L3", "r": "L4", "t": "L4",
    "y": "R4", "u": "R4", "i": "R3", "o": "R2", "p": "R1",
    "a": "L1", "s": "L2", "d": "L3", "f": "L4", "g": "L4",
    "h": "R4", "j": "R4", "k": "R3", "l": "R2",
    "z": "L1", "x": "L2", "c": "L3", "v": "L4", "b": "L4",
    "n": "R4", "m": "R4",
}
FINGER_NAMES = {
    "L1": "左小指", "L2": "左薬指", "L3": "左中指", "L4": "左人差し指",
    "R4": "右人差し指", "R3": "右中指", "R2": "右薬指", "R1": "右小指",
}

# ════════════════════════════════════════════════════════════════
#  統計関数（標準ライブラリのみで実装）
# ════════════════════════════════════════════════════════════════

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2

def stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))

def _rankdata(values):
    """同順位は平均ランク。タイ補正用に sum(t^3 - t) も返す。"""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_term = 0.0
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1始まり
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks, tie_term

def _norm_sf(z):
    """標準正規分布の上側確率（両側 p 用に 2*sf(|z|)）。"""
    return 0.5 * math.erfc(z / math.sqrt(2))

def mann_whitney_u(x, y):
    """
    Mann-Whitney U 検定（タイ補正つき正規近似）。
    戻り値: U, p値(両側), rank-biserial 効果量 r
    """
    n1, n2 = len(x), len(y)
    ranks, tie_term = _rankdata(list(x) + list(y))
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return u, 1.0, 0.0
    z = (u - mu) / math.sqrt(sigma2)
    p = 2 * _norm_sf(abs(z))
    r = 1 - 2 * u / (n1 * n2)  # rank-biserial（向きは u1基準）
    r = abs(2 * u1 / (n1 * n2) - 1)
    return u, min(p, 1.0), r

def _gammp(a, x):
    """下側正規化不完全ガンマ関数 P(a,x)（Numerical Recipes）。"""
    if x <= 0:
        return 0.0
    if x < a + 1:  # 級数展開
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(200):
            ap += 1
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-12:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:  # 連分数
        b = x + 1 - a
        c = 1e300
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-300:
                d = 1e-300
            c = b + an / c
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-12:
                break
        q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
        return 1.0 - q

def chi2_sf(x, k):
    """カイ二乗分布の上側確率（自由度 k）。"""
    if x <= 0:
        return 1.0
    return 1.0 - _gammp(k / 2.0, x / 2.0)

def kruskal_wallis(groups):
    """Kruskal-Wallis 検定（タイ補正つき）。戻り値: H, df, p"""
    all_vals = []
    for g in groups:
        all_vals.extend(g)
    n = len(all_vals)
    ranks, tie_term = _rankdata(all_vals)
    idx = 0
    rank_sums = []
    for g in groups:
        rsum = sum(ranks[idx:idx + len(g)])
        rank_sums.append(rsum)
        idx += len(g)
    h = 12.0 / (n * (n + 1)) * sum(rs ** 2 / len(g) for rs, g in zip(rank_sums, groups)) - 3 * (n + 1)
    correction = 1 - tie_term / (n ** 3 - n)
    if correction > 0:
        h /= correction
    df = len(groups) - 1
    return h, df, chi2_sf(h, df)

def cohen_d(x, y):
    """プールSDによる Cohen's d。"""
    s1, s2 = stdev(x), stdev(y)
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return (mean(x) - mean(y)) / pooled if pooled > 0 else 0.0

def effect_label(d):
    a = abs(d)
    if a >= 0.8:
        return "大"
    if a >= 0.5:
        return "中"
    if a >= 0.2:
        return "小"
    return "無視可"

# ════════════════════════════════════════════════════════════════
#  データ読み込み
# ════════════════════════════════════════════════════════════════

def load_trigrams():
    rows = []
    files = sorted(glob.glob(os.path.join(HERE, "jasp_trigram_*_clean.csv")))
    files = [f for f in files if "ALL" not in os.path.basename(f)]
    for fp in files:
        with open(fp, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    r["avg_interval_ms"] = float(r["avg_interval_ms"])
                except (ValueError, KeyError):
                    continue
                if "spans_boundary" in r and str(r["spans_boundary"]).lower() == "true":
                    continue
                if not (0 < r["avg_interval_ms"] <= PHRASE_BOUNDARY_MS):
                    continue
                rows.append(r)
    return rows, files

def load_keylogs():
    """セッションIDごとに打鍵リストを返す。"""
    sessions = {}
    files = sorted(glob.glob(os.path.join(HERE, "keylog_*_clean.json")))
    files = [f for f in files if "with_errors" not in os.path.basename(f)]
    for fp in files:
        sid = os.path.basename(fp).replace("keylog_", "").replace("_clean.json", "")
        with open(fp, encoding="utf-8") as f:
            sessions[sid] = json.load(f)
    return sessions, files

# ════════════════════════════════════════════════════════════════
#  レポート生成
# ════════════════════════════════════════════════════════════════

OUT = []
def emit(line=""):
    print(line)
    OUT.append(line)

def section(title):
    emit()
    emit("=" * 70)
    emit(f"  {title}")
    emit("=" * 70)

def main():
    trigrams, tg_files = load_trigrams()
    sessions, kl_files = load_keylogs()

    emit("タイピング研究 ─ 全セッション統合分析レポート")
    emit(f"トリグラム入力: {len(tg_files)}ファイル / {len(trigrams)}行（境界除外後）")
    emit(f"キーログ入力:   {len(kl_files)}ファイル / "
         f"{sum(len(v) for v in sessions.values())}打鍵")

    # ── A. セッション間トレンド + 疲労 ────────────────────────────
    section("A. セッション別トレンド（速度・ミス率・セッション内疲労）")
    emit(f"  {'session':14s} {'打鍵':>6s} {'中央間隔':>8s} {'平均間隔':>8s} "
         f"{'WPM*':>6s} {'ミス率':>6s} {'疲労Δ':>7s}")
    emit("  " + "-" * 62)
    for sid in sorted(sessions):
        keys = sessions[sid]
        ivs = [k["interval_ms"] for k in keys
               if not k.get("is_backspace") and 0 < k.get("interval_ms", 0) <= PHRASE_BOUNDARY_MS]
        if not ivs:
            continue
        errs = sum(1 for k in keys if k.get("is_error"))
        med = median(ivs)
        # WPM* = 1打鍵=1文字, 5文字=1語 として中央間隔から換算
        wpm = 60000.0 / (med * 5) if med else 0
        err_rate = errs / len(keys) * 100
        # 疲労: 前半1/3 vs 後半1/3 の中央間隔差
        third = max(len(ivs) // 3, 1)
        fatigue = median(ivs[-third:]) - median(ivs[:third])
        emit(f"  {sid:14s} {len(keys):6d} {med:7.0f}ms {mean(ivs):7.0f}ms "
             f"{wpm:6.0f} {err_rate:5.1f}% {fatigue:+6.0f}ms")
    emit()
    emit("  * WPM = 60000 / (中央打鍵間隔 × 5)。フレーズ待機を除いた瞬間速度の目安。")
    emit("  * 疲労Δ = 後半1/3 − 前半1/3 の中央間隔。＋なら終盤で失速、−なら加速。")

    # ── B. トリグラム3パターンの統計検定（全セッションプール）─────
    section("B. 運指パターン別 打鍵速度の統計検定（全セッションをプール）")
    PAT = ["左右交互", "同手3連", "同指連打"]
    groups = {p: [r["avg_interval_ms"] for r in trigrams if r.get("pattern_type") == p]
              for p in PAT}
    groups = {p: v for p, v in groups.items() if len(v) >= 10}

    emit(f"  {'パターン':10s} {'N':>5s} {'平均':>8s} {'中央値':>8s} {'SD':>7s}")
    emit("  " + "-" * 44)
    for p, v in groups.items():
        emit(f"  {p:10s} {len(v):5d} {mean(v):7.1f}ms {median(v):7.1f}ms {stdev(v):6.1f}ms")

    if len(groups) >= 2:
        h, df, p_kw = kruskal_wallis(list(groups.values()))
        emit()
        emit(f"  Kruskal-Wallis: H={h:.3f}  df={df}  p={p_kw:.3e}  "
             f"→ {'群間に有意差あり' if p_kw < 0.05 else '有意差なし'}")
        emit()
        emit(f"  {'対比較':24s} {'p値*':>10s} {'有意':>5s} {'d':>7s} {'効果量':>6s} {'差':>9s}")
        emit("  " + "-" * 66)
        pairs = list(combinations(groups.keys(), 2))
        for g1, g2 in pairs:
            v1, v2 = groups[g1], groups[g2]
            _, p_raw, _ = mann_whitney_u(v1, v2)
            p_adj = min(p_raw * len(pairs), 1.0)  # Bonferroni
            d = cohen_d(v1, v2)
            sig = "***" if p_adj < .001 else "**" if p_adj < .01 else "*" if p_adj < .05 else "n.s."
            diff = mean(v1) - mean(v2)
            emit(f"  {g1+' vs '+g2:24s} {p_adj:10.3e} {sig:>5s} "
                 f"{d:+7.3f} {effect_label(d):>6s} {diff:+7.1f}ms")
        emit()
        emit("  * p値は Bonferroni 補正済み（多重比較）。MWUはノンパラ＝分布の正規性を仮定しない。")

    # ── C. キー別の苦手スコア（キーログのみ・カメラ不要）──────────
    section("C. キー別 苦手スコア（遅延×ばらつき×ミス／カメラ不要版）")
    per_key = defaultdict(lambda: {"iv": [], "err": 0, "n": 0})
    for keys in sessions.values():
        for k in keys:
            ch = k.get("key", "")
            if len(ch) != 1 or ch not in STANDARD_FINGER:
                continue
            per_key[ch]["n"] += 1
            if k.get("is_error"):
                per_key[ch]["err"] += 1
            iv = k.get("interval_ms", 0)
            if not k.get("is_backspace") and 0 < iv <= PHRASE_BOUNDARY_MS:
                per_key[ch]["iv"].append(iv)

    MIN_N = 8
    keystats = []
    for ch, st in per_key.items():
        if st["n"] < MIN_N or len(st["iv"]) < 3:
            continue
        med = median(st["iv"])
        cv = stdev(st["iv"]) / mean(st["iv"]) if mean(st["iv"]) else 0
        err = st["err"] / st["n"]
        keystats.append({"key": ch, "n": st["n"], "med": med, "cv": cv, "err": err})

    if keystats:
        # 正規化（0..1）して合成
        def norm(vals):
            lo, hi = min(vals), max(vals)
            return lambda x: (x - lo) / (hi - lo) if hi > lo else 0.0
        nmed = norm([k["med"] for k in keystats])
        ncv = norm([k["cv"] for k in keystats])
        nerr = norm([k["err"] for k in keystats])
        for k in keystats:
            # 遅さ50% × ばらつき30% × ミス20%（TypeCoach の重み付けに準拠）
            k["score"] = 0.5 * nmed(k["med"]) + 0.3 * ncv(k["cv"]) + 0.2 * nerr(k["err"])
        keystats.sort(key=lambda x: -x["score"])
        emit(f"  {'Key':4s} {'担当指':9s} {'苦手':>6s} {'中央遅延':>8s} {'CV':>6s} "
             f"{'ミス率':>6s} {'N':>5s}")
        emit("  " + "-" * 52)
        for i, k in enumerate(keystats):
            star = "★" if i < 5 else " "
            fin = FINGER_NAMES.get(STANDARD_FINGER[k["key"]], "?")
            emit(f"{star} {k['key'].upper():3s} {fin:9s} {k['score']:6.3f} "
                 f"{k['med']:7.0f}ms {k['cv']:6.2f} {k['err']*100:5.1f}% {k['n']:5d}")
        emit()
        top5 = " ".join(f"[{k['key'].upper()}]" for k in keystats[:5])
        emit(f"  練習優先キー TOP5: {top5}")

    # ── D. 同指連打の最悪ビッグラム（矯正ドリル対象）──────────────
    section("D. 同指ビッグラム（同じ指で続けて打つ＝最も遅い組み合わせ）")
    bigrams = defaultdict(list)
    for keys in sessions.values():
        prev = None
        for k in keys:
            ch = k.get("key", "")
            iv = k.get("interval_ms", 0)
            if len(ch) == 1 and ch in STANDARD_FINGER and prev and prev in STANDARD_FINGER \
                    and not k.get("is_backspace") and 0 < iv <= PHRASE_BOUNDARY_MS:
                if STANDARD_FINGER[ch] == STANDARD_FINGER[prev] and ch != prev:
                    bigrams[prev + ch].append(iv)
            prev = ch
    same_fin = [(bg, median(ivs), len(ivs)) for bg, ivs in bigrams.items() if len(ivs) >= 4]
    same_fin.sort(key=lambda x: -x[1])
    if same_fin:
        all_iv = [iv for keys in sessions.values() for k in keys
                  if 0 < (iv := k.get("interval_ms", 0)) <= PHRASE_BOUNDARY_MS
                  and not k.get("is_backspace")]
        base = median(all_iv)
        emit(f"  全体中央間隔（基準）= {base:.0f}ms")
        emit(f"  {'ビッグラム':10s} {'担当指':9s} {'中央間隔':>8s} {'基準比':>8s} {'N':>5s}")
        emit("  " + "-" * 48)
        for bg, med, n in same_fin[:10]:
            fin = FINGER_NAMES.get(STANDARD_FINGER[bg[0]], "?")
            emit(f"  {bg.upper():10s} {fin:9s} {med:7.0f}ms {med-base:+6.0f}ms {n:5d}")
        emit()
        emit("  → これらを『連打→実単語』交互ドリルに入れると矯正効率が高い。")

    # ── E. 母音パターン別（ローマ字日本語入力のリズム）────────────
    section("E. 母音パターン別 打鍵速度（CVC=子母子 など）")
    vp = defaultdict(list)
    for r in trigrams:
        pat = r.get("vowel_pattern", "")
        if pat:
            vp[pat].append(r["avg_interval_ms"])
    vp_rows = [(p, mean(v), median(v), len(v)) for p, v in vp.items() if len(v) >= 10]
    vp_rows.sort(key=lambda x: x[1])
    emit(f"  {'母音pattern':12s} {'平均':>8s} {'中央値':>8s} {'N':>5s}")
    emit("  " + "-" * 38)
    for p, mu, md, n in vp_rows:
        emit(f"  {p:12s} {mu:7.1f}ms {md:7.1f}ms {n:5d}")
    emit()
    emit("  V=母音 C=子音。交互（CVC/VCV）が速く、連続子音（CCV等）は遅い傾向を確認できる。")

    # ── プール CSV 出力 ───────────────────────────────────────────
    if trigrams:
        out_csv = os.path.join(HERE, "jasp_trigram_ALL_clean.csv")
        cols = list(trigrams[0].keys())
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in trigrams:
                w.writerow(r)
        emit()
        emit(f"プールCSV出力: {os.path.basename(out_csv)}（{len(trigrams)}行）")

    # ── レポート保存 ──────────────────────────────────────────────
    report_path = os.path.join(HERE, "consolidated_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    print()
    print(f"レポート保存: {report_path}")


if __name__ == "__main__":
    main()
