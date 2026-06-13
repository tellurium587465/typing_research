"""
validate_finger.py ─ 運指推定の妥当性検証（ground truth との突き合わせ）

「足りていないこと⑤：推定運指の正解ラベルが無く、誤検出率を定量化できない」
への回答。少数の打鍵を人手でラベル付けして推定と突き合わせ、
  ・全体の正解率
  ・品質クラス別の正解率（uncertain は本当に当たっていないか／nonstandard は本物か）
  ・取り違え行列（特に R3↔R2、R3↔R4）
  ・信頼度の較正（当たり/外れで confidence に差が出るか）
を出す。これで finger_select の confidence/品質分類や z キュー・順序補正の
効果を客観的に評価できるようになる。

─ 使い方 ─────────────────────────────────────────────
  # 1) ラベル付け用テンプレートを書き出す（情報量の多い打鍵を優先抽出）
  python validate_finger.py --sample 30 --session-id 1778934176
      → sessions/<id>/validate_labels.csv が出る。
        各行の onset_ms を頼りに録画を見て true_finger 列を埋める。
        指コード: L1 L2 L3 L4 R4 R3 R2 R1（小指→人差し→…）

  # 2) 埋めた CSV を採点する
  python validate_finger.py --score --session-id 1778934176

依存なし（標準ライブラリのみ）。コア採点関数は test から import 可能。
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict, Counter

FINGER_NAMES = {
    "L1": "左小指", "L2": "左薬指", "L3": "左中指", "L4": "左人差し指",
    "R4": "右人差し指", "R3": "右中指", "R2": "右薬指", "R1": "右小指",
}


def compute_validation(records):
    """
    records: [{"predicted":str, "true":str, "quality":str, "confidence":float|None}, ...]
             true が None/空 の行は未ラベルとして無視する。
    戻り値: 指標の dict（純関数・テスト可能）
    """
    labeled = [r for r in records if r.get("true")]
    n = len(labeled)
    if n == 0:
        return {"n": 0}

    correct = sum(1 for r in labeled if r["predicted"] == r["true"])

    by_q = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in labeled:
        q = r.get("quality", "?")
        by_q[q]["n"] += 1
        by_q[q]["correct"] += int(r["predicted"] == r["true"])
    by_quality = {
        q: {"n": v["n"], "acc": v["correct"] / v["n"] if v["n"] else 0.0}
        for q, v in by_q.items()
    }

    confusion = Counter()
    for r in labeled:
        if r["predicted"] != r["true"]:
            confusion[(r["true"], r["predicted"])] += 1

    conf_correct = [r["confidence"] for r in labeled
                    if r.get("confidence") is not None and r["predicted"] == r["true"]]
    conf_wrong = [r["confidence"] for r in labeled
                  if r.get("confidence") is not None and r["predicted"] != r["true"]]

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n,
        "overall_acc": correct / n,
        "by_quality": by_quality,
        "confusion": dict(confusion),
        "mean_conf_correct": avg(conf_correct),
        "mean_conf_wrong": avg(conf_wrong),
    }


def print_report(m):
    if m.get("n", 0) == 0:
        print("ラベル付き行がありません。CSV の true_finger 列を埋めてください。")
        return
    print("=" * 60)
    print(f"  運指推定の妥当性検証（ラベル {m['n']} 件）")
    print("=" * 60)
    print(f"  全体正解率: {m['overall_acc']*100:.1f}%")

    print("\n  品質クラス別の正解率")
    print("  " + "-" * 44)
    order = ["standard", "nonstandard", "uncertain", "no_detection", "?"]
    for q in order:
        if q in m["by_quality"]:
            v = m["by_quality"][q]
            print(f"    {q:12s}: {v['acc']*100:5.1f}%  (N={v['n']})")
    print("    ※ uncertain の正解率が低く、standard が高ければ品質分類は妥当。")
    print("    ※ nonstandard の正解率が高ければ“誤検出ではなく本物の癖”が裏づく。")

    if m["confusion"]:
        print("\n  取り違え（正解→推定）多い順")
        print("  " + "-" * 44)
        for (t, p), c in sorted(m["confusion"].items(), key=lambda x: -x[1]):
            tn, pn = FINGER_NAMES.get(t, t), FINGER_NAMES.get(p, p)
            flag = ""
            if {t, p} == {"R3", "R2"}:
                flag = "  ← R3↔R2 既知の取り違え"
            elif {t, p} == {"R3", "R4"}:
                flag = "  ← R3↔R4"
            print(f"    {t}({tn})→{p}({pn}): {c}件{flag}")

    cc, cw = m.get("mean_conf_correct"), m.get("mean_conf_wrong")
    if cc is not None or cw is not None:
        print("\n  信頼度の較正（confidence の平均）")
        print("  " + "-" * 44)
        print(f"    当たり: {cc:.3f}" if cc is not None else "    当たり: N/A")
        print(f"    外れ  : {cw:.3f}" if cw is not None else "    外れ  : N/A")
        if cc is not None and cw is not None:
            verdict = "良好（当たりの方が高信頼）" if cc > cw else "要見直し（差が出ていない）"
            print(f"    → {verdict}")


# ════════════════════════════════════════════════════════════════
#  CLI（integrated.json を扱う部分。import 時には実行されない）
# ════════════════════════════════════════════════════════════════
def _labels_path(sf):
    return os.path.join(sf["session_dir"], "validate_labels.csv")


def do_sample(sf, n):
    if not os.path.exists(sf["integrated"]):
        print(f"integrated.json が見つかりません: {sf['integrated']}")
        sys.exit(1)
    with open(sf["integrated"], encoding="utf-8") as f:
        data = json.load(f)
    cam = [r for r in data if r.get("source") == "camera"]
    # 情報量の多い順に層化抽出: uncertain → nonstandard → standard
    pri = {"uncertain": 0, "nonstandard": 1, "standard": 2}
    cam.sort(key=lambda r: pri.get(r.get("finger_quality"), 3))
    picked = cam[:n]
    out = _labels_path(sf)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onset_ms", "key", "std_finger", "predicted_finger",
                    "finger_quality", "confidence", "true_finger"])
        for r in picked:
            w.writerow([r["onset_ms"], r["key"], r.get("std_finger", ""),
                        r.get("actual_finger", ""), r.get("finger_quality", ""),
                        r.get("detection_confidence", ""), ""])
    print(f"ラベル用テンプレートを書き出しました: {out}")
    print(f"  {len(picked)} 件。録画を見て true_finger 列を埋めてください。")
    print("  指コード: L1 L2 L3 L4 R4 R3 R2 R1")


def do_score(sf):
    out = _labels_path(sf)
    if not os.path.exists(out):
        print(f"ラベルCSVが見つかりません: {out}\n先に --sample で作成してください。")
        sys.exit(1)
    records = []
    with open(out, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            conf = row.get("confidence", "")
            records.append({
                "predicted": row.get("predicted_finger", "").strip(),
                "true": row.get("true_finger", "").strip(),
                "quality": row.get("finger_quality", "").strip(),
                "confidence": float(conf) if conf not in ("", None) else None,
            })
    print_report(compute_validation(records))


if __name__ == "__main__":
    from session_utils import get_session_files
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--sample", type=int, default=None, help="ラベル用に抽出する件数")
    ap.add_argument("--score", action="store_true", help="埋めたCSVを採点")
    args = ap.parse_args()
    sf = get_session_files(args.session_id)
    if args.sample:
        do_sample(sf, args.sample)
    elif args.score:
        do_score(sf)
    else:
        print("使い方: --sample N でテンプレ作成 → 録画を見て埋める → --score で採点")
