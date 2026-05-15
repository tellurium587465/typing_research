import json
import glob
import numpy as np

# -----------------------------------------------
# 外れ値処理：最初・最後の待ち時間除去 + 統計的外れ値除去
# -----------------------------------------------

TRIM_START_N = 3   # 最初の3打鍵を除外（ESC・Enter待ち）
TRIM_END_N   = 3   # 最後の3打鍵を除外（ESC・Enter待ち）
OUTLIER_STD  = 3   # 平均±3SD以外を除外

keylog_files = glob.glob("keylog_*_with_errors.json") + [
    f for f in glob.glob("keylog_*.json") if "with_errors" not in f
]

for fname in keylog_files:
    with open(fname, encoding="utf-8") as f:
        keylog = json.load(f)

    # 通常キーのみ抽出
    normal = [
        e for e in keylog
        if not e.get("is_backspace", False)
        and not e["key"].startswith("Key.")
    ]

    before = len(normal)

    # 最初・最後をトリム
    normal = normal[TRIM_START_N:len(normal)-TRIM_END_N]

    # interval_msの統計的外れ値を除去（平均±3SD）
    intervals = np.array([e["interval_ms"] for e in normal], dtype=float)
    mean, std = np.mean(intervals), np.std(intervals)
    lower, upper = mean - OUTLIER_STD * std, mean + OUTLIER_STD * std
    normal = [e for e in normal if lower <= e["interval_ms"] <= upper]

    after = len(normal)
    print(f"{fname}: {before}件 → {after}件（除去: {before-after}件）")
    print(f"  interval_ms: mean={mean:.1f} std={std:.1f} 範囲=[{lower:.1f}, {upper:.1f}]")

    # クリーニング済みとして保存
    out_fname = fname.replace(".json", "_clean.json")
    with open(out_fname, "w", encoding="utf-8") as f:
        json.dump(normal, f, ensure_ascii=False, indent=2)
    print(f"  保存: {out_fname}")