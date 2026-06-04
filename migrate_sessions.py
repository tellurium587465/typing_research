"""
migrate_sessions.py  ─ 既存セッションデータを新ディレクトリ構造に移行

旧構造:  session_{id}_keys.json（ルートに散在）
新構造:  sessions/{id}/keys.json

使い方:
  python migrate_sessions.py           # 確認のみ（移動しない）
  python migrate_sessions.py --apply   # 実際に移動する
"""
import argparse, glob, os, re, shutil

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true", help="実際にファイルを移動する")
args = parser.parse_args()

SESSIONS_DIR = "sessions"

# 旧ファイル名 suffix → 新ファイル名
SUFFIX_MAP = {
    "_keys.json":                "keys.json",
    "_audio.wav":                "audio.wav",
    "_video.avi":                "video.avi",
    "_pose.json":                "pose.json",
    "_pose_backup.json":         "pose_backup.json",
    "_onset_matched.json":       "onset_matched.json",
    "_integrated.json":          "integrated.json",
    "_keyboard_grid.json":       "keyboard_grid.json",
    "_meta.json":                "meta.json",
}
# typing_analysis_{id}.xlsx と jasp_trigram_{id}.csv は別パターン
SPECIAL_PATTERNS = [
    (r"typing_analysis_(\d+)\.xlsx",   "typing_analysis.xlsx"),
    (r"jasp_trigram_(\d+)\.csv",       "jasp_trigram.csv"),
    (r"frame_(\d+)\.png",              "frame.png"),
]

# 旧構造のセッションIDを探す
old_files = glob.glob("session_*_keys.json")
session_ids = sorted(set(
    re.search(r"session_(\d+)_keys", f).group(1) for f in old_files
))

print(f"移行対象セッション: {len(session_ids)}件")
print(f"モード: {'[適用]' if args.apply else '[ドライラン]'}\n")

total_moved = 0
for sid in session_ids:
    dest_dir = os.path.join(SESSIONS_DIR, sid)
    print(f"  session {sid} → {dest_dir}/")

    if args.apply:
        os.makedirs(dest_dir, exist_ok=True)

    # session_{id}_*.* パターン
    for suffix, new_name in SUFFIX_MAP.items():
        old_path = f"session_{sid}{suffix}"
        if os.path.exists(old_path):
            new_path = os.path.join(dest_dir, new_name)
            print(f"    {old_path:50s} → {new_name}")
            if args.apply:
                shutil.move(old_path, new_path)
            total_moved += 1

    # 特殊パターン（typing_analysis_*.xlsx など）
    for pattern, new_name in SPECIAL_PATTERNS:
        for old_path in glob.glob(pattern.replace(r"(\d+)", sid)):
            if os.path.exists(old_path):
                new_path = os.path.join(dest_dir, new_name)
                print(f"    {old_path:50s} → {new_name}")
                if args.apply:
                    shutil.move(old_path, new_path)
                total_moved += 1

    # jasp_trigram_{sid}_*.csv も対象
    for old_path in glob.glob(f"jasp_trigram_{sid}*.csv"):
        if os.path.exists(old_path):
            new_path = os.path.join(dest_dir, "jasp_trigram.csv")
            print(f"    {old_path:50s} → jasp_trigram.csv")
            if args.apply:
                shutil.move(old_path, new_path)
            total_moved += 1

    print()

# output/ ディレクトリに移動すべき横断ファイル
OUTPUT_DIR = "output"
CROSS_SESSION_FILES = [
    "trend.png", "weakness_heatmap.png", "arpeggio_map.json",
    "arpeggio_analysis.png", "error_analysis.png", "fatigue_analysis.png",
    "stats_test_result.png", "stats_test_result.csv", "finger_optimize.png",
    "jasp_trigram_ALL.csv", "jasp_trigram_ALL._boin_html.html",
    "jasp_trigram_ALL_hand_.html",
]
print(f"  output/ に移動する横断ファイル:")
for fname in CROSS_SESSION_FILES:
    if os.path.exists(fname):
        new_path = os.path.join(OUTPUT_DIR, fname)
        print(f"    {fname} → output/{fname}")
        if args.apply:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            shutil.move(fname, new_path)
        total_moved += 1

print(f"\n合計 {total_moved} ファイル{'を移動しました' if args.apply else 'を移動予定'}。")
if not args.apply:
    print("実際に移動するには --apply を付けて実行してください。")
