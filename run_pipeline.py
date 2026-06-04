"""
run_pipeline.py  ─ タイピング研究 エンドツーエンド分析パイプライン

使い方:
  python run_pipeline.py                         # 最新セッションを自動検出・全工程実行
  python run_pipeline.py --session-id 1778934176 # セッション指定
  python run_pipeline.py --from onset            # onsetステップから再実行
  python run_pipeline.py --skip-pose             # pose_analysis をスキップ（高速化）

ステップ:
  1. extract_frame    ─ キーボード検出用フレームを動画から自動抽出
  2. keyboard_detect  ─ キーボード位置・キーグリッドを検出して保存
  3. pose_analysis    ─ MediaPipe で全フレームの指先座標を取得
  4. onset_detection  ─ 打鍵音から打鍵タイミングを検出してキーログと照合
  5. integrate        ─ キーログ × 音 × カメラ映像を1打鍵ずつ統合
  6. export_excel     ─ 結果を Excel に出力
  7. report           ─ サマリーをターミナルに表示
  8. arpeggio         ─ 全セッションのアルペジオペアを更新（蓄積型）
  9. error_analysis   ─ アルペジオ考慮型エラー誘発パターン分析
"""
import argparse
import os
import sys
import glob
import subprocess
import time

import cv2
import numpy as np

from session_utils import get_session_files

# ───────────────────────────────────────────────
# 引数
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="タイピング研究 分析パイプライン")
parser.add_argument("--session-id", default=None,
                    help="セッションID（省略時は最新を自動検出）")
parser.add_argument("--skip-pose", action="store_true",
                    help="pose_analysis をスキップ（既に完了済みの場合など）")
parser.add_argument("--from", dest="from_step",
                    choices=["extract","keyboard","pose","onset","integrate",
                             "export","report","arpeggio","error"],
                    default="extract",
                    help="指定ステップから再実行（途中から再開するとき）")
parser.add_argument("--no-excel",   action="store_true", help="Excel出力をスキップ")
parser.add_argument("--no-report",  action="store_true", help="最終レポートをスキップ")
parser.add_argument("--no-arpeggio",action="store_true", help="アルペジオ分析をスキップ")
parser.add_argument("--no-error",   action="store_true", help="エラー分析をスキップ")
args = parser.parse_args()

_sf = get_session_files(args.session_id)
SESSION_ID = _sf["session_id"]
VIDEO_FILE = _sf["video"]
SESSION_ARGS = ["--session-id", SESSION_ID]

STEP_ORDER = ["extract","keyboard","pose","onset","integrate",
              "export","report","arpeggio","error"]

def should_run(step):
    return STEP_ORDER.index(step) >= STEP_ORDER.index(args.from_step)

def banner(title, step_n, total=9):
    print(f"\n{'='*55}")
    print(f"  Step {step_n}/{total}: {title}")
    print(f"{'='*55}")

def run_script(script, extra_args=None, allow_fail=False):
    """サブプロセスでスクリプトを実行する。失敗時は終了（allow_fail=True なら続行）"""
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [警告] {script} が終了コード {result.returncode} で終了（{elapsed:.1f}s）")
        if not allow_fail:
            sys.exit(result.returncode)
    else:
        print(f"  [完了] {elapsed:.1f}s")
    return result.returncode == 0

print(f"\n{'='*55}")
print(f"  タイピング研究 分析パイプライン")
print(f"  セッション: {SESSION_ID}")
print(f"{'='*55}")

# ───────────────────────────────────────────────
# Step 1: キーボード検出用フレーム抽出
# ───────────────────────────────────────────────
if should_run("extract"):
    banner("キーボード検出用フレーム抽出", 1)
    if not os.path.exists(VIDEO_FILE):
        print(f"  エラー: 動画ファイルが見つかりません: {VIDEO_FILE}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_FILE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS)

    candidates = list(range(3, min(int(fps_v * 2), total_frames), 3))
    best_frame, best_score = None, -1
    for fn in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([125, 40, 40]), np.array([165, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
        score = int(np.sum(cv2.bitwise_or(mask1, mask2) > 0))
        if score > best_score:
            best_score = score
            best_frame = (fn, frame.copy())
    cap.release()

    if best_frame is None:
        print("  エラー: フレーム取得失敗")
        sys.exit(1)

    fn, frame = best_frame
    cv2.imwrite("frame_10.png", frame)
    print(f"  選択フレーム: {fn}番  バックライトスコア: {best_score:,}")

# ───────────────────────────────────────────────
# Step 2: キーボードグリッド検出
# ───────────────────────────────────────────────
if should_run("keyboard"):
    banner("キーボードグリッド検出", 2)
    run_script("keyboard_detect.py", SESSION_ARGS + ["--headless"])

# ───────────────────────────────────────────────
# Step 3: 骨格推定（MediaPipe）
# ───────────────────────────────────────────────
if should_run("pose"):
    if args.skip_pose:
        print("\n  Step 3: pose_analysis をスキップ（--skip-pose）")
    else:
        banner("指先・骨格推定（MediaPipe）", 3)
        print("  ※ 動画の長さに比例して時間がかかります")
        run_script("pose_analysis.py", SESSION_ARGS)

# ───────────────────────────────────────────────
# Step 4: 音響 onset 検出
# ───────────────────────────────────────────────
if should_run("onset"):
    banner("打鍵音 onset 検出", 4)
    run_script("onset_detection.py", SESSION_ARGS + ["--no-plot"], allow_fail=True)
    # 音声が取れていなくても integrate でキーログ代用するため続行

# ───────────────────────────────────────────────
# Step 5: 3ソース統合
# ───────────────────────────────────────────────
if should_run("integrate"):
    banner("キーログ × 音 × カメラ の統合", 5)
    run_script("integrate.py", SESSION_ARGS)

# ───────────────────────────────────────────────
# Step 6: Excel 出力
# ───────────────────────────────────────────────
if should_run("export") and not args.no_excel:
    banner("Excel 出力", 6)
    run_script("export_excel.py", SESSION_ARGS, allow_fail=True)

# ───────────────────────────────────────────────
# Step 7: サマリーレポート
# ───────────────────────────────────────────────
if should_run("report") and not args.no_report:
    banner("分析レポート", 7)
    run_script("report.py", ["--session-id", SESSION_ID, "--detail"])

# ───────────────────────────────────────────────
# Step 8: アルペジオマップ更新（全セッション蓄積型）
# ───────────────────────────────────────────────
if should_run("arpeggio") and not args.no_arpeggio:
    banner("アルペジオペアマップ更新（全セッション）", 8, total=9)
    print("  全セッションのキーペア統計を再計算します（蓄積型）")
    run_script("arpeggio_analysis.py", allow_fail=True)

# ───────────────────────────────────────────────
# Step 9: エラー誘発パターン分析
# ───────────────────────────────────────────────
if should_run("error") and not args.no_error:
    banner("エラー誘発パターン分析（アルペジオ考慮）", 9, total=9)
    run_script("error_analysis.py",
               ["--session-id", SESSION_ID] if args.session_id else [],
               allow_fail=True)

# ───────────────────────────────────────────────
# 完了サマリ
# ───────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  パイプライン完了  セッション: {SESSION_ID}")
print(f"{'='*55}")
print("\n生成ファイル:")
for pattern in [
    f"session_{SESSION_ID}_integrated.json",
    f"session_{SESSION_ID}_keyboard_grid.json",
    f"typing_analysis_{SESSION_ID}.xlsx",
    "keyboard_detect.png", "onset_result.png",
    "arpeggio_analysis.png", "arpeggio_map.json",
    "error_analysis.png",
]:
    for f in glob.glob(pattern):
        size = os.path.getsize(f)
        print(f"  {f}  ({size/1024:.0f} KB)")
