"""
パイプライン自動実行スクリプト
最新セッションを自動検出して以下を順番に実行：
1. extract_frame.py    → 手のないフレームを取り出す
2. keyboard_detect.py  → キーボードグリッドを検出・保存
3. pose_analysis.py    → 骨格推定
4. onset_detection.py  → 音響onset検出
5. integrate.py        → 3ソース統合
"""
import os
import glob
import subprocess
import sys
import re
import cv2
import numpy as np

# -----------------------------------------------
# 最新セッションを自動検出
# -----------------------------------------------
key_files = glob.glob("session_*_keys.json")
if not key_files:
    print("セッションファイルが見つかりません")
    sys.exit(1)

latest = max(key_files, key=lambda f: int(re.search(r"session_(\d+)_keys", f).group(1)))
session_id = re.search(r"session_(\d+)_keys", latest).group(1)
print(f"最新セッション: {session_id}")

VIDEO_FILE = f"session_{session_id}_video.avi"

# -----------------------------------------------
# Step 1：手のないフレームを自動選択
# -----------------------------------------------
print("\n" + "="*40)
print("Step 1: 手のないフレームを選択")
print("="*40)

cap = cv2.VideoCapture(VIDEO_FILE)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"総フレーム数: {total}  FPS: {fps:.1f}")

# 最初の2秒以内のフレームを候補にする（タイピング前）
candidate_frames = list(range(5, min(int(fps * 2), total), 5))

best_frame = None
best_score = -1

for fn in candidate_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        continue
    # 紫色（バックライト）の面積でキーボードが見えているか評価
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([125,40,40]), np.array([165,255,255]))
    mask2 = cv2.inRange(hsv, np.array([160,40,40]), np.array([180,255,255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    score = np.sum(mask > 0)
    if score > best_score:
        best_score = score
        best_frame = (fn, frame.copy())

cap.release()

if best_frame is None:
    print("フレーム取得失敗")
    sys.exit(1)

fn, frame = best_frame
cv2.imwrite("frame_10.png", frame)
print(f"選択フレーム: {fn}番（バックライト面積スコア: {best_score}）")

# -----------------------------------------------
# Step 2：キーボードグリッド検出
# -----------------------------------------------
print("\n" + "="*40)
print("Step 2: キーボードグリッド検出")
print("="*40)

result = subprocess.run([sys.executable, "keyboard_detect.py"], capture_output=False)
if result.returncode != 0:
    print("keyboard_detect.py が失敗しました")
    sys.exit(1)

# -----------------------------------------------
# Step 3〜5：各スクリプトを実行
# -----------------------------------------------
def run_script(script, replacements):
    with open(script, encoding="utf-8") as f:
        original = f.read()
    modified = original
    for old, new in replacements.items():
        modified = modified.replace(old, new)
    tmp = f"_tmp_{script}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(modified)
    print(f"\n{'='*40}")
    print(f"実行中: {script}")
    print(f"{'='*40}")
    result = subprocess.run([sys.executable, tmp], capture_output=False)
    os.remove(tmp)
    return result.returncode == 0

def detect_current_session(script):
    with open(script, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"session_(\d+)", content)
    return match.group(1) if match else None

scripts = [
    "pose_analysis.py",
    "onset_detection.py",
    "integrate.py",
]

for script in scripts:
    if not os.path.exists(script):
        print(f"スキップ: {script} が見つかりません")
        continue
    current_id = detect_current_session(script)
    replacements = {current_id: session_id} if current_id and current_id != session_id else {}
    ok = run_script(script, replacements)
    if not ok:
        print(f"エラー: {script} が失敗しました。処理を中断します。")
        sys.exit(1)

print(f"\n{'='*40}")
print(f"パイプライン完了！セッション: {session_id}")
print(f"{'='*40}")
print("生成ファイル:")
for f in glob.glob(f"session_{session_id}*.json"):
    size = os.path.getsize(f)
    print(f"  {f} ({size:,}bytes)")