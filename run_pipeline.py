"""
パイプライン自動実行スクリプト
最新のセッションファイルを自動検出して以下を順番に実行する：
1. pose_analysis.py
2. onset_detection.py
3. integrate.py
"""
import os
import glob
import subprocess
import sys
import re

# -----------------------------------------------
# 最新セッションを自動検出
# -----------------------------------------------
key_files = glob.glob("session_*_keys.json")
if not key_files:
    print("セッションファイルが見つかりません")
    sys.exit(1)

# タイムスタンプが最大のものを選ぶ
latest = max(key_files, key=lambda f: int(re.search(r"session_(\d+)_keys", f).group(1)))
session_id = re.search(r"session_(\d+)_keys", latest).group(1)
print(f"最新セッション: {session_id}")

# -----------------------------------------------
# 各スクリプトのセッションIDを書き換えて実行
# -----------------------------------------------
def run_script(script, replacements):
    """スクリプトのセッションIDを一時的に書き換えて実行"""
    with open(script, encoding="utf-8") as f:
        original = f.read()

    modified = original
    for old, new in replacements.items():
        modified = modified.replace(old, new)

    # 一時ファイルに書き出して実行
    tmp = f"_tmp_{script}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(modified)

    print(f"\n{'='*40}")
    print(f"実行中: {script}")
    print(f"{'='*40}")
    result = subprocess.run([sys.executable, tmp], capture_output=False)

    os.remove(tmp)
    return result.returncode == 0

# セッションIDの置換パターンを動的に生成
# 既存のセッションIDを検出
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
    if current_id is None:
        print(f"スキップ: {script} にセッションIDが見つかりません")
        continue

    if current_id == session_id:
        print(f"\n{script} は既に最新セッション（{session_id}）です")
        replacements = {}
    else:
        print(f"\n{script}: {current_id} → {session_id} に更新")
        replacements = {current_id: session_id}

    ok = run_script(script, replacements)
    if not ok:
        print(f"エラー: {script} が失敗しました。処理を中断します。")
        sys.exit(1)

print(f"\n{'='*40}")
print(f"パイプライン完了！セッション: {session_id}")
print(f"{'='*40}")
print(f"生成ファイル:")
for f in glob.glob(f"session_{session_id}*.json"):
    size = os.path.getsize(f)
    print(f"  {f} ({size:,}bytes)")