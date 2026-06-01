"""
セッションID自動検出ユーティリティ
各スクリプトでこれをインポートして使う
"""
import glob
import re
import os

def get_latest_session():
    """最新のセッションIDを返す"""
    files = glob.glob("session_*_keys.json")
    if not files:
        raise FileNotFoundError("セッションファイルが見つかりません")
    latest = max(files, key=lambda f: int(re.search(r"session_(\d+)_keys", f).group(1)))
    session_id = re.search(r"session_(\d+)_keys", latest).group(1)
    return session_id

def get_session_files(session_id=None):
    """セッションIDに対応するファイルパスを返す"""
    if session_id is None:
        session_id = get_latest_session()
    return {
        "session_id": session_id,
        "keys":         f"session_{session_id}_keys.json",
        "audio":        f"session_{session_id}_audio.wav",
        "video":        f"session_{session_id}_video.avi",
        "pose":         f"session_{session_id}_pose.json",
        "onset":        f"session_{session_id}_onset_matched.json",
        "integrated":   f"session_{session_id}_integrated.json",
        "keyboard_grid": f"session_{session_id}_keyboard_grid.json",
        "meta":          f"session_{session_id}_meta.json",
    }

if __name__ == "__main__":
    sid = get_latest_session()
    files = get_session_files(sid)
    print(f"最新セッション: {sid}")
    for k, v in files.items():
        exists = "✓" if os.path.exists(v) else "✗"
        print(f"  {exists} {k}: {v}")