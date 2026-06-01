"""
fix_pose_timestamps.py  ─ 既存 pose JSON のタイムスタンプを補正する

recorder.py は VideoWriter を claimed_fps(30) で初期化しているが、
実際のキャプチャ速度は actual_fps ≈ 21.5fps。
このズレにより pose の timestamp_ms が 1.4倍圧縮されている。

このスクリプトは音声ファイルから actual_fps を算出し、
pose JSON の timestamp_ms を補正して上書き保存する。
"""
import argparse, json, glob, os, re
import cv2
import scipy.io.wavfile as wav
from session_utils import get_session_files

parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None, help="省略時は全セッション")
parser.add_argument("--dry-run", action="store_true", help="変更しないで確認だけ")
args = parser.parse_args()

if args.session_id:
    session_ids = [args.session_id]
else:
    files = glob.glob("session_*_keys.json")
    session_ids = sorted(set(re.search(r"session_(\d+)_keys", f).group(1) for f in files))

for sid in session_ids:
    sf = get_session_files(sid)
    pose_file  = sf["pose"]
    video_file = sf["video"]
    audio_file = sf["audio"]

    if not os.path.exists(pose_file):
        print(f"session {sid}: pose ファイルなし、スキップ")
        continue
    if not os.path.exists(audio_file):
        print(f"session {sid}: audio ファイルなし、スキップ")
        continue

    # 音声の長さ
    sr, aud = wav.read(audio_file)
    audio_dur_s = len(aud) / sr

    # 動画の情報
    cap = cv2.VideoCapture(video_file)
    claimed_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames == 0 or claimed_fps == 0:
        print(f"session {sid}: 動画情報取得失敗、スキップ")
        continue

    actual_fps = total_frames / audio_dur_s
    correction = actual_fps / claimed_fps  # ≈ 0.717 (= 21.5/30)

    print(f"\nsession {sid}:")
    print(f"  claimed_fps={claimed_fps:.1f}  actual_fps={actual_fps:.2f}  "
          f"correction={correction:.4f}")
    print(f"  audio={audio_dur_s:.1f}s  video_claimed={total_frames/claimed_fps:.1f}s")

    if abs(correction - 1.0) < 0.01:
        print("  補正不要（ズレ < 1%）")
        continue

    # pose JSON を読んで補正
    with open(pose_file, encoding="utf-8") as f:
        pose_log = json.load(f)

    already_fixed = pose_log[0]["timestamp_ms"] > total_frames / claimed_fps * 1000 + 1000 if pose_log else False
    if already_fixed:
        print("  既に補正済みの可能性あり、スキップ")
        continue

    # timestamp_ms を補正
    for frame_data in pose_log:
        old_ts = frame_data["timestamp_ms"]
        frame_data["timestamp_ms"] = int(old_ts / correction)

    before_last = pose_log[-1]["timestamp_ms"] * correction if pose_log else 0
    after_last  = pose_log[-1]["timestamp_ms"] if pose_log else 0
    print(f"  補正前 最終timestamp: {int(before_last)}ms  補正後: {after_last}ms")
    print(f"  目標（audio長）: {int(audio_dur_s * 1000)}ms")

    if not args.dry_run:
        # バックアップ
        backup = pose_file.replace(".json", "_backup.json")
        if not os.path.exists(backup):
            with open(backup, "w", encoding="utf-8") as f:
                json.dump(
                    [{"frame_idx": d["frame_idx"], "timestamp_ms": int(d["timestamp_ms"] * correction)}
                     for d in pose_log],
                    f, ensure_ascii=False
                )
            print(f"  バックアップ: {backup}")
        with open(pose_file, "w", encoding="utf-8") as f:
            json.dump(pose_log, f, ensure_ascii=False, indent=2)
        print(f"  保存完了: {pose_file}")
    else:
        print("  [dry-run] ファイルは変更しません")
