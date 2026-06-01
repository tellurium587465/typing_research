import json
import time
import threading
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from pynput import keyboard
import numpy as np
import cv2

# 設定
SAMPLE_RATE = 44100
SESSION_ID = f"session_{int(time.time())}"
FPS = 30

# -----------------------------------------------
# マイクデバイス自動選択
# -----------------------------------------------
def select_best_mic(test_duration=0.5, target_sr=SAMPLE_RATE):
    """入力デバイスを短時間テストし、最も音量が大きいものを返す"""
    devices = sd.query_devices()
    candidates = [
        (i, d) for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
        and "mapper" not in d["name"].lower()   # Windows サウンドマッパーは除外
        and "primary" not in d["name"].lower()  # プライマリドライバも除外
    ]
    print("=== マイクデバイス検索 ===")
    best_idx, best_rms = None, -1.0
    for idx, dev in candidates:
        try:
            sr = int(dev["default_samplerate"])
            frames_n = int(sr * test_duration)
            rec = sd.rec(frames_n, samplerate=sr, channels=1,
                         device=idx, dtype="float32")
            sd.wait()
            rms = float(np.sqrt(np.mean(rec ** 2)))
            status = f"RMS={rms:.6f}"
            print(f"  [{idx:2d}] {dev['name'][:40]:40s}  {status}")
            if rms > best_rms:
                best_rms = rms
                best_idx = idx
        except Exception as e:
            print(f"  [{idx:2d}] {dev['name'][:40]:40s}  エラー: {e}")
    if best_idx is None:
        print("利用可能なマイクが見つかりません。デフォルトを使用します。")
        return None
    chosen = devices[best_idx]["name"]
    print(f"\n選択: [{best_idx}] {chosen}  (RMS={best_rms:.6f})")
    return best_idx

AUDIO_DEVICE = select_best_mic()

# 共有データ
keylog = []
audio_frames = []
recording = [True]
prev_ts = [0]
start_time = [None]

# -----------------------------------------------
# 録音スレッド
# -----------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if recording[0]:
        audio_frames.append(indata.copy())

def start_recording():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        device=AUDIO_DEVICE, callback=audio_callback):
        while recording[0]:
            time.sleep(0.1)

# -----------------------------------------------
# キーロガースレッド
# -----------------------------------------------
def on_press(key):
    if start_time[0] is None:
        return
    ts = int((time.time() - start_time[0]) * 1000)
    interval = ts - prev_ts[0]
    prev_ts[0] = ts

    try:
        k = key.char
    except AttributeError:
        k = str(key)

    entry = {
        "session_id": SESSION_ID,
        "timestamp_ms": ts,
        "key": k,
        "interval_ms": interval,
        "is_backspace": k == "Key.backspace",
        "is_error": False
    }
    keylog.append(entry)
    print(f"{ts:8d}ms  interval:{interval:5d}ms  {k}")

def on_release(key):
    if key == keyboard.Key.esc:
        recording[0] = False
        return False

def start_keylogger():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def process_frame(frame):
    """回転・トリミング処理"""
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    y1, y2 = h // 3, h * 2 // 3
    return frame[y1:y2, :]  # 縦方向中心1/3のみ

# -----------------------------------------------
# メイン
# -----------------------------------------------
print(f"セッションID: {SESSION_ID}")
print("起動中...")

# カメラを先に初期化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# カメラが安定するまで数フレーム読み捨て
for _ in range(10):
    cap.read()

# 実際のフレームを1枚取得してprocess_frame後のサイズを確認
ret, test_frame = cap.read()
test_frame = process_frame(test_frame)
out_h, out_w = test_frame.shape[:2]
print(f"カメラ解像度: {int(cap.get(3))}x{int(cap.get(4))} → 処理後: {out_w}x{out_h}")

# VideoWriterを処理後のサイズで初期化
video_file = f"{SESSION_ID}_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_file, fourcc, FPS, (out_w, out_h))

if not out.isOpened():
    print("VideoWriter初期化失敗")
else:
    print("VideoWriter初期化成功")

# カメラウィンドウを先に表示
cv2.imshow("Recording...", test_frame)
cv2.waitKey(500)

# 録音・キーロガー起動
rec_thread = threading.Thread(target=start_recording, daemon=True)
key_thread = threading.Thread(target=start_keylogger, daemon=True)
rec_thread.start()
key_thread.start()

start_time[0] = time.time()
print("録音＋カメラ＋キーロガー開始（ESCで終了）")
print("-" * 40)

# カメラメインループ（フレームカウントと実時間を記録）
frame_count = 0
video_start_time = time.time()

while recording[0]:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break
    frame = process_frame(frame)
    out.write(frame)
    frame_count += 1
    cv2.imshow("Recording...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKeyを1msに変更（処理遅延を減らす）
        recording[0] = False
        break

video_end_time = time.time()
actual_duration = video_end_time - video_start_time
actual_fps_real = frame_count / actual_duration if actual_duration > 0 else FPS

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"実際のキャプチャ: {frame_count}フレーム / {actual_duration:.1f}秒 = {actual_fps_real:.2f}fps")

key_thread.join(timeout=2)
rec_thread.join(timeout=2)

# ミスフラグ付与
for i, entry in enumerate(keylog):
    if entry["is_backspace"] and i > 0:
        keylog[i - 1]["is_error"] = True

# キーログ保存
keylog_file = f"{SESSION_ID}_keys.json"
with open(keylog_file, "w", encoding="utf-8") as f:
    json.dump(keylog, f, ensure_ascii=False, indent=2)

# 音声保存
audio_file = f"{SESSION_ID}_audio.wav"
if audio_frames:
    audio_data = np.concatenate(audio_frames, axis=0)
    wav_write(audio_file, SAMPLE_RATE, audio_data)

# セッションメタデータ保存（actual_fps を記録して後続スクリプトが使えるように）
meta_file = f"{SESSION_ID}_meta.json"
meta = {
    "session_id": SESSION_ID,
    "claimed_fps": FPS,
    "actual_fps": round(actual_fps_real, 4),
    "frame_count": frame_count,
    "video_duration_s": round(actual_duration, 3),
    "sample_rate": SAMPLE_RATE,
}
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("-" * 40)
print(f"キーログ保存: {keylog_file}（{len(keylog)}件）")
print(f"音声保存:     {audio_file}")
print(f"動画保存:     {video_file}")