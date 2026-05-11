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
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
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

print(f"カメラ解像度: {int(cap.get(3))}x{int(cap.get(4))}")

video_file = f"{SESSION_ID}_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(video_file, fourcc, FPS,
                      (int(cap.get(3)), int(cap.get(4))))

# カメラウィンドウを先に表示してから録音・キーロガーを起動
ret, frame = cap.read()
if ret:
    cv2.imshow("Recording...", frame)
    cv2.waitKey(500)  # ウィンドウ表示を安定させる

# 録音・キーロガー起動
rec_thread = threading.Thread(target=start_recording, daemon=True)
key_thread = threading.Thread(target=start_keylogger, daemon=True)
rec_thread.start()
key_thread.start()

start_time[0] = time.time()
print("録音＋カメラ＋キーロガー開始（ESCで終了）")
print("-" * 40)

# カメラメインループ
while recording[0]:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # 縦方向だけ中心1/3をトリミング
    h, w = frame.shape[:2]
    y1, y2 = h // 3, h * 2 // 3
    frame = frame[y1:y2, :]

    out.write(frame)
    cv2.imshow("Recording...", frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        recording[0] = False
        break

cap.release()
out.release()
cv2.destroyAllWindows()

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

print("-" * 40)
print(f"キーログ保存: {keylog_file}（{len(keylog)}件）")
print(f"音声保存:     {audio_file}")
print(f"動画保存:     {video_file}")