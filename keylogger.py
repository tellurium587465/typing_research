import json
import time
from pynput import keyboard

# ログを保存するリスト
keylog = []
start_time = time.time()

prev_ts = [0]  # リストで持つことで関数内から更新可能にする

def on_press(key):
    ts = int((time.time() - start_time) * 1000)
    interval = ts - prev_ts[0]
    prev_ts[0] = ts

    try:
        k = key.char
    except AttributeError:
        k = str(key)

    entry = {
        "timestamp_ms": ts,
        "key": k,
        "interval_ms": interval,
        "is_backspace": k == "Key.backspace"
    }
    keylog.append(entry)
    print(f"{ts:8d}ms  interval:{interval:5d}ms  {k}")

def on_release(key):
    # ESCで終了
    if key == keyboard.Key.esc:
        return False

print("キーロガー開始（ESCで終了）")
print("-" * 30)

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# ミスフラグを付与（バックスペース直前の1キーをis_error=Trueに）
for i, entry in enumerate(keylog):
    entry["is_error"] = False

for i, entry in enumerate(keylog):
    if entry["is_backspace"] and i > 0:
        keylog[i - 1]["is_error"] = True

# JSONに保存
filename = f"keylog_{int(time.time())}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(keylog, f, ensure_ascii=False, indent=2)

print("-" * 30)
print(f"保存完了：{filename}（{len(keylog)}件）")