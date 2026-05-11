import json
import numpy as np
import matplotlib.pyplot as plt

POSE_FILE  = "session_1778404533_pose.json"
KEYLOG_FILE = "session_1778404533_keys.json"

FINGERTIPS = ["thumb", "index", "middle", "ring", "pinky"]
VELOCITY_THRESHOLD = 0.08  # 閾値を大幅に上げる
SMOOTH_WINDOW = 5           # スムージングのウィンドウサイズ

# -----------------------------------------------
# pose_logの読み込み
# -----------------------------------------------
with open(POSE_FILE, encoding="utf-8") as f:
    pose_log = json.load(f)

with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)

normal_keys = [e for e in keylog if not e["is_backspace"]]
key_times   = [e["timestamp_ms"] for e in normal_keys]

# -----------------------------------------------
# 指ごとにy座標の時系列を作成
# -----------------------------------------------
# { "Left_index": [(ts_ms, y), ...], ... }
finger_series = {}

for frame in pose_log:
    ts = frame["timestamp_ms"]
    for hand in frame["hands"]:
        label = hand["label"]  # "Left" or "Right"
        for fname, coords in hand["fingertips"].items():
            key = f"{label}_{fname}"
            if key not in finger_series:
                finger_series[key] = []
            finger_series[key].append((ts, coords["y"]))

# -----------------------------------------------
# 速度計算＋ピーク検出（押下タイミング候補）
# -----------------------------------------------
press_events = []  # { ts_ms, finger, velocity }

for finger, series in finger_series.items():
    series.sort(key=lambda x: x[0])
    times = np.array([s[0] for s in series])
    ys    = np.array([s[1] for s in series])

    # スムージング（移動平均でノイズ除去）
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    ys_smooth = np.convolve(ys, kernel, mode="same")

    # フレーム間のy速度（正=下方向=押下）
    velocities = np.diff(ys_smooth) / (np.diff(times) / 1000 + 1e-9)

    # ピーク検出（速度が閾値を超えた点）
    for i in range(1, len(velocities) - 1):
        v = velocities[i]
        if (v > VELOCITY_THRESHOLD and
            v >= velocities[i-1] and
            v >= velocities[i+1]):
            press_events.append({
                "ts_ms":    int(times[i]),
                "finger":   finger,
                "velocity": round(float(v), 4)
            })

press_events.sort(key=lambda x: x["ts_ms"])
print(f"指速度ピーク検出数: {len(press_events)}件")
print(f"キーログ打鍵数:     {len(key_times)}件")

# -----------------------------------------------
# キーログとマッチング（1対1・順序保存）
# -----------------------------------------------
TOLERANCE_MS = 100
matched_pairs = []
pi, ki = 0, 0
used_press = set()

while pi < len(press_events) and ki < len(key_times):
    diff = press_events[pi]["ts_ms"] - key_times[ki]
    if abs(diff) <= TOLERANCE_MS:
        matched_pairs.append({
            "ts_ms":   press_events[pi]["ts_ms"],
            "key":     normal_keys[ki]["key"],
            "finger":  press_events[pi]["finger"],
            "diff_ms": diff
        })
        pi += 1
        ki += 1
    elif diff < 0:
        pi += 1
    else:
        ki += 1

print(f"\nマッチング結果（±{TOLERANCE_MS}ms）")
print(f"  成功: {len(matched_pairs)}件 / {len(key_times)}件")
print(f"  マッチ率: {len(matched_pairs)/len(key_times)*100:.1f}%")

print("\n--- マッチしたキー×指 サンプル（先頭20件）---")
for p in matched_pairs[:20]:
    print(f"  {p['ts_ms']:8d}ms  key={p['key']:4s}  finger={p['finger']}")

# -----------------------------------------------
# 可視化：指ごとのy速度
# -----------------------------------------------
fig, axes = plt.subplots(len(finger_series), 1,
                         figsize=(14, 2 * len(finger_series)), sharex=True)
if len(finger_series) == 1:
    axes = [axes]

for ax, (finger, series) in zip(axes, finger_series.items()):
    series.sort(key=lambda x: x[0])
    times = np.array([s[0] for s in series]) / 1000
    ys    = np.array([s[1] for s in series])
    ys_smooth = np.convolve(ys, np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode="same")
    vel   = np.diff(ys_smooth) / (np.diff(times) + 1e-9)
    ax.plot(times[1:], vel, linewidth=0.8)
    ax.axhline(VELOCITY_THRESHOLD, color="red", linewidth=0.5, linestyle="--")
    ax.set_ylabel(finger, fontsize=8)
    ax.set_ylim(-0.5, 0.5)

axes[-1].set_xlabel("時間（秒）")
plt.suptitle("指先y方向速度（赤破線=閾値）")
plt.tight_layout()
plt.savefig("finger_velocity.png", dpi=150)
plt.show()
print("\nグラフ保存: finger_velocity.png")