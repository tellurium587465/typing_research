import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json

AUDIO_FILE = "session_1778410381_audio.wav"
KEYLOG_FILE = "session_1778410381_keys.json"

# -----------------------------------------------
# 音声読み込み＋無音区間のトリミング
# -----------------------------------------------
print("音声読み込み中...")
y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
duration = len(y) / sr
print(f"サンプリングレート: {sr}Hz  長さ: {duration:.2f}秒")
max_amp = np.max(np.abs(y))
print(f"録音最大音量: {max_amp:.6f}")

# 音量を常に正規化（大きすぎても小さすぎても補正）
target_amp = 0.3
y = y / (max_amp + 1e-9) * target_amp
print(f"音量正規化: {max_amp:.6f} → {target_amp}")

# キーログの最初の打鍵時刻から1秒前をトリミング開始点にする
with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)
normal_keys = [e for e in keylog if not e["is_backspace"]]
key_times = [e["timestamp_ms"] for e in normal_keys]

trim_start_ms = max(0, key_times[0] - 1000)  # 最初の打鍵の1秒前
trim_start_sample = int(trim_start_ms / 1000 * sr)
y = y[trim_start_sample:]
# タイムスタンプもずらす
key_times = [k - trim_start_ms for k in key_times]

print(f"トリミング: {trim_start_ms}ms以前をカット（残り{len(y)/sr:.2f}秒）")

# -----------------------------------------------
# onset detection
# -----------------------------------------------
onset_frames = librosa.onset.onset_detect(
    y=y, sr=sr,
    wait=10,
    pre_avg=7,
    post_avg=7,
    pre_max=7,
    post_max=7,
    delta=0.10
)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
onset_ms = (onset_times * 1000).astype(int)

print(f"\nonset検出数: {len(onset_ms)}件")
for i, ms in enumerate(onset_ms):
    print(f"  [{i+1:3d}] {ms:8d}ms")

# -----------------------------------------------
# キーログと比較
# -----------------------------------------------

print(f"\nキーログ打鍵数（BS除く）: {len(key_times)}件")
print(f"onset検出数:             {len(onset_ms)}件")
print(f"差分:                    {len(onset_ms) - len(key_times)}件")

# -----------------------------------------------
# ズレの分布を確認
# -----------------------------------------------
print("\n--- ズレ分布の確認 ---")
diffs = []
for oms in onset_ms:
    if len(key_times) == 0:
        break
    closest = min(key_times, key=lambda k: abs(oms - k))
    diff = oms - closest
    diffs.append(diff)

diffs = np.array(diffs)
print(f"平均ズレ:   {np.mean(diffs):+.1f}ms")
print(f"中央値ズレ: {np.median(diffs):+.1f}ms")
print(f"標準偏差:   {np.std(diffs):.1f}ms")
print(f"最小ズレ:   {np.min(diffs):+.1f}ms")
print(f"最大ズレ:   {np.max(diffs):+.1f}ms")

# -----------------------------------------------
# マッチング（1対1対応・順序保存）
# -----------------------------------------------
TOLERANCE_MS = 100
matched = 0
matched_pairs = []
unmatched_onsets = []

onset_list = list(onset_ms)
key_list = list(key_times)

oi, ki = 0, 0
while oi < len(onset_list) and ki < len(key_list):
    diff = onset_list[oi] - key_list[ki]
    if abs(diff) <= TOLERANCE_MS:
        matched += 1
        matched_pairs.append((onset_list[oi], key_list[ki], diff))
        oi += 1
        ki += 1
    elif diff < 0:
        unmatched_onsets.append(onset_list[oi])
        oi += 1
    else:
        ki += 1

print(f"\nマッチング結果（許容誤差±{TOLERANCE_MS}ms・1対1順序保存）")
print(f"  マッチ成功: {matched}件")
print(f"  未マッチ:   {len(unmatched_onsets)}件")
if len(onset_list) > 0:
    print(f"  マッチ率:   {matched/len(onset_list)*100:.1f}%")

# マッチしたペアをJSONに保存
print(f"\nmatched_pairs件数: {len(matched_pairs)}")
if matched_pairs:
    print("先頭1件:", matched_pairs[0])

matched_output = []
for pair in matched_pairs:
    ts_onset, ts_key, diff = pair  # タプル形式（onset_ms, key_ms, diff_ms）
    key_entry = next(
        (e for e in normal_keys if abs(e["timestamp_ms"] - ts_key) < 50),
        None
    )
    if key_entry:
        matched_output.append({
            "onset_ms":    int(ts_onset),
            "key_ms":      int(ts_key),
            "diff_ms":     int(diff),
            "key":         key_entry["key"],
            "is_error":    key_entry["is_error"],
            "interval_ms": key_entry["interval_ms"]
        })

onset_match_file = KEYLOG_FILE.replace("_keys.json", "_onset_matched.json")
with open(onset_match_file, "w", encoding="utf-8") as f:
    json.dump(matched_output, f, ensure_ascii=False, indent=2)
print(f"onsetマッチ保存: {onset_match_file}（{len(matched_output)}件）")

# -----------------------------------------------
# 可視化
# -----------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6)

# onset線（赤）
for ot in onset_times:
    ax.axvline(x=ot, color="red", linewidth=0.8, alpha=0.7)

# キーログ線（青）
for kt in key_times:
    ax.axvline(x=kt/1000, color="blue", linewidth=0.8, alpha=0.5, linestyle="--")

ax.set_title("波形 + onset検出（赤）vsキーログ（青破線）")
ax.set_xlabel("時間（秒）")
ax.legend(["波形", "onset（音響）", "キーログ"], loc="upper right")
plt.tight_layout()
plt.savefig("onset_result.png", dpi=150)
plt.show()
print("\nグラフ保存: onset_result.png")