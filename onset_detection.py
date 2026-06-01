"""
onset_detection.py  ─ 音響onset検出とキーログのマッチング

修正点:
  - セッションID: session_utils.py で自動検出 (--session-id で上書き可)
  - タイムスタンプ: trim_start_ms を加算して録画基準に統一
  - パラメータ: wait=5, delta=0.07 でより高感度に
"""
import argparse
import json
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ヘッドレス実行でも保存できるようにする
import matplotlib.pyplot as plt

from session_utils import get_session_files

# -----------------------------------------------
# 引数解析
# -----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--session-id", default=None, help="セッションID（省略時は最新を自動検出）")
parser.add_argument("--no-plot", action="store_true", help="グラフ保存をスキップ")
args = parser.parse_args()

_sf = get_session_files(args.session_id)
SESSION_ID  = _sf["session_id"]
AUDIO_FILE  = _sf["audio"]
KEYLOG_FILE = _sf["keys"]
OUTPUT_FILE = _sf["onset"]

print(f"セッション: {SESSION_ID}")
print(f"音声: {AUDIO_FILE}")
print(f"キーログ: {KEYLOG_FILE}")

# -----------------------------------------------
# 音声読み込み＋正規化
# -----------------------------------------------
print("\n音声読み込み中...")
y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
duration = len(y) / sr
print(f"サンプリングレート: {sr}Hz  長さ: {duration:.2f}秒")

max_amp = np.max(np.abs(y))
target_amp = 0.3
y = y / (max_amp + 1e-9) * target_amp
print(f"音量正規化: {max_amp:.6f} → {target_amp}")

# -----------------------------------------------
# キーログ読み込み（録画基準タイムスタンプ）
# -----------------------------------------------
with open(KEYLOG_FILE, encoding="utf-8") as f:
    keylog = json.load(f)

normal_keys = [
    e for e in keylog
    if not e.get("is_backspace", False)
    and not e["key"].startswith("Key.")
]
key_times_original = [e["timestamp_ms"] for e in normal_keys]  # 録画基準

if not key_times_original:
    print("有効な打鍵データがありません")
    raise SystemExit(1)

# -----------------------------------------------
# 音声のトリミング（最初の打鍵の1秒前から）
# -----------------------------------------------
trim_start_ms = max(0, key_times_original[0] - 1000)
trim_start_sample = int(trim_start_ms / 1000 * sr)
y_trimmed = y[trim_start_sample:]

# 解析用のタイムスタンプはトリム後基準（onset検出と合わせる）
key_times_trimmed = [t - trim_start_ms for t in key_times_original]

print(f"\nトリミング: 先頭{trim_start_ms}ms をカット（残り{len(y_trimmed)/sr:.2f}秒）")
print(f"最初の打鍵（録画基準）: {key_times_original[0]}ms")
print(f"最初の打鍵（トリム基準）: {key_times_trimmed[0]}ms")

# -----------------------------------------------
# onset 検出（高感度設定）
# -----------------------------------------------
onset_frames = librosa.onset.onset_detect(
    y=y_trimmed, sr=sr,
    wait=5,       # 最低検出間隔: ~116ms (旧: 10 = ~232ms)
    pre_avg=5,
    post_avg=5,
    pre_max=5,
    post_max=5,
    delta=0.07    # 感度 (旧: 0.10)
)
onset_times_trimmed = librosa.frames_to_time(onset_frames, sr=sr)
onset_ms_trimmed    = (onset_times_trimmed * 1000).astype(int)

print(f"\nonset検出数: {len(onset_ms_trimmed)}件")
print(f"キーログ打鍵数: {len(key_times_trimmed)}件")
print(f"差分: {len(onset_ms_trimmed) - len(key_times_trimmed):+d}件")

# -----------------------------------------------
# ズレ分布の確認
# -----------------------------------------------
if len(onset_ms_trimmed) > 0 and len(key_times_trimmed) > 0:
    diffs = []
    for oms in onset_ms_trimmed:
        closest = min(key_times_trimmed, key=lambda k: abs(oms - k))
        diffs.append(oms - closest)
    diffs = np.array(diffs)
    print(f"\n--- ズレ分布（onset - 最近傍キー）---")
    print(f"  平均:   {np.mean(diffs):+.1f}ms")
    print(f"  中央値: {np.median(diffs):+.1f}ms")
    print(f"  標準偏差: {np.std(diffs):.1f}ms")
    print(f"  最小: {np.min(diffs):+.1f}ms  最大: {np.max(diffs):+.1f}ms")

# -----------------------------------------------
# マッチング（1対1・順序保存、TOLERANCE=80ms）
# -----------------------------------------------
TOLERANCE_MS = 80  # 旧: 100ms → より厳密に

matched_pairs = []
unmatched_onsets = []

oi, ki = 0, 0
onset_list = list(onset_ms_trimmed)
key_list   = list(key_times_trimmed)

while oi < len(onset_list) and ki < len(key_list):
    diff = onset_list[oi] - key_list[ki]
    if abs(diff) <= TOLERANCE_MS:
        matched_pairs.append((onset_list[oi], key_list[ki], diff))
        oi += 1
        ki += 1
    elif diff < 0:
        unmatched_onsets.append(onset_list[oi])
        oi += 1
    else:
        ki += 1

print(f"\nマッチング結果（許容誤差±{TOLERANCE_MS}ms）")
print(f"  マッチ成功: {len(matched_pairs)}件")
print(f"  未マッチonset: {len(unmatched_onsets)}件")
if len(onset_list) > 0:
    print(f"  マッチ率: {len(matched_pairs)/len(onset_list)*100:.1f}%  "
          f"/ キーログカバー率: {len(matched_pairs)/len(key_list)*100:.1f}%")

# -----------------------------------------------
# JSON保存 ─ タイムスタンプを録画基準に変換して保存
# -----------------------------------------------
matched_output = []
for ts_onset_trimmed, ts_key_trimmed, diff in matched_pairs:
    key_entry = next(
        (e for e in normal_keys
         if abs(e["timestamp_ms"] - (ts_key_trimmed + trim_start_ms)) < 50),
        None
    )
    if key_entry:
        matched_output.append({
            # 録画基準タイムスタンプ（integrate.py / pose_analysis.py と同一軸）
            "onset_ms":    int(ts_onset_trimmed) + int(trim_start_ms),
            "key_ms":      int(ts_key_trimmed)   + int(trim_start_ms),
            "diff_ms":     int(diff),
            "key":         key_entry["key"],
            "is_error":    key_entry.get("is_error", False),
            "interval_ms": key_entry["interval_ms"],
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(matched_output, f, ensure_ascii=False, indent=2)
print(f"\n保存: {OUTPUT_FILE}（{len(matched_output)}件）")

# -----------------------------------------------
# 可視化
# -----------------------------------------------
if not args.no_plot:
    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.waveshow(y_trimmed, sr=sr, ax=ax, alpha=0.6)
    for ot in onset_times_trimmed:
        ax.axvline(x=ot, color="red", linewidth=0.8, alpha=0.7)
    for kt in key_times_trimmed:
        ax.axvline(x=kt/1000, color="blue", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_title(f"波形 + onset（赤）vs キーログ（青破線）  session={SESSION_ID}")
    ax.set_xlabel("時間（秒）")
    plt.tight_layout()
    plt.savefig("onset_result.png", dpi=150)
    plt.close()
    print("グラフ保存: onset_result.png")
