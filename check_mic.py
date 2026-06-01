import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write

# デバイス一覧
print("=== 利用可能なデバイス ===")
print(sd.query_devices())
print(f"\nデフォルト入力デバイス: {sd.query_devices(kind='input')['name']}")

# 3秒録音テスト
print("\n3秒間録音テスト中...キーボードを叩いてください")
SAMPLE_RATE = 44100
recording = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()

max_vol = np.max(np.abs(recording))
mean_vol = np.mean(np.abs(recording))
print(f"最大音量: {max_vol:.6f}")
print(f"平均音量: {mean_vol:.6f}")

if max_vol < 0.001:
    print("⚠ 音が録れていません。マイクを確認してください。")
elif max_vol < 0.01:
    print("⚠ 音量が極端に小さいです。マイクの音量設定を確認してください。")
else:
    print("✓ マイク正常")

wav_write("mic_test.wav", SAMPLE_RATE, recording)
print("保存: mic_test.wav")
