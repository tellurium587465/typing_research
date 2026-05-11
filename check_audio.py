import librosa
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILE = "session_1778506938_audio.wav"

y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
print(f"長さ: {len(y)/sr:.2f}秒")
print(f"最大音量: {np.max(np.abs(y)):.4f}")
print(f"平均音量: {np.mean(np.abs(y)):.4f}")
print(f"無音割合: {np.mean(np.abs(y) < 0.001)*100:.1f}%")

# 波形を表示
plt.figure(figsize=(14, 3))
plt.plot(np.linspace(0, len(y)/sr, len(y)), y, linewidth=0.3)
plt.title("波形確認")
plt.xlabel("時間（秒）")
plt.savefig("check_audio.png", dpi=150)
plt.show()
print("保存: check_audio.png")