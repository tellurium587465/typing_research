import json

with open("session_1778410381_onset_matched.json") as f:
    onset = json.load(f)

with open("session_1778410381_pose.json") as f:
    pose = json.load(f)

first_key_ts = 12672

print("onset補正後 vs 最近傍poseフレーム（先頭10件）")
print("-" * 60)
for o in onset[:10]:
    corrected = o["onset_ms"] - first_key_ts
    closest = min(pose, key=lambda f: abs(f["timestamp_ms"] - corrected))
    diff = abs(closest["timestamp_ms"] - corrected)
    hands = len(closest["hands"])
    print(f"onset={o['onset_ms']:6d}ms → 補正={corrected:6d}ms → pose={closest['timestamp_ms']:6d}ms 差={diff:4d}ms 手={hands}本 key={o['key']}")