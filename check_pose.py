import json

with open("session_1778410381_pose.json") as f:
    pose = json.load(f)

# 手が検出されたフレームのサンプルを表示
for frame in pose:
    if frame["hands"]:
        print("timestamp:", frame["timestamp_ms"])
        for hand in frame["hands"]:
            print(" ", hand["label"])
            for fname, coords in hand["fingertips"].items():
                print(f"    {fname}: x={coords['x']:.3f} y={coords['y']:.3f} px={coords['px']} py={coords['py']}")
        break