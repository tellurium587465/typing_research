import cv2
import json

POSE_FILE  = "session_1778410381_pose.json"
VIDEO_FILE = "session_1778410381_video.avi"
GRID_FILE  = "keyboard_grid.json"

with open(POSE_FILE) as f:
    pose = json.load(f)

with open(GRID_FILE) as f:
    grid = json.load(f)

bbox = grid["bbox"]
x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

# キーボード領域内に指先があるフレームを探す
target_ts = None
pose_frame = None
for frame_data in pose:
    if not frame_data["hands"]:
        continue
    for hand in frame_data["hands"]:
        for fname, coords in hand["fingertips"].items():
            px, py = coords["px"], coords["py"]
            if x <= px <= x+w and y <= py <= y+h:
                target_ts = frame_data["timestamp_ms"]
                pose_frame = frame_data
                break
        if pose_frame:
            break
    if pose_frame:
        break

if pose_frame is None:
    print("キーボード領域内に指先が見つかりませんでした")
    exit()

cap = cv2.VideoCapture(VIDEO_FILE)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = int(target_ts / 1000 * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    print("フレーム取得失敗")
    exit()

# グリッドを描画
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
for r in grid["row_centers"]:
    cv2.line(frame, (x, y+r), (x+w, y+r), (0, 255, 255), 1)
for c in grid["col_centers"]:
    cv2.line(frame, (x+c, y), (x+c, y+h), (255, 100, 0), 1)

# 指先座標を描画
for hand in pose_frame["hands"]:
    for fname, coords in hand["fingertips"].items():
        px, py = coords["px"], coords["py"]
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(frame, fname, (px, py-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite("grid_check2.png", frame)
print(f"保存: grid_check2.png (timestamp={target_ts}ms)")