import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import json
import time

VIDEO_FILE = "session_1778410381_video.avi"
OUTPUT_FILE = "session_1778410381_pose.json"
MODEL_PATH = "hand_landmarker.task"

# 指先ランドマーク番号と名前
FINGERTIPS = {
    4:  "thumb",
    8:  "index",
    12: "middle",
    16: "ring",
    20: "pinky"
}

# 骨格接続定義
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_lms, h, w, label):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (255, 255, 255), -1)
    for idx, name in FINGERTIPS.items():
        cx, cy = pts[idx]
        cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
        cv2.putText(frame, f"{label[0]}-{name}", (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

# HandLandmarker設定
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"動画: {VIDEO_FILE}")
print(f"FPS: {fps}  総フレーム数: {total}")

pose_log = []  # フレームごとの指先座標を保存
frame_idx = 0
start_ts = time.time()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        ts_ms = int(frame_idx / fps * 1000)  # 動画内のタイムスタンプ

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ms": ts_ms,
            "hands": []
        }

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].display_name

                fingertips = {}
                for idx, name in FINGERTIPS.items():
                    lm = hand_lms[idx]
                    fingertips[name] = {
                        "x": round(lm.x, 4),
                        "y": round(lm.y, 4),
                        "z": round(lm.z, 4),
                        "px": int(lm.x * w),
                        "py": int(lm.y * h)
                    }

                frame_data["hands"].append({
                    "label": label,
                    "fingertips": fingertips
                })

                draw_landmarks(frame, hand_lms, h, w, label)

        pose_log.append(frame_data)

        # 進捗表示
        if frame_idx % 30 == 0:
            progress = frame_idx / total * 100
            print(f"  進捗: {progress:.1f}% ({frame_idx}/{total}フレーム)")

        cv2.imshow("Pose Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# 保存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(pose_log, f, ensure_ascii=False, indent=2)

hands_detected = sum(1 for f in pose_log if f["hands"])
print(f"\n完了: {frame_idx}フレーム処理")
print(f"手を検出したフレーム: {hands_detected}/{frame_idx} ({hands_detected/frame_idx*100:.1f}%)")
print(f"保存: {OUTPUT_FILE}")