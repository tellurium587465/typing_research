import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import urllib.request

# モデルファイルのダウンロード（初回のみ）
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("モデルをダウンロード中...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("ダウンロード完了")

# 指先ランドマーク番号
FINGERTIPS = {
    "thumb": 4, "index": 8,
    "middle": 12, "ring": 16, "pinky": 20
}

# 手の骨格接続定義（21点）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # 親指
    (0,5),(5,6),(6,7),(7,8),        # 人差し指
    (0,9),(9,10),(10,11),(11,12),   # 中指
    (0,13),(13,14),(14,15),(15,16), # 薬指
    (0,17),(17,18),(18,19),(19,20), # 小指
    (5,9),(9,13),(13,17)            # 手のひら
]

def draw_landmarks(frame, hand_lms, h, w, label):
    """ランドマークと骨格線を描画"""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

    # 骨格線
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)

    # 全関節に小さい丸
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

    # 指先だけ大きい丸＋ラベル
    for name, idx in FINGERTIPS.items():
        cx, cy = pts[idx]
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(frame, f"{label[0]}-{name}",
                    (cx, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)

# HandLandmarker設定
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
frame_idx = 0
start_ts = None

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # タイムスタンプを単調増加で管理
        import time
        if start_ts is None:
            start_ts = time.time()
        ts_ms = int((time.time() - start_ts) * 1000) + frame_idx
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].display_name
                draw_landmarks(frame, hand_lms, h, w, label)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()