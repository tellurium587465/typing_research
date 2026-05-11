import cv2

VIDEO_FILE = "session_1778404533_video.avi"

cap = cv2.VideoCapture(VIDEO_FILE)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 複数フレームを保存して手のないフレームを探す
for frame_num in [10, 30, 50, 100, total-50, total-20]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"frame_{frame_num}.png", frame)
        print(f"保存: frame_{frame_num}.png")

cap.release()