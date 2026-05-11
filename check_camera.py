import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"カメラ {i}: OK（解像度 {int(cap.get(3))}x{int(cap.get(4))}）")
        cap.release()
    else:
        print(f"カメラ {i}: なし")