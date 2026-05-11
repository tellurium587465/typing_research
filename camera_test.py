import cv2

cap = cv2.VideoCapture(0)  # 外付けカメラなら1に変える

while True:
    ret, frame = cap.read()
    if not ret:
        print("カメラが取得できません")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()