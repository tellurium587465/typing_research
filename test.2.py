import cv2
import time

cap = cv2.VideoCapture(0)
time.sleep(1)
print("取得開始")
for i in range(100):
    ret, frame = cap.read()
    print(i, ret)
    if ret:
        cv2.imshow("test", frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()