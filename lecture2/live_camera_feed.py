import cv2
from ultralytics import YOLO

model  = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = model(frame)
    annotated_frame = result[0].plot()
    cv2.imshow("Live Camera Feed", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()