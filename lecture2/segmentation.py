import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture('street.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, classes=[0], persist=True, verbose=False)

    for r in results: 
        annotated_frame = frame.copy()
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy()
            boxes = r.boxes.xyxy.numpy().astype(int)

            for i, mask in enumerate(masks):
                person_id = r.boxes.id.numpy()[i]
                mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
                color = (0, 0, 255)
                contour, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contour, -1, color, 2)
                cv2.putText(annotated_frame, f"ID: {person_id}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Segmentation and Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;  

cap.release()
cv2.destroyAllWindows()
