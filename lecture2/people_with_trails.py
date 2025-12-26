import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")
# cap = cv2.VideoCapture('walking_man.mp4')
cap = cv2.VideoCapture('street.mp4')

unique_ids = set()

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

id_map = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

    for box, oid in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2)//2, (y1+y2)//2

        appear[oid]+= 1

        if appear[oid] >= 5 and oid not in id_map:
            id_map[oid] = next_id
            next_id += 1

        if oid in id_map:
            sid = id_map[oid]
            trail[sid].append((cx, cy))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated_frame, f"ID: {sid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (255,0,0), -1) 
            for i in range(1, len(trail[sid])):
                cv2.line(annotated_frame, trail[sid][i-1], trail[sid][i], (255,0,0), 2)

    cv2.imshow("People with Trails", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()