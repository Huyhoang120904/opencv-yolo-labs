import cv2 
from ultralytics import YOLO 

model  = YOLO("yolov8n.pt")

image = cv2.imread("image3.jpg")

result = model(image)

annotated_image = result[0].plot()

cv2.imshow("Predicted image",annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

