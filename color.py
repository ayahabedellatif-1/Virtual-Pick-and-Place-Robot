import cv2
import numpy as np 
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
while(1):
    ret,Frame=cap.read()
    if not ret:
         break
    
    result=model(Frame)
    detecton=result[0].plot()
    cv2.imshow("YOLO Detection", detecton)
    results = model.track(Frame, persist=True)
    for box in results[0].boxes:
         
         x1, y1, x2, y2 = box.xyxy[0]
         cx = int((x1 + x2) / 2)
         cy = int((y1 + y2) / 2)
         center=cv2.circle(detecton, (cx, cy), 3, (0, 0, 0), -1)
         cv2.imshow("centroid",center)
         if box.id is not None:
            track_id = int(box.id[0])
            print(f"ID: {track_id}, Centroid: ({cx}, {cy})")

    
    hsv =cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 120, 70)
    upper_red2 = (179, 255, 255)
    lower_green = (40, 70, 70)
    upper_green = (80, 255, 255)
    lower_blue = (100, 150, 70)
    upper_blue = (140, 255, 255)
    red1=cv2.inRange(hsv,lower_red1,upper_red1)
    red2=cv2.inRange(hsv,lower_red2,upper_red2)
    red=red1+red2
    green=cv2.inRange(hsv,lower_green,upper_green)
    blue=cv2.inRange(hsv,lower_blue,upper_blue)
    cv2.imshow("Red",red)
    cv2.imshow("Green",green)
    cv2.imshow("Blue",blue)


    #contours, hierarchy=cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)git 

    if cv2.waitKey(1)==ord('a'):
         break
cap.release()
cv2.destroyAllWindows()