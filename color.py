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
    hsv =cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
    lower_red1 = (0, 120, 70)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 120, 70)
    upper_red2 = (179, 255, 255)
    red1=cv2.inRange(hsv,lower_red1,upper_red1)
    red2=cv2.inRange(hsv,lower_red2,upper_red2)
    red=red1+red2
    cv2.imshow("Red",red)
    #contours, hierarchy=cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 



    if cv2.waitKey(1)==ord('a'):
         break
cap.release()
cv2.destroyAllWindows()