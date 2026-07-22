import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
while(1):
    ret,Frame=cap.read()
    if not ret:
         break
    cv2.imshow("original",Frame)
    if cv2.waitKey(1)==ord('q'):
         break
cap.release()
cv2.destroyAllWindows()

 