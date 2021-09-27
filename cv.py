import cv2
#import numpy as np

cap = cv2.VideoCapture(0)
object_detector = cv2.createBackgroundSubtractorMOG2()
cascade = cv2.CascadeClassifier("cars2.xml")

try: 
    while True:
        ret, frame= cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #fc = face_cascade.load("")
        obj = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
        mask = object_detector.apply(frame)
        #contour, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for (x,y,w,h) in obj:
            print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            img_item = "my-image.png"
            cv2.imwrite(img_item,roi_gray)

            color = (255,0,0)
            stroke = 2
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame,(x, y), (end_cord_x,end_cord_y), color, stroke)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print("done")
