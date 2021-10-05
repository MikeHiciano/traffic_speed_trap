import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=35)
cascade = cv2.CascadeClassifier("cars.xml")

try: 
    while True:
        ret, frame= cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obj = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
        mask = object_detector.apply(frame)
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
