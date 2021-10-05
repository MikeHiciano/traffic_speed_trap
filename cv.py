import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=45)
#cascade = cv2.CascadeClassifier("cars.xml")


while True:
    ret, frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width, _ = frame.shape
    #roi = frame[340: 600,500: 700]       
    #obj = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #for (x,y,w,h) in obj:
    for cnt in contours:
        # print(x,y,w,h)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        # img_item = "my-image.png"
        # cv2.imwrite(img_item,roi_gray)

        area = cv2.contourArea(cnt)
        if area > 100:
            color = (255,0,0)
            stroke = 2
            # end_cord_x = x+w
            # end_cord_y = y+h
            # cv2.rectangle(frame,(x, y), (end_cord_x,end_cord_y), color, stroke)
            #cv2.drawContours(frame,[cnt], -1, color, stroke)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()