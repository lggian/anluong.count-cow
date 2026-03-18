from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data\\input\\videodemo.mp4")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 19:   # cow
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,"Cow",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,255,0),2)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()