import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo12x.pt")   # có thể đổi sang model bạn train

# Video input
cap = cv2.VideoCapture("data/input/vd_stable.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("data/output/vd_stable_v12.mp4", fourcc, fps, (width, height))

tracker = sv.ByteTrack(
    track_activation_threshold=0.3,
    lost_track_buffer=60,
    minimum_matching_threshold=0.85
)

# Vùng chuồng (polygon)
PEN = np.array([
    [100, 100],
    [1800, 100],
    [1800, 850],
    [100, 850]
], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect object
    results = model(frame, conf=0.4)[0]

    cow_count = 0

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # class 19 = cow trong COCO
        if cls != 19:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # check nếu center nằm trong chuồng
        inside = cv2.pointPolygonTest(PEN, (cx, cy), False)

        if inside >= 0:
            cow_count += 1

            # vẽ box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"cow {conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    # vẽ vùng chuồng
    # cv2.polylines(frame, [PEN], True, (0,255,255), 3)

    # vẽ counter
    cv2.rectangle(frame, (20,10), (400,90), (0,0,0), -1)

    cv2.putText(
        frame,
        f"COW IN PEN: {cow_count}",
        (30,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0,255,0),
        3
    )

    out.write(frame)
    cv2.imshow("Cow Counter AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()