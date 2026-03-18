import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load model
model = YOLO("yolo12x.pt")

# Video input
cap = cv2.VideoCapture("data/input/vd_stable.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("data/output/vd_stable_t.mp4", fourcc, fps, (width, height))

# ByteTrack
tracker = sv.ByteTrack(
    track_activation_threshold=0.4,
    lost_track_buffer=150,
    minimum_matching_threshold=0.9
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect
    results = model(frame, conf=0.5)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 19]
    detections = detections[detections.confidence > 0.45]
    detections = detections[(detections.area > 5000)]

    # update tracker
    detections = tracker.update_with_detections(detections)

    seen_ids = set()
    for track_id in detections.tracker_id:
        seen_ids.add(track_id)

    cow_count = len(seen_ids)   

    labels = [
        f"cow ID:{track_id}"
        for track_id
        in detections.tracker_id
    ]

    frame = box_annotator.annotate(
    scene=frame,
    detections=detections
    )

    frame = label_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )

    # counter
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
out.release()
cv2.destroyAllWindows()