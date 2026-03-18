from ultralytics import YOLO
import supervision as sv
import cv2

# load YOLO model
model = YOLO("yolo12x.pt")

cap = cv2.VideoCapture("data/input/vd_gate.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("data/output/vd_gate_v12.mp4", fourcc, fps, (width, height))

# tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.3,
    lost_track_buffer=60,
    minimum_matching_threshold=0.85
)

line_start = sv.Point(20, 940)
line_end   = sv.Point(1910, 1050)

# line_start = sv.Point(0, 980)
# line_end   = sv.Point(1920, 980)

line_counter = sv.LineZone(
    start=line_start,
    end=line_end,
    triggering_anchors=[sv.Position.BOTTOM_CENTER]
)

# annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

line_annotator = sv.LineZoneAnnotator(
    thickness=2,
    text_thickness=0,
    text_scale=0
)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    # chỉ giữ class cow (COCO id = 19)
    detections = detections[detections.class_id == 19]
    
    detections = detections[detections.confidence > 0.45]
    
    detections = detections.with_nms(threshold=0.5)
    # tracking
    detections = tracker.update_with_detections(detections)

    # labels
    labels = []
    for tracker_id in detections.tracker_id:
        if tracker_id is None:
            labels.append("cow")
        else:
            labels.append(f"cow #{tracker_id}")

    # ===== COUNT =====
    line_counter.trigger(detections)
    
    total = line_counter.out_count

    # draw box
    frame = box_annotator.annotate(frame, detections)

    # draw labels
    frame = label_annotator.annotate(frame, detections, labels)

    # draw line + counter
    frame = line_annotator.annotate(
        frame=frame,
        line_counter=line_counter
    )

    cv2.putText(
        frame,
        f"TOTAL COW: {total}",
        (30, 60),                 
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )
    
    out.write(frame)

    cv2.imshow("Cow Counter AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()