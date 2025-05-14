import cv2
import argparse
import numpy as np
import threading
import supervision as sv
from ultralytics import YOLO
from supervision import BoxAnnotator, Detections

# Define detection zone polygon (covers full 1280x720 frame)
ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1280, 720],
    [0, 720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Multi-Camera Detection (Webcam or CCTV)")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int,
        help="Set webcam resolution (width height)"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=str,  # Accept integers (e.g., "0") or RTSP/HTTP URLs
        default=["0"],
        help="List of camera indices or stream URLs (e.g., 0 rtsp://...)"
    )
    return parser.parse_args()

def setup_camera(source, width, height):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Video source {source} could not be opened.")

    # If source is a digit (e.g., "0", "1"), it's a webcam, so we set resolution
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap

def camera_thread(cap, model, camera_id):
    box_annotator = BoxAnnotator()
    zone = sv.PolygonZone(polygon=ZONE_POLYGON)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, color=sv.Color.RED, thickness=2, text_thickness=2, text_scale=0.8
    )
    class_names = model.model.names

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Camera {camera_id}: Failed to grab frame")
            break

        results = model(frame)[0]
        detections = Detections.from_ultralytics(results)

        detections.data["class_name"] = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        detections_in_zone = detections[zone.trigger(detections=detections)]
        frame = box_annotator.annotate(scene=frame.copy(), detections=detections_in_zone)

        for box, label in zip(detections_in_zone.xyxy, detections_in_zone.data["class_name"]):
            x1, y1, _, _ = map(int, box)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        frame = zone_annotator.annotate(scene=frame)
        object_count = len(detections_in_zone)
        cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(f"Camera {camera_id}", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyWindow(f"Camera {camera_id}")

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    camera_sources = args.cameras

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")
    threads = []

    for cam_id, source in enumerate(camera_sources):
        cap = setup_camera(source, frame_width, frame_height)
        thread = threading.Thread(
            target=camera_thread,
            args=(cap, model, cam_id),
            daemon=True
        )
        thread.start()
        threads.append(thread)

    try:
        while True:
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
