"""
live_detect.py
--------------
Runs YOLO object detection (or tracking) on a webcam or a video file and
draws boxes, class names, confidence scores and live FPS on each frame.

Examples:
    python live_detect.py                                  # webcam, COCO-pretrained YOLOv8n
    python live_detect.py --weights best.pt                # webcam, your fine-tuned model
    python live_detect.py --source traffic.mp4 --track     # video file with object tracking
    python live_detect.py --source traffic.mp4 --save out.mp4

Press Q to quit.
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt", help="model weights (.pt)")
    p.add_argument("--source", default="0", help="webcam index (0) or path to a video file")
    p.add_argument("--conf", type=float, default=0.4, help="minimum confidence to show a box")
    p.add_argument("--track", action="store_true", help="track objects across frames (shows IDs)")
    p.add_argument("--save", default="", help="optional path to save the output video (.mp4)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    writer = None
    frame_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.perf_counter()
        if args.track:
            result = model.track(frame, conf=args.conf, persist=True, verbose=False)[0]
        else:
            result = model(frame, conf=args.conf, verbose=False)[0]
        frame_times.append(time.perf_counter() - start)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            if args.track and box.id is not None:
                label = f"#{int(box.id[0])} " + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        recent = frame_times[-30:]
        fps = len(recent) / sum(recent)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)

        cv2.imshow("YOLO detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        print(f"Processed {len(frame_times)} frames | average {avg * 1000:.1f} ms/frame | {1 / avg:.1f} FPS")


if __name__ == "__main__":
    main()
