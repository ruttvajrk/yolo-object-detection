# Object Detection with YOLO

Three parts:

1. **`files/yolov3_tiny_opencv.ipynb`** – YOLOv3-tiny inference with OpenCV's DNN module, with the post-processing written by hand: 416×416 blob, parsing the raw output of both detection layers (507 + 2,028 candidate boxes × 85 values), converting normalized coordinates to pixels, confidence filtering and Non-Max Suppression.
2. **`yolov8_finetune.ipynb`** – fine-tuning a COCO-pretrained YOLOv8n on a new dataset, with evaluation (precision, recall, mAP) and speed measurement.
3. **`live_detect.py`** – live webcam / video detection and tracking with boxes, confidence scores and FPS.

## Results (fine-tuned YOLOv8n)

Fine-tuned from COCO-pretrained `yolov8n.pt` on the public Ultralytics **African Wildlife** dataset (4 classes: buffalo, elephant, rhino, zebra) for 30 epochs at 640 px on a Google Colab Tesla T4 GPU.

| Metric | Validation | Test (held out) |
|---|---|---|
| Precision | 0.935 | 0.966 |
| Recall | 0.920 | 0.927 |
| mAP@50 | 0.959 | 0.971 |
| mAP@50-95 | 0.800 | 0.826 |

Per-class mAP@50-95 (test): buffalo 0.856, elephant 0.789, rhino 0.908, zebra 0.751.

| Device | ms / image | FPS |
|---|---|---|
| Tesla T4 GPU (Colab) | 12.3 | 81.4 |
| CPU (Colab) | 117.7 | 8.5 |

Full metrics: `yolo_metrics.json`. Training curves, confusion matrix and sample predictions are in the `runs` folder.

## Setup

```bash
pip install -r requirements.txt
```

## Live detection

```bash
python live_detect.py                               # webcam, pretrained model
python live_detect.py --weights best.pt             # your fine-tuned model
python live_detect.py --source video.mp4 --track    # tracking on a video
```

Press **Q** to quit. Average FPS is printed at the end.

## Project structure

```
├── yolov8_finetune.ipynb          Training, evaluation, speed test
├── live_detect.py                 Webcam / video detection and tracking
├── files/
│   ├── yolov3_tiny_opencv.ipynb   OpenCV DNN inference + manual NMS
│   ├── yolov3-tiny.cfg, yolov3-tiny.weights, coco.names
│   └── images/                    test images
└── requirements.txt
```
