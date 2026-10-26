"""YOLOv8 inference with ONNX Runtime only - no PyTorch, no Ultralytics at run time.

Pre- and post-processing are written by hand (same idea as the YOLOv3-tiny OpenCV notebook):
letterbox resize -> ONNX model -> decode the (1, 4 + classes, 8400) output -> confidence filter
-> per-class Non-Max Suppression -> map boxes back to the original image.

Why: the PyTorch + Ultralytics stack is several GB; onnxruntime + numpy + Pillow is about
100 MB, so the API container is small and starts fast. `scripts/export_benchmark.py` checks
that this detector gives the same boxes as Ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = ["buffalo", "elephant", "rhino", "zebra"]


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original-image pixels


def letterbox(img: Image.Image, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to size x size with grey (114), like Ultralytics."""
    w, h = img.size
    gain = min(size / w, size / h)
    nw, nh = round(w * gain), round(h * gain)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, gain, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS: keep the best box, drop boxes overlapping it by more than iou_thr, repeat."""
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class OnnxDetector:
    def __init__(self, weights: str | Path, imgsz: int = 640, threads: int | None = None):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(weights), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz

    def __call__(self, img: Image.Image, conf: float = 0.25, iou: float = 0.7,
                 max_det: int = 300) -> list[Detection]:
        img = img.convert("RGB")
        x, gain, (px, py) = letterbox(img, self.imgsz)
        out = self.session.run(None, {self.input_name: x})[0][0].T     # (8400, 4 + classes)
        scores_all = out[:, 4:]
        cls_ids = scores_all.argmax(1)
        scores = scores_all[np.arange(len(out)), cls_ids]
        mask = scores >= conf
        if not mask.any():
            return []
        xywh, scores, cls_ids = out[mask, :4], scores[mask], cls_ids[mask]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        # class-aware NMS: offset each class so boxes of different classes never suppress each other
        offset = cls_ids[:, None] * 4096.0
        keep = nms(boxes + offset, scores, iou)[:max_det]
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        # undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / gain
        w, h = img.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [Detection(CLASSES[c], round(float(s), 4), tuple(round(float(v), 1) for v in b))
                for b, s, c in zip(boxes, scores, cls_ids)]


def count(dets: list[Detection]) -> dict[str, int]:
    out = {c: 0 for c in CLASSES}
    for d in dets:
        out[d.cls] += 1
    return out
