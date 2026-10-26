"""Export the fine-tuned model to ONNX, check it matches PyTorch, and time both on CPU.

    python scripts/export_benchmark.py --weights runs/detect/runs/finetune/weights/best.pt \
        --images datasets/african-wildlife/images/test

1. export best.pt -> models/best.onnx (opset 12, 640x640, static shape)
2. pipeline check: Ultralytics running the ONNX file vs the hand-written ONNX detector on the
   same images (same model, so any difference comes from my pre/post-processing)
3. accuracy check: mAP of PyTorch vs ONNX on the leakage-free test set (needs scripts/evaluate.py
   to have written reports/_splits/data_test_clean.yaml)
4. latency on CPU, single image, end to end (pre-process + model + post-process), after warm-up
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detector_onnx import CLASSES, OnnxDetector  # noqa: E402
from scripts.evaluate import iou_xyxy  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs/finetune/weights/best.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/onnx_benchmark.json")
    args = ap.parse_args(argv)

    Path("models").mkdir(exist_ok=True)
    pt = YOLO(args.weights)
    onnx_path = Path(pt.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False))
    shutil.copy(onnx_path, "models/best.onnx")
    onnx = OnnxDetector("models/best.onnx")

    ux_onnx = YOLO("models/best.onnx", task="detect")
    images = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    matched = total = count_match = 0
    pm = ptotal = pcount = 0
    t_pt, t_onnx = [], []
    for i, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        t0 = time.perf_counter()
        r = pt.predict(str(p), conf=args.conf, imgsz=640, device="cpu", verbose=False)[0]
        t1 = time.perf_counter()
        dets = onnx(img, conf=args.conf)
        t2 = time.perf_counter()
        if i >= 5:                      # first 5 images = warm-up
            t_pt.append((t1 - t0) * 1000); t_onnx.append((t2 - t1) * 1000)
        pb, pc = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int)
        ob = np.array([d.box for d in dets]).reshape(-1, 4)
        oc = np.array([CLASSES.index(d.cls) for d in dets], dtype=int)
        ious = iou_xyxy(pb, ob)
        for k in range(len(pb)):
            total += 1
            if len(ob) and ((ious[k] >= 0.9) & (oc == pc[k])).any():
                matched += 1
        count_match += int(sorted(pc.tolist()) == sorted(oc.tolist()))
        r2 = ux_onnx.predict(str(p), conf=args.conf, imgsz=640, device="cpu", verbose=False)[0]
        ub, uc = r2.boxes.xyxy.numpy(), r2.boxes.cls.numpy().astype(int)
        ious2 = iou_xyxy(ub, ob)
        for k in range(len(ub)):
            ptotal += 1
            if len(ob) and ((ious2[k] >= 0.9) & (oc == uc[k])).any():
                pm += 1
        pcount += int(sorted(uc.tolist()) == sorted(oc.tolist()))

    clean_yaml = Path("reports/_splits/data_test_clean.yaml")
    accuracy = {}
    if clean_yaml.exists():
        for name, w in (("pytorch", args.weights), ("onnx", "models/best.onnx")):
            m = YOLO(w, task="detect").val(data=str(clean_yaml), split="test", imgsz=640,
                                           device="cpu", plots=False, verbose=False)
            accuracy[name] = {"mAP50": round(float(m.box.map50), 3),
                              "mAP50_95": round(float(m.box.map), 3)}

    result = {
        "onnx_file_mb": round(Path("models/best.onnx").stat().st_size / 1e6, 1),
        "pt_file_mb": round(Path(args.weights).stat().st_size / 1e6, 1),
        "images": len(images), "conf": args.conf,
        "handwritten_vs_ultralytics_same_onnx": {
            "boxes_matched_iou90": f"{pm}/{ptotal}",
            "images_with_identical_counts": f"{pcount}/{len(images)}"},
        "handwritten_onnx_vs_pytorch_model": {
            "boxes_matched_iou90": f"{matched}/{total}",
            "images_with_identical_counts": f"{count_match}/{len(images)}",
            "why_not_identical": "PyTorch predict letterboxes to a minimal rectangle; the static "
                                 "ONNX model always takes a 640x640 square"},
        "test_clean_map": accuracy,
        "cpu_latency_ms_median": {"pytorch_ultralytics": round(statistics.median(t_pt), 1),
                                  "onnxruntime_handwritten_postprocess": round(statistics.median(t_onnx), 1)},
        "cpu": platform.processor() or platform.machine(),
        "note": "Measured in a 2-vCPU cloud sandbox; numbers on another machine will differ.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
