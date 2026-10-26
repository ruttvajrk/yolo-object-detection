"""Re-check the fine-tuned model's metrics and add the evaluation the notebook was missing.

    python scripts/evaluate.py --weights runs/detect/runs/finetune/weights/best.pt \
        --data-root datasets/african-wildlife --audit reports/dataset_audit.json

1. mAP on val and test exactly as Ultralytics reports it (reproduces yolo_metrics.json)
2. mAP on a *leakage-free* test set: the test images that are (near-)copies of training
   images (found by scripts/audit_dataset.py) are removed
3. error analysis on the clean test set: recall by class and box size, class confusions,
   and confident detections with no matching label (candidates for missing annotations)
4. the counting use case: pick the confidence threshold on the clean val set that gives the
   most accurate animal counts, then measure counting accuracy on the clean test set
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
SIZE_EDGES = [(0.01, "small (<1%)"), (0.10, "medium (1-10%)"), (1.01, "large (>10%)")]


def size_name(area: float) -> str:
    return next(name for edge, name in SIZE_EDGES if area < edge)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """YOLO labels -> class ids and normalised xyxy boxes."""
    rows = [list(map(float, ln.split())) for ln in label_path.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows)
    c, x, y, w, h = a.T
    return c.astype(int), np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


def predict_all(model: YOLO, images: list[Path], conf: float = 0.05) -> dict:
    """Predict once at a low confidence; any higher threshold is applied afterwards."""
    preds = {}
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        for img, r in zip(batch, model.predict([str(p) for p in batch], conf=conf, imgsz=640,
                                               device="cpu", verbose=False)):
            preds[img.name] = (r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy(),
                               r.boxes.xyxyn.numpy())
    return preds


def ultralytics_metrics(model: YOLO, yaml_path: Path, split: str) -> dict:
    m = model.val(data=str(yaml_path), split=split, imgsz=640, device="cpu", plots=False,
                  verbose=False)
    return {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
            "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
            "per_class_mAP50_95": {CLASSES[i]: round(float(v), 3) for i, v in enumerate(m.box.maps)}}


def count_errors(preds: dict, gts: dict, names: list[str], conf: float) -> dict:
    abs_err, exact, total_pred, total_true = [], 0, Counter(), Counter()
    per_class_err = defaultdict(list)
    for n in names:
        pc, ps, _ = preds[n]
        pred_counts = Counter(pc[ps >= conf].tolist())
        true_counts = Counter(gts[n][0].tolist())
        errs = [abs(pred_counts[k] - true_counts[k]) for k in range(len(CLASSES))]
        for k in range(len(CLASSES)):
            per_class_err[CLASSES[k]].append(errs[k])
        abs_err.append(sum(errs))
        exact += int(sum(errs) == 0)
        total_pred.update({CLASSES[k]: v for k, v in pred_counts.items()})
        total_true.update({CLASSES[k]: v for k, v in true_counts.items()})
    return {"conf": conf, "images": len(names),
            "mean_abs_count_error_per_image": round(float(np.mean(abs_err)), 3),
            "images_with_exact_counts": round(exact / len(names), 4),
            "per_class_mean_abs_error": {k: round(float(np.mean(v)), 3)
                                         for k, v in per_class_err.items()},
            "total_animals_true": dict(total_true), "total_animals_predicted": dict(total_pred)}


def error_analysis(preds: dict, gts: dict, names: list[str], conf: float) -> dict:
    found, total = Counter(), Counter()
    confusions = Counter()
    suspect = []
    for n in names:
        gc, gb = gts[n]
        pc, ps, pb = preds[n]
        keep = ps >= conf
        pc, ps, pb = pc[keep], ps[keep], pb[keep]
        ious = iou_xyxy(gb, pb)
        used = set()
        for g in range(len(gc)):
            key = (CLASSES[gc[g]], size_name(float((gb[g, 2] - gb[g, 0]) * (gb[g, 3] - gb[g, 1]))))
            total[key] += 1
            cands = [p for p in np.argsort(-ious[g]) if ious[g, p] >= 0.5 and p not in used]
            same = [p for p in cands if pc[p] == gc[g]]
            if same:
                used.add(same[0]); found[key] += 1
            elif cands:
                confusions[f"{CLASSES[gc[g]]} -> {CLASSES[pc[cands[0]]]}"] += 1
        for p in range(len(pc)):
            if ps[p] >= 0.7 and (len(gc) == 0 or ious[:, p].max() < 0.3):
                suspect.append({"image": n, "pred": CLASSES[pc[p]], "conf": round(float(ps[p]), 3)})
    recall = {f"{c} / {s}": {"recall": round(found[(c, s)] / total[(c, s)], 3), "boxes": total[(c, s)]}
              for (c, s) in sorted(total)}
    return {"conf": conf, "recall_iou50_by_class_and_size": recall,
            "class_confusions": dict(confusions),
            "confident_detections_without_label": suspect}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs/finetune/weights/best.pt")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--audit", default="reports/dataset_audit.json")
    ap.add_argument("--out", default="reports/evaluation.json")
    args = ap.parse_args(argv)

    root = Path(args.data_root).resolve()
    leaked = json.loads(Path(args.audit).read_text())["leaked_files"]
    work = Path("reports/_splits"); work.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for split in ("val", "test"):
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        clean = [p for p in imgs if p.name not in set(leaked.get(split, []))]
        f = work / f"{split}_clean.txt"
        f.write_text("\n".join(str(p) for p in clean) + "\n")
        split_files[split] = (imgs, clean, f)
    yaml_path = work / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"val_clean: {split_files['val'][2].resolve()}\ntest_clean: {split_files['test'][2].resolve()}\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))

    model = YOLO(args.weights)
    result = {"weights": args.weights,
              "leaked_images_removed": {s: len(split_files[s][0]) - len(split_files[s][1])
                                        for s in split_files},
              "map": {}}
    for split in ("val", "test"):
        result["map"][split] = ultralytics_metrics(model, yaml_path, split)
    # Ultralytics accepts a txt file of image paths as a split; labels are found next to them.
    for split in ("val", "test"):
        yp = work / f"data_{split}_clean.yaml"
        yp.write_text(yaml_path.read_text().replace(f"\n{split}: images/{split}\n",
                                                    f"\n{split}: {split_files[split][2].resolve()}\n"))
        result["map"][f"{split}_clean"] = ultralytics_metrics(model, yp, split)

    gts = {}
    for split in ("val", "test"):
        for p in split_files[split][1]:
            gts[p.name] = load_gt(root / "labels" / split / (p.stem + ".txt"))
    val_names = [p.name for p in split_files["val"][1]]
    test_names = [p.name for p in split_files["test"][1]]
    preds = {**predict_all(model, split_files["val"][1]), **predict_all(model, split_files["test"][1])}

    # Choose the counting threshold on val only
    grid = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]
    val_curve = [count_errors(preds, gts, val_names, t) for t in grid]
    best = min(val_curve, key=lambda r: (r["mean_abs_count_error_per_image"], -r["conf"]))
    conf = best["conf"]
    result["counting"] = {
        "threshold_chosen_on_val": conf,
        "val_curve": [{k: r[k] for k in ("conf", "mean_abs_count_error_per_image",
                                          "images_with_exact_counts")} for r in val_curve],
        "test_clean": count_errors(preds, gts, test_names, conf),
        "test_clean_at_default_0.25": count_errors(preds, gts, test_names, 0.25),
    }
    result["error_analysis_test_clean"] = error_analysis(preds, gts, test_names, conf)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "counting"}, indent=2)[:4000])
    print(json.dumps(result["counting"]["test_clean"], indent=2))
    return result


if __name__ == "__main__":
    main()
