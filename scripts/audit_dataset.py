"""Check the annotation quality of a YOLO-format dataset before trusting any metric on it.

    python scripts/audit_dataset.py --data datasets/african-wildlife --out reports/dataset_audit.json

Checks, per split: images without a label file, empty label files, malformed lines,
coordinates outside [0, 1], zero-size or tiny boxes, near-duplicate boxes, class balance,
boxes per image and box-size distribution. Also checks for the same image appearing in
two splits (by file hash), which would inflate test scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TINY_AREA = 0.001          # box smaller than 0.1% of the image
SIZE_BUCKETS = {"small (<1% of image)": 0.01, "medium (1-10%)": 0.10, "large (>10%)": 1.01}


def iou(a, b) -> float:
    """IoU of two boxes in (xc, yc, w, h) normalised format."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dhash(path: Path, size: int = 16) -> int:
    """Perceptual 'difference hash': resized/re-saved copies of the same photo get almost the
    same hash, unlike an exact file hash."""
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def size_bucket(area: float) -> str:
    for name, upper in SIZE_BUCKETS.items():
        if area < upper:
            return name
    return list(SIZE_BUCKETS)[-1]


def audit_split(root: Path, split: str) -> tuple[dict, dict]:
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    stats = Counter()
    per_class = Counter()
    size_by_class = defaultdict(Counter)
    boxes_per_image = []
    areas = []
    hashes = {}
    perceptual = {}
    problems = defaultdict(list)

    for img in images:
        hashes[hashlib.md5(img.read_bytes()).hexdigest()] = img.name
        perceptual[img.name] = dhash(img)
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            problems["missing_label_file"].append(img.name)
            continue
        lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            problems["empty_label_file"].append(img.name)
        boxes = []
        for parts in lines:
            if len(parts) != 5:
                problems["malformed_line"].append(img.name)
                continue
            c, *xywh = parts
            c, xywh = int(float(c)), [float(v) for v in xywh]
            if c not in range(len(CLASSES)):
                problems["unknown_class_id"].append(img.name)
                continue
            x, y, w, h = xywh
            if w <= 0 or h <= 0:
                problems["zero_size_box"].append(img.name)
                continue
            if min(x - w / 2, y - h / 2) < -0.01 or max(x + w / 2, y + h / 2) > 1.01:
                problems["box_outside_image"].append(img.name)
            if w * h < TINY_AREA:
                problems["tiny_box"].append(img.name)
            for c2, b2 in boxes:
                if c2 == c and iou(xywh, b2) > 0.9:
                    problems["near_duplicate_box"].append(img.name)
            boxes.append((c, xywh))
            per_class[CLASSES[c]] += 1
            areas.append(w * h)
            size_by_class[CLASSES[c]][size_bucket(w * h)] += 1
        stats["boxes"] += len(boxes)
        boxes_per_image.append(len(boxes))

    bpi = np.array(boxes_per_image) if boxes_per_image else np.zeros(1)
    a = np.array(areas) if areas else np.zeros(1)
    report = {
        "images": len(images),
        "boxes": int(stats["boxes"]),
        "boxes_per_class": dict(per_class),
        "boxes_per_image": {"mean": round(float(bpi.mean()), 2), "median": float(np.median(bpi)),
                            "max": int(bpi.max())},
        "images_with_5_plus_boxes": int((bpi >= 5).sum()),
        "box_area_share_of_image": {"median": round(float(np.median(a)), 4),
                                    "p10": round(float(np.percentile(a, 10)), 4),
                                    "p90": round(float(np.percentile(a, 90)), 4)},
        "size_buckets_by_class": {k: dict(v) for k, v in size_by_class.items()},
        "problems": {k: {"count": len(v), "examples": sorted(set(v))[:5]}
                     for k, v in problems.items()},
    }
    return report, (hashes, perceptual)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with images/ and labels/")
    ap.add_argument("--out", default="reports/dataset_audit.json")
    ap.add_argument("--near-bits", type=int, default=10,
                    help="max differing hash bits to call two images near-identical")
    args = ap.parse_args(argv)
    root = Path(args.data)

    result, all_hashes, all_phash = {}, {}, {}
    for split in ("train", "val", "test"):
        if (root / "images" / split).exists():
            result[split], (all_hashes[split], all_phash[split]) = audit_split(root, split)

    # Same image in two splits = the model is evaluated on pictures it trained on.
    exact, near, leaked = {}, {}, defaultdict(set)
    splits = list(all_hashes)
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            common = set(all_hashes[s1]) & set(all_hashes[s2])
            exact[f"{s1}-{s2}"] = len(common)
            h1 = np.array(list(all_phash[s1].values()), dtype=object)
            names1 = list(all_phash[s1])
            count = 0
            for name2, h2 in all_phash[s2].items():
                dists = [bin(int(h) ^ h2).count("1") for h in h1]
                if min(dists) <= args.near_bits:
                    count += 1
                    leaked[s2].add(name2)
            near[f"{s1}-{s2}"] = count
    result["identical_images_across_splits"] = exact
    result[f"near_identical_images_across_splits (dhash <= {args.near_bits} of 256 bits)"] = near
    # Files to drop when computing a leakage-free score on val/test
    result["leaked_files"] = {k: sorted(v) for k, v in leaked.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "leaked_files"}
    summary["leaked_file_counts"] = {k: len(v) for k, v in result["leaked_files"].items()}
    print(json.dumps({k: v for k, v in summary.items() if k not in ("train", "val", "test")},
                     indent=2))
    return result


if __name__ == "__main__":
    main()
