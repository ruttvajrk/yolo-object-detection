"""Count animals in a folder of camera-trap / survey images and write a CSV.

    python count_animals.py --images path/to/folder --out counts.csv
    python count_animals.py --images path/to/folder --out counts.csv --save-boxes annotated/

One row per image: file name, count per species, total, and the most confident detection.
Uses the ONNX model (no PyTorch needed). The default confidence is the one tuned for counting
accuracy on the validation set (see reports/evaluation.json).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from api import DEFAULT_CONF, WEIGHTS
from detector_onnx import CLASSES, OnnxDetector, count

COLORS = {"buffalo": "orange", "elephant": "cyan", "rhino": "magenta", "zebra": "yellow"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="counts.csv")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--save-boxes", default="", help="folder for images with boxes drawn")
    args = ap.parse_args(argv)

    det = OnnxDetector(args.weights)
    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"No images found in {args.images}")
    if args.save_boxes:
        Path(args.save_boxes).mkdir(parents=True, exist_ok=True)

    totals = {c: 0 for c in CLASSES}
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", *CLASSES, "total", "max_conf"])
        for p in files:
            img = Image.open(p).convert("RGB")
            dets = det(img, conf=args.conf)
            c = count(dets)
            for k in CLASSES:
                totals[k] += c[k]
            w.writerow([p.name, *[c[k] for k in CLASSES], len(dets),
                        max((d.conf for d in dets), default="")])
            if args.save_boxes:
                draw = ImageDraw.Draw(img)
                for d in dets:
                    draw.rectangle(d.box, outline=COLORS[d.cls], width=3)
                    draw.text((d.box[0] + 4, d.box[1] + 4), f"{d.cls} {d.conf:.2f}", fill=COLORS[d.cls])
                img.save(Path(args.save_boxes) / p.name)
    print(f"{len(files)} images -> {args.out}. Totals: {totals}")


if __name__ == "__main__":
    main()
