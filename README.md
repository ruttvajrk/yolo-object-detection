# Wildlife Detection and Counting with YOLO

**Use case:** camera-trap and wildlife-survey teams spend hours counting animals by hand in photos. This project fine-tunes YOLOv8 to detect and **count** four species (buffalo, elephant, rhino, zebra), checks whether the dataset and the reported scores can be trusted, and serves the model as a small API that runs without PyTorch.

**Tools:** Python · Ultralytics YOLOv8 · PyTorch · OpenCV · ONNX / ONNX Runtime · FastAPI · Docker · pytest

## Results

| What | Result |
|---|---|
| Test mAP@50 / mAP@50-95 (227 images, as Ultralytics reports it) | 0.971 / 0.826 (reproduced exactly on CPU) |
| **Leakage-free test** mAP@50 / mAP@50-95 (206 images) | **0.968 / 0.819** |
| Duplicate images found across splits | 20 test and 8 val images are copies of training images |
| Counting accuracy, leakage-free test (threshold tuned on val) | 86.4% of images counted exactly; 0.165 counting errors per image |
| Same, at the default 0.25 threshold | 84.0% exact; 0.267 errors per image (tuning cut the error by 38%) |
| ONNX model, hand-written pre/post-processing | mAP@50 0.967; same counts as Ultralytics on 99 of 100 images |
| CPU latency per image (2-vCPU sandbox) | ONNX Runtime 59.5 ms vs PyTorch 74.5 ms |
| Serving environment size | ~206 MB (ONNX Runtime) vs >1.2 GB for PyTorch alone |

## Project parts

| File | What it does |
|---|---|
| `yolov8_finetune.ipynb`, `train.py` | Fine-tune COCO-pretrained YOLOv8n, 30 epochs, 640 px, seed 42 (Colab T4, about 9 min). `train.py` repeats the exact settings in `runs/.../args.yaml` |
| `scripts/audit_dataset.py` | Annotation and split checks → `reports/dataset_audit.json` |
| `scripts/evaluate.py` | mAP on full and leakage-free splits, error analysis, counting threshold → `reports/evaluation.json` |
| `scripts/export_benchmark.py` | ONNX export, correctness checks, CPU latency → `reports/onnx_benchmark.json` |
| `detector_onnx.py` | YOLOv8 inference with ONNX Runtime only: letterbox, output decoding, per-class NMS written by hand |
| `api.py` | FastAPI: `POST /detect` (upload image → counts per species + boxes), `GET /health` |
| `count_animals.py` | Folder of images → CSV of counts per image, optional images with boxes drawn |
| `live_detect.py` | Webcam / video detection and tracking with FPS |
| `files/yolov3_tiny_opencv.ipynb` | YOLOv3-tiny with OpenCV DNN and hand-written post-processing (2,535 candidate boxes, NMS) |

## 1. Can the dataset be trusted? (`scripts/audit_dataset.py`)

Ultralytics African Wildlife dataset: 1,052 / 225 / 227 images (train / val / test), 2,685 boxes.

- **Format checks passed:** no missing or empty label files, no malformed lines, no boxes outside the image, no zero-size or near-duplicate boxes.
- **Split leakage found:** 12 test images are byte-identical to training images, 8 more are near-identical (re-saved copies, found with a perceptual hash and checked by eye), and 1 matches a val image. In total 21 test images (9%) and 8 val images were removed to build leakage-free splits.
- **Effect:** small. Test mAP@50-95 drops from 0.826 to 0.819. The model is not just memorising, but the original number was slightly optimistic.

## 2. Error analysis (leakage-free test, `reports/evaluation.json`)

- **Box size matters most.** Recall at IoU 0.5 is 0.90 to 0.99 for large animals (>10% of the image) but 0.40 for medium buffalo and small elephants, and 0.69 for medium rhinos. These groups have only 5 to 13 boxes each, so the numbers are rough, but the pattern is consistent.
- **Zebra is the weakest class** (mAP@50-95 0.747): many zebras stand in overlapping groups, so box edges are harder to place precisely.
- **Class mix-ups are rare:** 2 rhinos were labelled as elephant or buffalo.
- **Confident detections with no label (5):** checked by eye. Two are real animals the annotators missed (an elephant and a zebra cut off at the image edge), so the labels have some gaps. Two are animals outside the 4 classes (a lioness detected as "rhino", an antelope as "elephant"): the model has no "other animal" class, which matters for real camera traps. The fifth is a box overlapping two labelled zebras.

## 3. Counting (the use case)

For counting, a false box and a missed animal both make the count wrong, so the confidence threshold is chosen to minimise count error on the **validation** set (best: 0.65), then measured once on the leakage-free test set: 86.4% of images exactly right, 0.165 counting errors per image, 321 animals counted vs 341 present (slight undercount, mostly zebras in groups).

```bash
python count_animals.py --images path/to/photos --out counts.csv --save-boxes annotated/
```

## 4. Deployment without PyTorch

`scripts/export_benchmark.py` exports `best.pt` to ONNX. `detector_onnx.py` then runs it with only ONNX Runtime, NumPy and Pillow, with the pre- and post-processing written by hand. Checks:

- Against Ultralytics running the same ONNX file: 158 of 160 boxes match (IoU ≥ 0.9), identical counts on 99 of 100 images. The small gap comes from image resizing (Pillow vs OpenCV).
- Against the PyTorch model: mAP@50-95 0.809 vs 0.819, because the static ONNX model always takes a 640×640 square while PyTorch pads to the smallest rectangle.

```bash
uvicorn api:app --port 8000                       # docs at http://localhost:8000/docs
curl -F "file=@tests/sample_zebra.jpg" localhost:8000/detect
docker build -t wildlife-api . && docker run -p 8000:8000 wildlife-api
pytest -q                                         # 4 API tests
```

## Reproduce

```bash
pip install -r requirements.txt
# dataset: https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip -> datasets/african-wildlife
python scripts/audit_dataset.py --data datasets/african-wildlife
python scripts/evaluate.py --data-root datasets/african-wildlife
python scripts/export_benchmark.py --images datasets/african-wildlife/images/test
```

## Limitations

- Training was not re-run for this version (no GPU available); all new numbers come from the existing `best.pt`.
- Only 4 species; other animals get forced into one of them. A real camera-trap model needs an "other / empty" class and night (infrared) images.
- Small and distant animals are under-represented (under 3% of training boxes) and are detected poorly.
- The Docker image was written and its requirements tested in a clean environment, but the image itself was not built (Docker Hub was not reachable from the build machine).
