# What changed in this version (and why)

Read this before interviews: you should be able to explain every change and every number.

| Change | File | Why |
|---|---|---|
| Framed as a counting tool for wildlife surveys | `README.md`, `count_animals.py` | A detector on a public dataset is a tutorial; counting animals per photo is a real task with its own success measure. |
| Dataset audit | `scripts/audit_dataset.py` | Checks label files and boxes, and finds the same photo in two splits (exact file hash + perceptual hash). Found 21 test images (9%) that are copies of train/val images. |
| Leakage-free evaluation | `scripts/evaluate.py` | Re-scores on the test set without those copies: mAP@50-95 0.826 → 0.819. Small drop, but now the number is honest. |
| Error analysis | `scripts/evaluate.py` | Recall by class and box size (small/medium animals are the weak spot), class confusions, and confident detections with no label (found 2 missing labels and 2 out-of-class animals). |
| Counting threshold tuned on val | `scripts/evaluate.py` | The default 0.25 over-counts. 0.65 (chosen on val) cut counting errors per image by 38% on the clean test set. |
| ONNX export + hand-written ONNX Runtime pipeline | `scripts/export_benchmark.py`, `detector_onnx.py` | Same skill as the YOLOv3-tiny notebook (decode + NMS by hand), applied to YOLOv8. Serving no longer needs PyTorch: ~206 MB environment instead of >1.2 GB. Checked against Ultralytics: 158/160 boxes match. |
| FastAPI service + Dockerfile + 4 tests | `api.py`, `Dockerfile`, `tests/` | Upload a photo, get counts and boxes. Rejects non-images (415) and files over 10 MB (413). |
| `train.py` | root | One command repeats the training settings from `args.yaml`. |

## Not verified here

- Training was not re-run (no GPU). All new numbers use the existing `best.pt`.
- `docker build` was not run (Docker Hub blocked in the build environment); the serving requirements were tested in a clean virtual environment.
- Latency numbers come from a 2-vCPU cloud sandbox; re-measure on your machine before quoting them.
