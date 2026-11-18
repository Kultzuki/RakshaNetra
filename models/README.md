# Model artifacts

The canonical release manifest is [manifest.json](manifest.json). Default weights live locally in `stgcnpp/`, `yolo/`, `surveillance/` and `fire/` and are excluded from Git. The verifier always checks this checkout; explicit runtime weight overrides remain configurable.

`best_fight_stgcnpp_v1.pth` contains the binary fight head. `j.pth` is an upstream representation checkpoint used for compatibility checks. The ST-GCN++ implementation loads both directly with PyTorch. The project loads `yolo26m-pose.pt` and the two auxiliary YOLO checkpoints through Ultralytics on PyTorch. Real inference requires CUDA.

Abandoned-bag detection is off by default. To opt in, set `enable_bag` to `true` and provide a compatible COCO object detection checkpoint via `det_weights` or `VIGILANCE_DET_WEIGHTS`.

Run `python tools/verify_models.py` for a hash-only check. On the GPU machine, run `python tools/verify_models.py --inference --auxiliary --device cuda` to check restricted loading, detector reset, and numerical fixtures as well.

`regression.json` anchors deterministic zero-tensor and moving-pair preprocessing/inference checks. These are runtime regressions, not semantic accuracy claims.

The auxiliary [surveillance model](https://huggingface.co/xDevine/yolo11s-multiclass-surveillance-detection) detects weapons and provides fallback fire/smoke detections. The dedicated [YOLOv26 fire model](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection) supplies fire and smoke detections by default. Their immutable Hugging Face revisions and SHA-256 digests are recorded in `core/threat_detection.py`. Downloads are verified before restricted loading. Run `python tools/verify_models.py --auxiliary` to provision and hash-check both artifacts.
