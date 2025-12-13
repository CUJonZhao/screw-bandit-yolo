# Bandit-Based Sample Reweighting for YOLO Screw Detection

This repo contains the code and artifacts for a two-stage pipeline:
1) Train a baseline YOLO detector (stage-1) and extract per-image error statistics into `state.csv`.
2) Compute per-image difficulty and generate training weights (`weights.csv`).
3) Construct reweighted training data via oversampling and run stage-2 fine-tuning.
4) Evaluate baseline (M0), unweighted fine-tune control (M1), and weighted fine-tune (M2).

## Environment
- Ultralytics: 8.3.x
- Python: 3.12
- GPU: CUDA supported (optional)

## Key artifacts
- `state.csv`: per-image statistics from stage-1 inference
- `error_summary.csv`: aggregated error summary
- `weights.csv`: bandit-derived sample weights
- `m1_val.log`, `m2_val.log`: evaluation logs for M1 and M2

## Reproducibility (commands)
Create dataset yaml (adjust paths to your dataset):
- `data_base.yaml`: base dataset split
- `data_uniform.yaml`: uniform-expanded training set (matched size)
- `data_weighted.yaml`: weighted training set

Evaluate M0 (baseline checkpoint):
yolo val model="/path/to/M0/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5

Train M1 (unweighted fine-tune, K=30):
yolo train model="/path/to/M0/best.pt" data=data_uniform.yaml epochs=30 imgsz=640 batch=16 seed=0 patience=0 device=0 amp=False project=runs/detect name=ft_unweighted_uniform

Train M2 (weighted fine-tune, K=30):
yolo train model="/path/to/M0/best.pt" data=data_weighted.yaml epochs=30 imgsz=640 batch=16 seed=0 patience=0 device=0 amp=False project=runs/detect name=ft_weighted

Evaluate M1/M2:
yolo val model="runs/detect/ft_unweighted_uniform/weights/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5
yolo val model="runs/detect/ft_weighted/weights/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5

## How to reproduce Table 2 (minimal)
1) Place your dataset locally (images/labels split into train/val) and edit the yaml `path` accordingly.
2) Evaluate baseline M0:
yolo val model="/path/to/M0/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5

3) Run controlled fine-tuning and evaluation:
yolo train model="/path/to/M0/best.pt" data=data_uniform.yaml epochs=30 imgsz=640 batch=16 seed=0 patience=0 device=0 amp=False project=runs/detect name=ft_unweighted_uniform
yolo val model="runs/detect/ft_unweighted_uniform/weights/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5

yolo train model="/path/to/M0/best.pt" data=data_weighted.yaml epochs=30 imgsz=640 batch=16 seed=0 patience=0 device=0 amp=False project=runs/detect name=ft_weighted
yolo val model="runs/detect/ft_weighted/weights/best.pt" data=data_base.yaml imgsz=640 conf=0.25 iou=0.5
