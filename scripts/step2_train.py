from pathlib import Path
from ultralytics import YOLO 

DATASET  = Path("./dataset")
YAML     = DATASET / "data.yaml"
RUN_DIR  = Path("./runs")


BATCH    = 4 
EPOCHS   = 100

if not YAML.exists():
    print(f"[ERROR] data.yaml not found at {YAML.resolve()}")
    print("  → Run step1_prepare_dataset.py first.")
    exit(1)

print("=" * 55)
print("JJH YOLOv8s — Local CPU Training")
print("=" * 55)
print(f"  data.yaml : {YAML.resolve()}")
print(f"  epochs    : {EPOCHS}")
print(f"  batch     : {BATCH}")
print(f"  device    : cpu")
print("  (Training on CPU will be slow — ~5–15 min/epoch depending on your machine)")
print("=" * 55 + "\n")

model = YOLO('yolov8s.pt')
results = model.train(
    data          = str(YAML),
    epochs        = EPOCHS,
    imgsz         = 640,
    batch         = BATCH,
    device        = 'cpu',
    workers       = 0,

    lr0           = 1e-3,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,

    augment       = True,
    mosaic        = 1.0,
    mixup         = 0.1,
    degrees       = 10,
    flipud        = 0.3,
    fliplr        = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,

    patience      = 0,
    save          = True,
    project       = str(RUN_DIR.resolve()),
    name          = 'jjh_yolov8s',
    exist_ok      = True,
    verbose       = True,
    amp           = False,
)

print("\n[DONE] Training complete.")
print(f"  Best weights → {RUN_DIR / 'jjh_yolov8s' / 'weights' / 'best.pt'}")
print("  Run step3_evaluate.py next for test-set metrics.")