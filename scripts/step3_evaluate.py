
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

DATASET   = Path("./dataset")
YAML      = DATASET / "data.yaml"
RUN_DIR   = Path("./runs/jjh_yolov8s")
BEST_PT   = RUN_DIR / "weights" / "best.pt"
OUT_DIR   = RUN_DIR / "test_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['handpump', 'lake', 'pine', 'pine_dry', 'well']


if not BEST_PT.exists():
    print(f"[ERROR] best.pt not found at {BEST_PT.resolve()}")
    print("  → Run step2_train.py first.")
    exit(1)

print("=" * 55)
print("JJH YOLOv8s — Test Set Evaluation")
print("=" * 55)

model = YOLO(str(BEST_PT))


print("\n[1/4] Running test-set evaluation ...")
test_results = model.val(
    data    = str(YAML),
    split   = 'test',
    imgsz   = 640,
    conf    = 0.25,
    iou     = 0.5,
    verbose = True, 
    device  = 'cpu',
    workers = 0,
    project = str(OUT_DIR.resolve()),
    name    = 'metrics',
    exist_ok= True,
)

map50    = test_results.box.map50
map5095  = test_results.box.map
prec     = test_results.box.mp
rec      = test_results.box.mr
ap50     = test_results.box.ap50
ap5095   = test_results.box.ap

print("\n" + "=" * 40)
print("  TEST SET RESULTS")
print("=" * 40)
print(f"  mAP@50      : {map50:.4f}")
print(f"  mAP@50-95   : {map5095:.4f}")
print(f"  Precision   : {prec:.4f}")
print(f"  Recall      : {rec:.4f}")
print("=" * 40)
print(f"\n  {'Class':<12} {'AP@50':>8} {'AP@50-95':>10}")
print("  " + "-" * 34)
rows = []
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<12} {ap50[i]:>8.4f} {ap5095[i]:>10.4f}")
    rows.append({'class': cls, 'AP@50': round(float(ap50[i]), 4), 'AP@50-95': round(float(ap5095[i]), 4)})
print("  " + "-" * 34)
print(f"  {'mAP':<12} {np.mean(ap50):>8.4f} {np.mean(ap5095):>10.4f}")

# Save CSV
csv_path = OUT_DIR / "test_per_class_ap.csv"
rows.append({'class': 'mAP', 'AP@50': round(float(np.mean(ap50)), 4), 'AP@50-95': round(float(np.mean(ap5095)), 4)})
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\n  Per-class AP saved → {csv_path}")

#  Per-class AP bar chart 
print("\n[2/4] Saving per-class AP chart ...")
x  = np.arange(len(CLASS_NAMES))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, ap50,   0.38, label='AP@50',    color='#2196F3')
ax.bar(x + 0.2, ap5095, 0.38, label='AP@50-95', color='#FF9800')
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Average Precision')
ax.set_title('Per-class AP — Test Set (JJH YOLOv8s)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (a50, a5095) in enumerate(zip(ap50, ap5095)):
    ax.text(i - 0.2, a50 + 0.02,   f'{a50:.2f}',   ha='center', fontsize=8)
    ax.text(i + 0.2, a5095 + 0.02, f'{a5095:.2f}', ha='center', fontsize=8)
plt.tight_layout()
chart_path = OUT_DIR / "test_per_class_ap.png"
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"  Saved → {chart_path}")

# Training Curves
print("\n[3/4] Saving training curves ...")
csv_train = RUN_DIR / "results.csv"
if csv_train.exists():
    df = pd.read_csv(csv_train)
    df.columns = df.columns.str.strip()

    pairs = [
        ('train/box_loss', 'val/box_loss',             'Box Loss'),
        ('train/cls_loss', 'val/cls_loss',             'Class Loss'),
        ('train/dfl_loss', 'val/dfl_loss',             'DFL Loss'),
        ('metrics/precision(B)', None,                  'Precision'),
        ('metrics/recall(B)',    None,                  'Recall'),
        ('metrics/mAP50(B)',     'metrics/mAP50-95(B)', 'mAP'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col1, col2, title) in zip(axes.flat, pairs):
        if col1 in df:
            ax.plot(df['epoch'], df[col1], label=col1.split('/')[-1])
        if col2 and col2 in df:
            ax.plot(df['epoch'], df[col2], label=col2.split('/')[-1], linestyle='--')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('YOLOv8s Training Curves — Jal Jeevan Hariyali', fontsize=13)
    plt.tight_layout()
    curves_path = OUT_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Saved → {curves_path}")
else:
    print("  [WARN] results.csv not found, skipping training curves.")

# Sample Predictions on Test Set
print("\n[4/4] Generating sample predictions grid ...")
test_imgs = list((DATASET / "test" / "images").glob("*.*"))
if test_imgs:
    sample = random.sample(test_imgs, min(8, len(test_imgs)))
    preds  = model.predict(source=sample, conf=0.25, iou=0.5, save=False, device='cpu')

    n = len(preds)
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flat if rows > 1 else [axes] if n == 1 else axes
    for ax, result in zip(axes, preds):
        rendered = result.plot()
        ax.imshow(rendered[..., ::-1])
        ax.set_title(Path(result.path).name, fontsize=8)
        ax.axis('off')
    for ax in list(axes)[n:]:  
        ax.axis('off')
    plt.suptitle('Sample Predictions — Test Set (conf≥0.25)', fontsize=12)
    plt.tight_layout()
    preds_path = OUT_DIR / "sample_predictions.png"
    plt.savefig(preds_path, dpi=150)
    plt.close()
    print(f"  Saved → {preds_path}")
else:
    print("  [WARN] No test images found.")

print("\n" + "=" * 55)
print("All outputs saved in:", OUT_DIR.resolve())
print("=" * 55)
