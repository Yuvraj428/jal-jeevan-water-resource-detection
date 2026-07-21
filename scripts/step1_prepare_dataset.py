import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter 


SRC_ROOT = Path("./raw_dataset")  


CLASS_NAMES = ['handpump', 'lake', 'pine', 'pine_dry', 'well']
cls2id = {c: i for i, c in enumerate(CLASS_NAMES)}

DATASET   = Path("./dataset")
TRAIN_SRC = SRC_ROOT / "train"
TEST_SRC  = SRC_ROOT / "test"
IMG_EXT   = {'.jpg', '.jpeg', '.png'}


def voc_to_yolo(xml_path: Path, dst_label: Path) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        W = float(root.find('size/width').text)
        H = float(root.find('size/height').text)
    except (AttributeError, TypeError):
        print(f"  [WARN] Cannot read image size from {xml_path.name}, skipping.")
        return False

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in cls2id:
            print(f"  [WARN] Unknown class '{name}' in {xml_path.name}, skipping object.")
            continue
        cid = cls2id[name]
        bb  = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        cx = ((xmin + xmax) / 2) / W
        cy = ((ymin + ymax) / 2) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if lines:
        dst_label.write_text('\n'.join(lines))
        return True
    return False


def process_split(src_folder: Path, split: str):
    img_dir = DATASET / split / "images"
    lbl_dir = DATASET / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for f in sorted(src_folder.iterdir()):
        if f.suffix.lower() not in IMG_EXT:
            continue
        xml = f.with_suffix('.xml')
        if not xml.exists():
            print(f"  [WARN] No XML for {f.name}, skipping.")
            skipped += 1
            continue
        shutil.copy2(f, img_dir / f.name)
        ok = voc_to_yolo(xml, lbl_dir / (f.stem + '.txt'))
        if ok:
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def count_classes(label_dir: Path):
    c = Counter()
    for lf in label_dir.glob('*.txt'):
        for line in lf.read_text().strip().splitlines():
            cid = int(line.split()[0])
            c[CLASS_NAMES[cid]] += 1
    return c


def write_yaml():
    yaml_path = DATASET / "data.yaml"
    content = f"""# JJH YOLOv8s — Local Training Config


path: {str(DATASET.resolve())}
train: train/images
val:   train/images
test:  test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(content)
    print(f"\ndata.yaml written → {yaml_path.resolve()}")
    return yaml_path


if __name__ == "__main__":
    print("=" * 55)
    print("JJH Dataset Preparation — Pascal VOC → YOLO")
    print("=" * 55)

    if not TRAIN_SRC.exists():
        print(f"\n[ERROR] Source train folder not found: {TRAIN_SRC.resolve()}")
        print("  → Edit SRC_ROOT in this script to point at your extracted zip folder.")
        exit(1)

    print(f"\nSource : {SRC_ROOT.resolve()}")
    print(f"Output : {DATASET.resolve()}\n")

    print("Processing TRAIN split ...")
    t_ok, t_skip = process_split(TRAIN_SRC, "train")
    print(f"  ✓ {t_ok} images copied  |  {t_skip} skipped\n")

    print("Processing TEST split ...")
    e_ok, e_skip = process_split(TEST_SRC, "test")
    print(f"  ✓ {e_ok} images copied  |  {e_skip} skipped\n")

    print("Class distribution (instances):")
    tr_c = count_classes(DATASET / "train" / "labels")
    te_c = count_classes(DATASET / "test"  / "labels")
    print(f"  {'Class':<12} {'Train':>8} {'Test':>6}")
    print("  " + "-" * 28)
    for cls in CLASS_NAMES:
        print(f"  {cls:<12} {tr_c[cls]:>8} {te_c[cls]:>6}")

    write_yaml()
    print("\n[DONE] Dataset is ready. Run step2_train.py next.")
