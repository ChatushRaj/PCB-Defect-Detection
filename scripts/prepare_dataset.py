"""Build a YOLOv5-ready dataset from PCB_DATASET.

Converts every Pascal-VOC XML annotation to YOLO format and writes a
reproducible, class-stratified train/val split:

    dataset-pcb/
    ├── images/{train,val}/*.jpg
    └── labels/{train,val}/*.txt

Usage (from the repository root):
    python scripts/prepare_dataset.py                 # 80/20 split, seed 42
    python scripts/prepare_dataset.py --val 0.2 --seed 7 --out dataset-pcb
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = [c.strip() for c in (ROOT / "XmlToTxt-master" / "classes.txt").read_text().splitlines() if c.strip()]


def voc_to_yolo(xml_path: Path) -> list[str]:
    """Return YOLO label lines (class cx cy w h, normalised) for one XML file."""
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip()
        if name not in CLASSES:
            sys.exit(f"Unknown class '{name}' in {xml_path} - add it to XmlToTxt-master/classes.txt")
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
        w, h = (xmax - xmin) / width, (ymax - ymin) / height
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "PCB_DATASET"), help="folder with Annotations/ and images/")
    parser.add_argument("--out", default=str(ROOT / "dataset-pcb"), help="output folder")
    parser.add_argument("--val", type=float, default=0.2, help="validation fraction per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data, out = Path(args.data), Path(args.out)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0}
    for class_dir in sorted((data / "Annotations").iterdir()):
        xmls = sorted(class_dir.glob("*.xml"))
        rng.shuffle(xmls)
        n_val = round(len(xmls) * args.val)
        for i, xml_path in enumerate(xmls):
            split = "val" if i < n_val else "train"
            image = data / "images" / class_dir.name / (xml_path.stem + ".jpg")
            if not image.exists():
                sys.exit(f"Missing image for {xml_path}")
            shutil.copy2(image, out / "images" / split / image.name)
            (out / "labels" / split / (xml_path.stem + ".txt")).write_text("\n".join(voc_to_yolo(xml_path)) + "\n")
            counts[split] += 1

    print(f"Wrote {counts['train']} train and {counts['val']} val images to {out}")


if __name__ == "__main__":
    main()
