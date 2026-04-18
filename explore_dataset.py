import os, json
from collections import Counter

BASE = r"C:\Project\Emseble"

def explore_dir(path, depth=0, max_depth=3, max_files=5):
    if depth > max_depth or not os.path.isdir(path):
        return
    entries = sorted(os.listdir(path))
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    pad = "  " * depth
    for d in dirs:
        count = len(os.listdir(os.path.join(path, d)))
        print(f"{pad}[DIR]  {d}/  ({count} items)")
        explore_dir(os.path.join(path, d), depth+1, max_depth, max_files)
    for f in files[:max_files]:
        print(f"{pad}[FILE] {f}")
    if len(files) > max_files:
        print(f"{pad}       ... and {len(files)-max_files} more files")

print("=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)
explore_dir(BASE)

# ── FieldPlant: parse classes from filenames ──────────────
fp_train = os.path.join(BASE, "FieldPlant", "train")
if os.path.isdir(fp_train):
    subdirs = [d for d in os.listdir(fp_train)
               if os.path.isdir(os.path.join(fp_train, d))]
    if subdirs:
        print("\n" + "=" * 60)
        print(f"FIELDPLANT CLASSES FROM SUBFOLDERS ({len(subdirs)} total)")
        print("=" * 60)
        for c in sorted(subdirs):
            n = len(os.listdir(os.path.join(fp_train, c)))
            print(f"  {c:50s}  {n:5d} images")
    else:
        # Flat folder — extract class from filename
        files = [f for f in os.listdir(fp_train)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        class_counts = Counter()
        for f in files:
            # Roboflow pattern: ClassName-N-_jpg.rf.hash.jpg
            # Class is everything before the first "-digit-" pattern
            name = f
            # strip roboflow suffix: .rf..
            if ".rf." in name:
                name = name[:name.index(".rf.")]
            # remove trailing -N- index
            import re
            name = re.sub(r'-\d+-?$', '', name)
            # normalize
            label = name.replace("_", " ").replace("-", " ").strip().lower()
            class_counts[label] += 1

        print("\n" + "=" * 60)
        print(f"FIELDPLANT CLASSES FROM FILENAMES ({len(class_counts)} total, {len(files)} images)")
        print("=" * 60)
        for label, count in sorted(class_counts.items()):
            print(f"  {label:50s}  {count:5d} images")

# ── PlantDoc Classification ───────────────────────────────
plantdoc_train = os.path.join(BASE, "PlantDoc Classification dataset", "train")
plantdoc_test  = os.path.join(BASE, "PlantDoc Classification dataset", "test")
if os.path.isdir(plantdoc_train):
    classes = sorted([d for d in os.listdir(plantdoc_train)
                      if os.path.isdir(os.path.join(plantdoc_train, d))])
    print("\n" + "=" * 60)
    print(f"PLANTDOC CLASSES ({len(classes)} total)")
    print("=" * 60)
    for c in classes:
        n_train = len(os.listdir(os.path.join(plantdoc_train, c)))
        n_test  = len(os.listdir(os.path.join(plantdoc_test, c))) if os.path.isdir(os.path.join(plantdoc_test, c)) else 0
        print(f"  {c:50s}  train={n_train:4d}  test={n_test:4d}")

# ── plantsegv2: COCO JSON ─────────────────────────────────
coco_path = os.path.join(BASE, "plantsegv2", "coco_annotations.json")
if os.path.exists(coco_path):
    with open(coco_path) as f:
        coco = json.load(f)
    cats = coco.get("categories", [])
    imgs = coco.get("images", [])
    anns = coco.get("annotations", [])
    print("\n" + "=" * 60)
    print(f"PLANTSEGV2 — {len(imgs)} images, {len(anns)} annotations, {len(cats)} categories")
    print("=" * 60)
    cat_counts = Counter(a["category_id"] for a in anns)
    cat_map = {c["id"]: c["name"] for c in cats}
    for cid, name in sorted(cat_map.items()):
        print(f"  id={cid:3d}  {name:45s}  {cat_counts.get(cid,0):5d} anns")
    if anns and "segmentation" in anns[0]:
        print("\n  -> SEGMENTATION dataset")
    elif anns and "bbox" in anns[0]:
        print("\n  -> DETECTION dataset")

# ── plantsegv2: CSV ───────────────────────────────────────
csv_path = os.path.join(BASE, "plantsegv2", "Metadatav2.csv")
if os.path.exists(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()
    print(f"\nCSV header: {lines[0].strip()}")
    print(f"CSV rows:   {len(lines)-1}")
