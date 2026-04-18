"""
build_dataset.py
Merges PlantDoc + FieldPlant + plantsegv2 into data/unified/{train,val,test}/{class}/

Exact structure expected:
  PlantDoc Classification dataset/{train,test}/{ClassName}/images...
  FieldPlant/train/  (flat, class encoded in filename)
  plantsegv2/images/{train,val,test}/images...
  plantsegv2/coco_annotations.json  (single file, image-level labels used)
"""

import re, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────
BASE           = Path(r"C:\Project\Emseble")
PLANTDOC_DIR   = BASE / "PlantDoc Classification dataset"
FIELDPLANT_DIR = BASE / "FieldPlant" / "train"          # flat folder
PLANTSEG_IMG   = BASE / "plantsegv2" / "images"         # {train,val,test} inside
PLANTSEG_JSON  = BASE / "plantsegv2" / "coco_annotations.json"
OUT_DIR        = BASE / "data" / "unified"

IMG_SIZE    = (224, 224)
MIN_SAMPLES = 20
SPLITS      = dict(train=0.70, val=0.10, test=0.20)
SEED        = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── LABEL MAPS ──────────────────────────────────────────

PLANTDOC_MAP = {
    "Apple_Scab_Leaf":                       "apple_scab",
    "Apple_leaf":                            "apple_healthy",
    "Apple_rust_leaf":                       "apple_rust",
    "Bell_pepper_leaf":                      "pepper_healthy",
    "Bell_pepper_leaf_spot":                 "pepper_bacterial_spot",
    "Blueberry_leaf":                        "blueberry_healthy",
    "Cherry_leaf":                           "cherry_healthy",
    "Corn_Gray_leaf_spot":                   "corn_gray_leaf_spot",
    "Corn_leaf_blight":                      "corn_leaf_blight",
    "Corn_rust_leaf":                        "corn_rust",
    "Peach_leaf":                            "peach_healthy",
    "Potato_leaf_early_blight":              "potato_early_blight",
    "Potato_leaf_late_blight":               "potato_late_blight",
    "Raspberry_leaf":                        "raspberry_healthy",
    "Soyabean_leaf":                         "soybean_healthy",
    "Squash_Powdery_mildew_leaf":            "squash_powdery_mildew",
    "Strawberry_leaf":                       "strawberry_healthy",
    "Tomato_Early_blight_leaf":              "tomato_early_blight",
    "Tomato_Septoria_leaf_spot":             "tomato_septoria_spot",
    "Tomato_leaf":                           "tomato_healthy",
    "Tomato_leaf_bacterial_spot":            "tomato_bacterial_spot",
    "Tomato_leaf_late_blight":               "tomato_late_blight",
    "Tomato_leaf_mosaic_virus":              "tomato_mosaic_virus",
    "Tomato_leaf_yellow_virus":              "tomato_yellow_leaf_curl",
    "Tomato_mold_leaf":                      "tomato_leaf_mold",
    "Tomato_two_spotted_spider_mites_leaf":  "tomato_spider_mites",
    "grape_leaf":                            "grape_healthy",
    "grape_leaf_black_rot":                  "grape_black_rot",
}

# None = intentionally dropped
FIELDPLANT_MAP = {
    # Corn / Maize
    "mais brulure":                "corn_leaf_blight",
    "mais brulures feuilles":      "corn_leaf_blight",
    "mais jaunissement":           "corn_yellowing",
    "jaunissement dessechement":   "corn_yellowing",
    "mais mildiou":                "corn_downy_mildew",
    "mais striure":                "corn_streak_virus",
    "striure du mais":             "corn_streak_virus",
    "mais rayure":                 "corn_streak_virus",
    "mais taches brunes":          "corn_brown_spot",
    "mais taches jaunes":          "corn_yellow_spot",
    "taches jaunes":               "corn_yellow_spot",
    "mais taches chlorotiques":    "corn_chlorotic_spot",
    "mais cercosporiose":          "corn_gray_leaf_spot",
    "mais sains":                  "corn_healthy",
    "mais rouille":                "corn_rust",
    "rouille de mais":             "corn_rust",
    "mais decoloration violette":  "corn_purple_discoloration",
    "mais degats insectes":        "corn_insect_damage",
    "mais charbons":               "corn_smut",
    "charbon de mais":             "corn_smut",
    "coloration rouge feuilles":   "corn_red_leaf",
    # Cassava / Manioc
    "manioc mosaique":             "cassava_mosaic",
    "manioc healthy":              "cassava_healthy",
    "manioc taches brunes":        "cassava_brown_spot",
    "manioc bacteriose":           "cassava_bacterial_blight",
    "manioc pourriture tubercules": None,   # only 2 samples
    # Tomato
    "tomate taches brunes":        "tomato_early_blight",
    "tomate mildiou":              "tomato_late_blight",
    "tomate mosaique":             "tomato_mosaic_virus",
    "tomate feuilles saines":      "tomato_healthy",
    "tomate fletrissement":        None,    # only 1 sample
    # Noise / unlabeled
    "img":                         None,
    "annotations.csv":             None,
}

PLANTSEG_OVERRIDES = {
    "soyabean rust":    "soybean_rust",
    "soyabean healthy": "soybean_healthy",
}

# ─── HELPERS ─────────────────────────────────────────────

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def file_hash(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(8192))
    return h.hexdigest()

def to_snake(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())

def parse_fieldplant_label(filename: str):
    name = filename
    if ".rf." in name:
        name = name[:name.index(".rf.")]
    name = re.sub(r"-\d+-?$", "", name)
    name = re.sub(r"_jpe?g$|_png$", "", name, flags=re.IGNORECASE)
    label = name.replace("_", " ").replace("-", " ").strip().lower()
    label = re.sub(r"\s+\d+\s*$", "", label).strip()
    return label or None

def copy_resized(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=95)
    except Exception as e:
        print(f"  [WARN] skip {src.name}: {e}")

# ─── COLLECTORS ──────────────────────────────────────────

def collect_plantdoc():
    data = defaultdict(list)
    unmapped = Counter()
    for split in ["train", "test"]:
        split_dir = PLANTDOC_DIR / split
        if not split_dir.exists():
            print(f"[WARN] PlantDoc/{split} not found"); continue
        for folder in split_dir.iterdir():
            if not folder.is_dir(): continue
            unified = PLANTDOC_MAP.get(folder.name)
            if unified is None:
                unmapped[folder.name] += 1; continue
            data[unified].extend(p for p in folder.iterdir() if is_image(p))
    if unmapped:
        print(f"  [PlantDoc] unmapped folders: {dict(unmapped)}")
    return data

def collect_fieldplant():
    data = defaultdict(list)
    unmapped = Counter()
    if not FIELDPLANT_DIR.exists():
        print("[WARN] FieldPlant/train not found"); return data
    for p in FIELDPLANT_DIR.iterdir():
        if not is_image(p): continue
        raw = parse_fieldplant_label(p.name)
        if raw is None: continue
        # Drop timestamp-style names (img20221216..., img 20221118...)
        if re.match(r"img\s*\d{6,}", raw): continue
        if raw not in FIELDPLANT_MAP:
            unmapped[raw] += 1; continue
        unified = FIELDPLANT_MAP[raw]
        if unified is None: continue
        data[unified].append(p)
    if unmapped:
        print(f"  [FieldPlant] unmapped labels — add these to FIELDPLANT_MAP:")
        for k, v in sorted(unmapped.items(), key=lambda x: -x[1]):
            print(f"    {v:5d}  '{k}'")
    return data

def collect_plantsegv2():
    """
    Single coco_annotations.json covers all splits.
    Images live in plantsegv2/images/{train,val,test}/
    Uses image-level label = dominant annotation category.
    """
    data = defaultdict(list)
    if not PLANTSEG_JSON.exists():
        print("[WARN] plantsegv2/coco_annotations.json not found"); return data

    with open(PLANTSEG_JSON) as f:
        coco = json.load(f)

    cat_map = {}
    for c in coco["categories"]:
        name = c["name"].lower().strip()
        cat_map[c["id"]] = PLANTSEG_OVERRIDES.get(name, to_snake(name))

    img_id_to_fname = {i["id"]: Path(i["file_name"]).name for i in coco["images"]}

    # Build dominant category per image
    img_cats = defaultdict(Counter)
    for ann in coco["annotations"]:
        img_cats[ann["image_id"]][ann["category_id"]] += 1

    # Index all images by filename across all splits
    fname_to_path = {}
    for split in ["train", "val", "test"]:
        d = PLANTSEG_IMG / split
        if not d.exists(): continue
        for p in d.iterdir():
            if is_image(p):
                fname_to_path[p.name] = p

    matched = missing = 0
    for img_id, cat_counter in img_cats.items():
        label = cat_map[cat_counter.most_common(1)[0][0]]
        fname = img_id_to_fname.get(img_id)
        if not fname: continue
        
        # Try exact match first, then fallback to prefixed name
        path = fname_to_path.get(fname)
        if path is None:
            path = fname_to_path.get(f"{label}_{fname}")
            
        if path is None:
            missing += 1; continue
            
        data[label].append(path)
        matched += 1

    print(f"  [plantsegv2] matched {matched} images, could not find {missing}")
    return data

# ─── MAIN ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: Collecting images...")
    all_data = collect_plantdoc()
    for label, paths in collect_fieldplant().items():
        all_data[label].extend(paths)
    for label, paths in collect_plantsegv2().items():
        all_data[label].extend(paths)

    print("\nStep 2: Deduplicating by MD5 hash...")
    seen = set()
    deduped = defaultdict(list)
    for label, paths in all_data.items():
        for p in paths:
            h = file_hash(p)
            if h not in seen:
                seen.add(h)
                deduped[label].append(p)

    print("\nStep 3: Filtering low-sample classes...")
    filtered = {k: v for k, v in deduped.items() if len(v) >= MIN_SAMPLES}
    dropped  = {k: len(v) for k, v in deduped.items() if len(v) < MIN_SAMPLES}
    if dropped:
        print(f"  Dropped {len(dropped)} classes with < {MIN_SAMPLES} samples:")
        for k, v in sorted(dropped.items()): print(f"    {v:4d}  {k}")

    print(f"\n{'CLASS':<45} {'IMAGES':>6}")
    print("-" * 55)
    grand_total = 0
    for label in sorted(filtered):
        n = len(filtered[label])
        print(f"  {label:<43} {n:>6}")
        grand_total += n
    print(f"\n  {len(filtered)} classes | {grand_total} total images")

    print(f"\nStep 4: Splitting & copying to {OUT_DIR} ...")
    split_counts = Counter()
    for label, paths in tqdm(filtered.items()):
        paths_str = [str(p) for p in paths]
        train_val, test = train_test_split(
            paths_str, test_size=SPLITS["test"], random_state=SEED)
        val_adj = SPLITS["val"] / (SPLITS["train"] + SPLITS["val"])
        train, val = train_test_split(
            train_val, test_size=val_adj, random_state=SEED)

        for split_name, split_paths in [("train", train), ("val", val), ("test", test)]:
            for i, src in enumerate(split_paths):
                dst = OUT_DIR / split_name / label / f"{label}_{i:05d}.jpg"
                copy_resized(Path(src), dst)
                split_counts[split_name] += 1

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  train : {split_counts['train']:6d} images")
    print(f"  val   : {split_counts['val']:6d} images")
    print(f"  test  : {split_counts['test']:6d} images")
    print(f"  total : {sum(split_counts.values()):6d} images")
    print(f"\nOutput: {OUT_DIR}")
    print("Next step: python train.py")

if __name__ == "__main__":
    main()