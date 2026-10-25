import os
import glob
import shutil
import yaml
import json

base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
extracted_dirs = glob.glob(os.path.join(base_dir, 'dataset_raw', '**', 'detection'), recursive=True)
detection_dir = extracted_dirs[0]
dataset_dir = os.path.join(base_dir, "dataset")

splits = {
    "train": ["_bee_20230609a", "_bee_20230609b", "_bee_20230609d", "_bee_20230609e", "_bee_20230711a"],
    "val": ["_bee_20230609c"],
    "test": ["_bee_20230711b", "_bee_20230711c"]
}

# 1. Create directories
for s in splits.keys():
    os.makedirs(os.path.join(dataset_dir, "images", s), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", s), exist_ok=True)

# 2 & 3. Copy files and handle missing labels
print("Copying dataset files. This may take a moment...")
for split_name, sessions in splits.items():
    for session in sessions:
        session_path = os.path.join(detection_dir, session)
        images_dir = os.path.join(session_path, 'images')
        labels_dir = os.path.join(session_path, 'labels')
        
        images = glob.glob(os.path.join(images_dir, '*'))
        for img_path in images:
            fname = os.path.basename(img_path)
            dest_img = os.path.join(dataset_dir, "images", split_name, fname)
            
            # Copy image
            if not os.path.exists(dest_img):
                shutil.copy2(img_path, dest_img)
            
            # Handle label
            base_name = os.path.splitext(fname)[0]
            src_label = os.path.join(labels_dir, base_name + '.txt')
            dest_label = os.path.join(dataset_dir, "labels", split_name, base_name + '.txt')
            
            if os.path.exists(src_label):
                if not os.path.exists(dest_label):
                    shutil.copy2(src_label, dest_label)
            else:
                # Create empty label
                with open(dest_label, 'w') as f:
                    pass

# 7. Generate data.yaml
yaml_path = os.path.join(dataset_dir, "data.yaml")
yaml_data = {
    "path": dataset_dir.replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "honey_bee"}
}
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("Dataset construction complete. Starting verification...")

# 8. Complete Verification
report = {
    "TRAIN": {"images": 0, "labels": 0, "empty_labels": 0, "total_bboxes": 0},
    "VAL": {"images": 0, "labels": 0, "empty_labels": 0, "total_bboxes": 0},
    "TEST": {"images": 0, "labels": 0, "empty_labels": 0, "total_bboxes": 0}
}

verification = {
    "all_images_have_labels": True,
    "all_labels_valid_yolo": True,
    "all_class_ids_0": True,
    "all_coords_normalized": True,
    "all_dimensions_positive": True,
    "no_bboxes_outside_image": True,
    "no_orphaned_labels": True,
    "no_duplicate_filenames": True
}

all_filenames = set()

for split_name in splits.keys():
    upper_split = split_name.upper()
    split_images_dir = os.path.join(dataset_dir, "images", split_name)
    split_labels_dir = os.path.join(dataset_dir, "labels", split_name)
    
    images = glob.glob(os.path.join(split_images_dir, '*'))
    labels = glob.glob(os.path.join(split_labels_dir, '*.txt'))
    
    report[upper_split]["images"] = len(images)
    report[upper_split]["labels"] = len(labels)
    
    if len(images) != len(labels):
        verification["all_images_have_labels"] = False
        verification["no_orphaned_labels"] = False
        
    for img_path in images:
        fname = os.path.basename(img_path)
        if fname in all_filenames:
            verification["no_duplicate_filenames"] = False
        all_filenames.add(fname)
        
        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(split_labels_dir, base_name + '.txt')
        
        if not os.path.exists(label_path):
            verification["all_images_have_labels"] = False
            continue
            
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            
        is_empty = True
        bboxes_count = 0
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            is_empty = False
            
            if len(parts) != 5:
                verification["all_labels_valid_yolo"] = False
                continue
                
            try:
                c, x, y, w, h = map(float, parts)
                if int(c) != 0:
                    verification["all_class_ids_0"] = False
                    
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    verification["all_coords_normalized"] = False
                    
                if w <= 0.0 or h <= 0.0:
                    verification["all_dimensions_positive"] = False
                    
                # check if bounding box extends outside normalized image coordinates
                # x is center. left is x - w/2, right is x + w/2
                # Note: floating point inaccuracies might cause slight out of bounds, so allow epsilon
                eps = 1e-4
                if (x - w/2 < -eps) or (x + w/2 > 1.0 + eps) or (y - h/2 < -eps) or (y + h/2 > 1.0 + eps):
                    # Ultralytics often clamps, but let's strictly check
                    verification["no_bboxes_outside_image"] = False
                    
                bboxes_count += 1
            except ValueError:
                verification["all_labels_valid_yolo"] = False
                
        if is_empty or bboxes_count == 0:
            report[upper_split]["empty_labels"] += 1
            
        report[upper_split]["total_bboxes"] += bboxes_count

# Check orphaned labels (labels without images)
for split_name in splits.keys():
    split_images_dir = os.path.join(dataset_dir, "images", split_name)
    split_labels_dir = os.path.join(dataset_dir, "labels", split_name)
    labels = glob.glob(os.path.join(split_labels_dir, '*.txt'))
    for label_path in labels:
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        # Image could be .jpg
        img_path = os.path.join(split_images_dir, base_name + '.jpg')
        if not os.path.exists(img_path):
            verification["no_orphaned_labels"] = False

total_images = sum(report[s]["images"] for s in report)
total_bboxes = sum(report[s]["total_bboxes"] for s in report)

print("\n--- VERIFICATION REPORT ---")
print(json.dumps(report, indent=4))
print(f"\nTotal images: {total_images} (Expected: 7200)")
print(f"Total bounding boxes: {total_bboxes} (Expected: 54815)")
print("\nValidation Checks:")
for k, v in verification.items():
    print(f"- {k}: {v}")

# Write to file
report_path = os.path.join(base_dir, "outputs", "dataset_build_report.json")
with open(report_path, "w") as f:
    json.dump({"counts": report, "checks": verification, "totals": {"images": total_images, "bboxes": total_bboxes}}, f, indent=4)
