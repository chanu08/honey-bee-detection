import os
import urllib.request
import zipfile
import glob
import json
try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Installing...")
    os.system("pip install pillow")
    from PIL import Image

base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
dirs = ["dataset_raw", "dataset", "test_images", "test_videos", "outputs", "scripts", "runs"]

print("1. Creating project structure...")
for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

zip_path = os.path.join(base_dir, "dataset_raw", "mendeley_dataset_v6.zip")
url = "https://data.mendeley.com/public-api/zip/8gb9r2yhfc/download/6"

if not os.path.exists(zip_path):
    print(f"2. Downloading dataset from {url} to {zip_path}...")
    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        downloaded = 0
        with open(zip_path, 'wb') as file:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                file.write(data)
                if downloaded % (100 * 1024 * 1024) < block_size:
                    if total_size > 0:
                        print(f"Downloaded {downloaded / (1024**3):.2f}GB / {total_size / (1024**3):.2f}GB")
                    else:
                        print(f"Downloaded {downloaded / (1024**3):.2f}GB")
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        exit(1)
else:
    print(f"2. Zip file already exists at {zip_path}.")

print("3. Extracting ONLY 'detection' directory...")
extracted_detection_dir = None
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    members = zip_ref.namelist()
    # Filter for paths that contain /detection/ or start with detection/
    detection_members = [m for m in members if '/detection/' in m or m.startswith('detection/')]
    
    print(f"Found {len(detection_members)} files related to 'detection/'. Extracting...")
    
    for i, m in enumerate(detection_members):
        if i % 1000 == 0:
            print(f"Extracted {i}/{len(detection_members)} files...")
        zip_ref.extract(m, os.path.join(base_dir, "dataset_raw"))
print("Extraction complete.")

print("4. Inspecting dataset...")
extracted_dirs = glob.glob(os.path.join(base_dir, "dataset_raw", "**", "detection"), recursive=True)
if not extracted_dirs:
    print("Error: Could not find 'detection' directory after extraction.")
    exit(1)

detection_dir = extracted_dirs[0]
sessions = [d for d in os.listdir(detection_dir) if os.path.isdir(os.path.join(detection_dir, d))]

report = {
    "total_images": 0,
    "total_labels": 0,
    "image_extensions": set(),
    "image_dimensions": set(),
    "total_bboxes": 0,
    "bees_per_image": [],
    "missing_labels": [],
    "malformed_labels": [],
    "unique_classes": set(),
    "all_classes_are_0": True,
    "normalized_coords_valid": True,
    "sessions_found": sessions
}

for session in sessions:
    session_path = os.path.join(detection_dir, session)
    images_dir = os.path.join(session_path, "images")
    labels_dir = os.path.join(session_path, "labels")
    
    if not os.path.exists(images_dir):
        continue
        
    images = glob.glob(os.path.join(images_dir, "*"))
    report["total_images"] += len(images)
    
    for img_path in images:
        ext = os.path.splitext(img_path)[1].lower()
        report["image_extensions"].add(ext)
        
        with Image.open(img_path) as img:
            report["image_dimensions"].add(img.size)
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        if not os.path.exists(label_path):
            report["missing_labels"].append(img_path)
            report["bees_per_image"].append(0)
            continue
            
        report["total_labels"] += 1
        bboxes_in_img = 0
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    report["malformed_labels"].append(label_path)
                    continue
                try:
                    c, x, y, w, h = map(float, parts)
                    c = int(c)
                    report["unique_classes"].add(c)
                    if c != 0:
                        report["all_classes_are_0"] = False
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        report["normalized_coords_valid"] = False
                    bboxes_in_img += 1
                    report["total_bboxes"] += 1
                except ValueError:
                    report["malformed_labels"].append(label_path)
                    
        report["bees_per_image"].append(bboxes_in_img)

report["image_extensions"] = list(report["image_extensions"])
report["image_dimensions"] = [f"{w}x{h}" for w, h in report["image_dimensions"]]
report["unique_classes"] = list(report["unique_classes"])
if report["bees_per_image"]:
    report["min_bees"] = min(report["bees_per_image"])
    report["max_bees"] = max(report["bees_per_image"])
    report["avg_bees"] = sum(report["bees_per_image"]) / len(report["bees_per_image"])
else:
    report["min_bees"] = report["max_bees"] = report["avg_bees"] = 0

del report["bees_per_image"]
report["num_missing_labels"] = len(report["missing_labels"])
report["num_malformed_labels"] = len(report["malformed_labels"])
# truncate list for printing if too long
report["missing_labels_sample"] = report["missing_labels"][:10]
report["malformed_labels_sample"] = report["malformed_labels"][:10]
del report["missing_labels"]
del report["malformed_labels"]

print("\n--- INSPECTION REPORT ---")
print(json.dumps(report, indent=4))

report_path = os.path.join(base_dir, "outputs", "dataset_inspection_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)
print(f"\nReport saved to {report_path}")
