import os
import glob
import random
import cv2
import numpy as np

base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
dataset_dir = os.path.join(base_dir, "dataset")
output_dir = os.path.join(base_dir, "outputs", "annotation_check")

# Prepare directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

def get_label_count(label_path):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return len(lines)

metadata = {'train': [], 'val': [], 'test': []}
for split in ['train', 'val', 'test']:
    images_dir = os.path.join(dataset_dir, 'images', split)
    labels_dir = os.path.join(dataset_dir, 'labels', split)
    
    for img_path in glob.glob(os.path.join(images_dir, '*.jpg')):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        count = get_label_count(label_path)
        metadata[split].append({'img': img_path, 'label': label_path, 'count': count})

selected = {'train': [], 'val': [], 'test': []}

def pick_image(split, condition_fn):
    candidates = [item for item in metadata[split] if condition_fn(item['count'])]
    if not candidates:
        return None
    choice = random.choice(candidates)
    metadata[split].remove(choice)
    return choice

# Force 2 empty, 2 few, 2 many into train
reqs = []
for _ in range(2): reqs.append(pick_image('train', lambda c: c == 0))
for _ in range(2): reqs.append(pick_image('train', lambda c: 1 <= c <= 5))
for _ in range(2): reqs.append(pick_image('train', lambda c: c >= 20))

# Fill train up to 10
for _ in range(4):
    if metadata['train']:
        choice = random.choice(metadata['train'])
        metadata['train'].remove(choice)
        reqs.append(choice)

for item in reqs:
    if item:
        selected['train'].append(item)

# 5 val
for _ in range(5):
    choice = random.choice(metadata['val'])
    metadata['val'].remove(choice)
    selected['val'].append(choice)

# 5 test
for _ in range(5):
    choice = random.choice(metadata['test'])
    metadata['test'].remove(choice)
    selected['test'].append(choice)

total_boxes_rendered = 0
boxes_within_image = True
images_used = []

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
color = (0, 255, 0) # Green for ground truth

contact_images = []

for split, items in selected.items():
    for item in items:
        img_path = item['img']
        label_path = item['label']
        images_used.append(os.path.basename(img_path))
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    _, nx, ny, nw, nh = map(float, parts)
                    
                    x_center = nx * w
                    y_center = ny * h
                    box_w = nw * w
                    box_h = nh * h
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                        boxes_within_image = False
                    
                    # Draw
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    # Text background
                    text = "honey_bee GT"
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
                    
                    total_boxes_rendered += 1

        out_path = os.path.join(output_dir, split, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
        contact_images.append(img)

# Contact sheet
# Resize to 1/4th resolution for grid
thumbs = [cv2.resize(img, (w//4, h//4)) for img in contact_images]
rows = []
for i in range(0, 20, 5):
    row = np.hstack(thumbs[i:i+5])
    rows.append(row)
contact_sheet = np.vstack(rows)
contact_sheet_path = os.path.join(output_dir, "contact_sheet.jpg")
cv2.imwrite(contact_sheet_path, contact_sheet)

print("--- VISUAL VERIFICATION REPORT ---")
print(f"Number of images checked: {len(images_used)}")
print(f"Number of boxes rendered: {total_boxes_rendered}")
print(f"Every rendered box strictly lies within image mathematical bounds: {boxes_within_image}")
print("Images used:")
for name in images_used:
    print(f" - {name}")
print(f"\nContact sheet saved to: {contact_sheet_path}")
