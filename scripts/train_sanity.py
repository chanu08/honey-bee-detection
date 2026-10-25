import os
import torch
import json
import pandas as pd
from ultralytics import YOLO

def main():
    base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
    os.chdir(base_dir)

    print("--- GPU Information ---")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
        print(f"Allocated VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
        print(f"Reserved VRAM: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
    else:
        print("CUDA is NOT available!")
        exit(1)

    print("\n--- YOLO11 Sanity Check Training ---")
    model = YOLO("yolo11n.pt")
    print(f"Model Task: {model.task}")

    # Train
    results = model.train(
        data="dataset/data.yaml",
        epochs=1,
        imgsz=640,
        batch=-1,
        device=0,
        workers=2,
        amp=True,
        project="runs/detect",
        name="honey_bee_sanity",
        exist_ok=True
    )

    print("\n--- Training Results ---")
    results_file = os.path.join(base_dir, "runs", "detect", "honey_bee_sanity", "results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        
        latest = df.iloc[-1]
        
        # Train loss sum
        train_loss = 0
        val_loss = 0
        for col in df.columns:
            if col.startswith('train/') and 'loss' in col:
                train_loss += latest[col]
            if col.startswith('val/') and 'loss' in col:
                val_loss += latest[col]
                
        report = {
            "epochs_completed": 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": latest.get('metrics/precision(B)', 0),
            "recall": latest.get('metrics/recall(B)', 0),
            "mAP50": latest.get('metrics/mAP50(B)', 0),
            "mAP50_95": latest.get('metrics/mAP50-95(B)', 0)
        }
        
        with open("outputs/sanity_metrics.json", "w") as f:
            json.dump(report, f, indent=4)
            
        print(json.dumps(report, indent=4))
    else:
        print(f"Results file not found at {results_file}")

if __name__ == '__main__':
    # Required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
