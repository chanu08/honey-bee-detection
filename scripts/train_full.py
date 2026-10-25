import os
import torch
import json
import pandas as pd
import time
from ultralytics import YOLO

def main():
    base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
    os.chdir(base_dir)

    print("--- YOLO11 Full Training Started ---")
    start_time = time.time()
    
    oom_occurred = False
    try:
        model = YOLO("yolo11n.pt")
        
        results = model.train(
            data="dataset/data.yaml",
            epochs=100,
            imgsz=640,
            batch=13,
            device=0,
            workers=2,
            amp=True,
            patience=20,
            project="runs/detect",
            name="honey_bee_yolo11n_640",
            exist_ok=True
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "OOM" in str(e):
            oom_occurred = True
            print("CUDA OOM occurred!")
        else:
            raise e
            
    end_time = time.time()
    total_time_hours = (end_time - start_time) / 3600.0
    
    print("\n--- Training Finished ---")
    
    results_dir = os.path.join(base_dir, "runs", "detect", "honey_bee_yolo11n_640")
    results_csv = os.path.join(results_dir, "results.csv")
    
    if os.path.exists(results_csv) and not oom_occurred:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        actual_epochs = len(df)
        early_stopping = actual_epochs < 100
        
        # Best epoch based on mAP50-95
        best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_row = df.iloc[best_epoch_idx]
        last_row = df.iloc[-1]
        
        # df index starts at 0, epoch might be a column or we just add 1
        best_epoch = best_row.get('epoch', best_epoch_idx + 1)
        if isinstance(best_epoch, pd.Series):
            best_epoch = int(best_epoch.iloc[0])
        else:
            best_epoch = int(best_epoch)
        
        vram = 0.0
        if torch.cuda.is_available():
            # Approx max memory allocated during this process
            vram = torch.cuda.max_memory_allocated(0) / (1024**3)
            # if 0, then we just fallback to checking total allocated
            if vram == 0:
                vram = torch.cuda.memory_allocated(0) / (1024**3)
                
        report = {
            "total_training_time_hrs": round(total_time_hours, 3),
            "actual_epochs_completed": actual_epochs,
            "best_epoch": best_epoch,
            "best_checkpoint_path": os.path.join(results_dir, "weights", "best.pt").replace("\\", "/"),
            "final_train_box_loss": float(last_row.get('train/box_loss', 0)),
            "final_train_cls_loss": float(last_row.get('train/cls_loss', 0)),
            "final_train_dfl_loss": float(last_row.get('train/dfl_loss', 0)),
            "best_val_precision": float(best_row.get('metrics/precision(B)', 0)),
            "best_val_recall": float(best_row.get('metrics/recall(B)', 0)),
            "best_val_mAP50": float(best_row.get('metrics/mAP50(B)', 0)),
            "best_val_mAP50_95": float(best_row.get('metrics/mAP50-95(B)', 0)),
            "gpu_vram_usage_gb": round(vram, 2),
            "cuda_oom_occurred": oom_occurred,
            "early_stopping_status": early_stopping,
            "results_csv_path": results_csv.replace("\\", "/"),
            "training_plots_dir": results_dir.replace("\\", "/")
        }
        
        with open(os.path.join(base_dir, "outputs", "full_training_report.json"), "w") as f:
            json.dump(report, f, indent=4)
            
        print(json.dumps(report, indent=4))
    else:
        print(f"Results file not found at {results_csv} or OOM occurred.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
