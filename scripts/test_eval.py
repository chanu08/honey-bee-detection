import os
import torch
import json
from ultralytics import YOLO

def main():
    base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
    os.chdir(base_dir)

    print("--- YOLO11 Test Set Evaluation Started ---")
    
    oom_occurred = False
    report = {}
    
    try:
        model = YOLO("runs/detect/runs/detect/honey_bee_yolo11n_640/weights/best.pt")
        
        # We explicitly use split="test"
        metrics = model.val(
            data="dataset/data.yaml",
            split="test",
            imgsz=640,
            batch=13,
            device=0,
            project="runs/detect",
            name="honey_bee_test_eval",
            exist_ok=True
        )
        
        results_dir = os.path.join(base_dir, "runs", "detect", "honey_bee_test_eval")
        
        speed = metrics.speed
        
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(0) / (1024**3)
            if vram == 0:
                vram = torch.cuda.memory_allocated(0) / (1024**3)
                
        report = {
            "test_precision": float(metrics.results_dict.get('metrics/precision(B)', 0)),
            "test_recall": float(metrics.results_dict.get('metrics/recall(B)', 0)),
            "test_mAP50": float(metrics.results_dict.get('metrics/mAP50(B)', 0)),
            "test_mAP50_95": float(metrics.results_dict.get('metrics/mAP50-95(B)', 0)),
            "inference_speed_ms_per_image": {
                "preprocess": float(speed.get('preprocess', 0)),
                "inference": float(speed.get('inference', 0)),
                "postprocess": float(speed.get('postprocess', 0)),
                "total": float(sum(speed.values()))
            },
            "gpu_vram_usage_gb": round(vram, 2),
            "cuda_oom_occurred": oom_occurred,
            "plots_location": {
                "confusion_matrix": os.path.join(results_dir, "confusion_matrix.png").replace("\\", "/"),
                "pr_curve": os.path.join(results_dir, "PR_curve.png").replace("\\", "/"),
                "f1_curve": os.path.join(results_dir, "F1_curve.png").replace("\\", "/"),
                "p_curve": os.path.join(results_dir, "P_curve.png").replace("\\", "/"),
                "r_curve": os.path.join(results_dir, "R_curve.png").replace("\\", "/"),
                "visualizations_dir": results_dir.replace("\\", "/")
            }
        }
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "OOM" in str(e):
            oom_occurred = True
            report["cuda_oom_occurred"] = True
            print("CUDA OOM occurred!")
        else:
            raise e
            
    if not oom_occurred:
        with open("outputs/test_eval_report.json", "w") as f:
            json.dump(report, f, indent=4)
        print("\n--- TEST EVALUATION COMPLETE ---")
        print(json.dumps(report, indent=4))

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
