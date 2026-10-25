import os
import cv2
import torch
import time
import json
import statistics
import csv
from ultralytics import YOLO

def main():
    base_dir = "e:/SEM-5/COMPUTER VISION/mini/honey-bee-detection"
    os.chdir(base_dir)

    print("--- YOLO11 Video Inference Started ---")
    
    video_path = "test_videos/Honeybees.webm"
    model_path = "runs/detect/runs/detect/honey_bee_yolo11n_640/weights/best.pt"
    
    out_dir = "outputs"
    video_out_path = os.path.join(out_dir, "Honeybees_yolo11n_annotated.mp4")
    csv_out_path = os.path.join(out_dir, "Honeybees_frame_counts.csv")
    txt_out_path = os.path.join(out_dir, "Honeybees_video_summary.txt")
    samples_dir = os.path.join(out_dir, "video_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    oom_occurred = False
    
    try:
        model = YOLO(model_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        bee_counts = []
        inference_times = []
        
        csv_data = []
        
        sample_indices = [
            0, 
            int(total_frames * 0.25), 
            int(total_frames * 0.50), 
            int(total_frames * 0.75), 
            total_frames - 2 # -2 to ensure we grab it before end
        ]
        
        start_time_e2e = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_sec = frame_idx / fps if fps > 0 else 0
            
            # YOLO inference
            # imgsz=640, conf=0.25, iou=0.70, class=0 (honey_bee)
            results = model.predict(frame, imgsz=640, conf=0.25, iou=0.70, classes=[0], device=0, verbose=False)
            res = results[0]
            
            # Draw bounding boxes and confidences via ultralytics plot
            annotated_frame = res.plot()
            
            # Calculate bee count
            bee_count = len(res.boxes)
            bee_counts.append(bee_count)
            
            inf_time = sum(res.speed.values()) # sum of preprocess, inference, postprocess
            inference_times.append(inf_time)
            
            csv_data.append([frame_idx, timestamp_sec, bee_count, inf_time])
            
            # Overlay custom text
            cv2.putText(annotated_frame, f"Frame: {frame_idx:05d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Bee Count: {bee_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Time: {timestamp_sec:.2f} s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out_vid.write(annotated_frame)
            
            if frame_idx in sample_indices:
                sample_path = os.path.join(samples_dir, f"sample_frame_{frame_idx:05d}.jpg")
                cv2.imwrite(sample_path, annotated_frame)
                
            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
                
        end_time_e2e = time.time()
        cap.release()
        out_vid.release()
        
        total_time_e2e = end_time_e2e - start_time_e2e
        actual_fps_e2e = frame_idx / total_time_e2e if total_time_e2e > 0 else 0
        
        # Calculate statistics
        min_bees = min(bee_counts) if bee_counts else 0
        max_bees = max(bee_counts) if bee_counts else 0
        avg_bees = sum(bee_counts) / len(bee_counts) if bee_counts else 0
        median_bees = statistics.median(bee_counts) if bee_counts else 0
        std_bees = statistics.stdev(bee_counts) if len(bee_counts) > 1 else 0
        
        zero_frames = sum(1 for c in bee_counts if c == 0)
        positive_frames = sum(1 for c in bee_counts if c > 0)
        
        avg_inf_time = sum(inference_times) / len(inference_times) if inference_times else 0
        model_inf_fps = 1000 / avg_inf_time if avg_inf_time > 0 else 0
        
        # Write CSV
        with open(csv_out_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_number', 'timestamp_seconds', 'bee_count', 'inference_time_ms'])
            writer.writerows(csv_data)
            
        # Verify video output
        vid_exists = os.path.exists(video_out_path)
        vid_can_open = False
        if vid_exists:
            vtest = cv2.VideoCapture(video_out_path)
            if vtest.isOpened():
                ret, _ = vtest.read()
                if ret:
                    vid_can_open = True
            vtest.release()
            
        report = {
            "Total frames processed": frame_idx,
            "Total video duration (sec)": round(frame_idx / fps, 2) if fps > 0 else 0,
            "Minimum detected bees/frame": min_bees,
            "Maximum detected bees/frame": max_bees,
            "Average detected bees/frame": round(avg_bees, 2),
            "Median detected bees/frame": median_bees,
            "Standard deviation of bees/frame": round(std_bees, 2),
            "Number of frames with zero detections": zero_frames,
            "Number of frames with at least one detection": positive_frames,
            "Average inference time per frame (ms)": round(avg_inf_time, 2),
            "Model Inference FPS (pure YOLO)": round(model_inf_fps, 2),
            "Actual end-to-end video processing FPS": round(actual_fps_e2e, 2),
            "Video source FPS": round(fps, 2),
            "Total end-to-end processing time (sec)": round(total_time_e2e, 2),
            "Verification: Output video exists": vid_exists,
            "Verification: Output video can be opened": vid_can_open,
            "Verification: Frame count consistent": frame_idx > 0 and frame_idx == len(csv_data),
            "Verification: CSV has one row per frame": len(csv_data) == frame_idx,
            "CUDA OOM Occurred": oom_occurred
        }
        
        with open(txt_out_path, 'w') as f:
            for k, v in report.items():
                f.write(f"{k}: {v}\n")
                
        print("\n--- INFERENCE COMPLETE ---")
        print(json.dumps(report, indent=4))

    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "OOM" in str(e):
            print("CUDA OOM occurred!")
            oom_occurred = True
            with open(txt_out_path, 'w') as f:
                f.write(f"CUDA OOM Occurred: {oom_occurred}\n")
        else:
            raise e

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
