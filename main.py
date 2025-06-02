from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import json
import traceback
import sys
import os
import torch
import numpy as np
from PIL import Image
import pytz
from datetime import datetime

from waggle.plugin import Plugin
from waggle.data.vision import Camera

def get_chicago_time():
    chicago_tz = pytz.timezone('America/Chicago')
    return datetime.now(chicago_tz).isoformat()

def load_model_offline():
    try:
        print("Attempting to load RF-DETR model...", flush=True)
        model = RFDETRBase()
        print("RF-DETR model loaded successfully from cache", flush=True)
        return model
    except Exception as e:
        print(f"Failed to load model normally: {e}", flush=True)
        
        backup_path = "/app/models/rfdetr_backup.pt"
        if os.path.exists(backup_path):
            try:
                print(f"Loading model from backup: {backup_path}", flush=True)
                model = RFDETRBase()
                
                checkpoint = torch.load(backup_path, map_location='cpu')
                model.model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded successfully from backup", flush=True)
                return model
            except Exception as backup_e:
                print(f"Failed to load from backup: {backup_e}", flush=True)
        
        raise e

def detect_objects(image, model):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    detections = model.predict(image, threshold=0.5)
    
    detection_list = []
    for i in range(len(detections.class_id)):
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        bbox = detections.xyxy[i]
        
        detection_list.append({
            "class": COCO_CLASSES[class_id],
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        })
    
    return detection_list

def main():
    plugin = None
    start_time = get_chicago_time()
    
    try:
        plugin = Plugin()
        model = load_model_offline()

        with Camera("bottom_camera") as camera:
            snapshot = camera.snapshot()
        
        timestamp = snapshot.timestamp
    
        if isinstance(snapshot.data, np.ndarray):
            if len(snapshot.data.shape) == 3 and snapshot.data.shape[2] == 3:
                image_rgb = snapshot.data[:, :, ::-1]
            else:
                image_rgb = snapshot.data
        else:
            image_rgb = snapshot.data
        
        detections = detect_objects(image_rgb, model)
        
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_data = {
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections)
        }
        
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        
        snapshot.save("rfdetr_inference_image.jpg")
        plugin.upload_file("rfdetr_inference_image.jpg", timestamp=timestamp)
        
        finish_time = get_chicago_time()
        
        snapshot_dt = datetime.fromtimestamp(timestamp / 1e9, tz=pytz.UTC)
        chicago_snapshot_time = snapshot_dt.astimezone(pytz.timezone('America/Chicago')).isoformat()
        
        timing_data = {
            "plugin_start_time_chicago": start_time,
            "plugin_finish_time_chicago": finish_time,
            "image_timestamp_chicago": chicago_snapshot_time,
            "image_timestamp_ns": timestamp,
            "model_type": "RF-DETR"
        }
        
        plugin.publish("plugin.timing", json.dumps(timing_data), timestamp=timestamp)
        
    except Exception as e:
        try:
            error_timestamp = timestamp
        except NameError:
            import time
            error_timestamp = time.time_ns()

        error_data = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }

        plugin.publish("plugin.error", json.dumps(error_data), timestamp=error_timestamp)

        print(f"Error in plugin: {e}", file=sys.stderr)
        traceback.print_exc()

        raise
    
    sys.exit(0)

if __name__ == "__main__":
    main()