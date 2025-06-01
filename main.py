from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import json
import traceback
import sys
import os
import torch
import numpy as np
from PIL import Image

from waggle.plugin import Plugin
from waggle.data.vision import Camera

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
                
                model = torch.load(backup_path, map_location='cpu')
                print("Model loaded successfully from backup (full model)", flush=True)
                return model
                
            except Exception as backup_e:
                print(f"Failed to load from backup: {backup_e}", flush=True)
                raise backup_e
        
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
            "confidence": float(confidence),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        })
    
    return detection_list

def main():
    plugin = None
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
        print(detections)
        class_counts = {}
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detection_data = {
            "detections": detections,
            "counts": class_counts,
            "total_objects": len(detections)
        }

        print(f"Publishing detection data: {len(detections)} objects detected", flush=True)
        print(f"Class counts: {class_counts}", flush=True)
        
        plugin.publish("object.detections", json.dumps(detection_data), timestamp=timestamp)
        print("Data published successfully!", flush=True)
        
        plugin.publish("test.message", "RF-DETR plugin is running", timestamp=timestamp)
        print("Test message published", flush=True)

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

if __name__ == "__main__":
    main()
