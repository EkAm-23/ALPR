import os
import io
import base64
from PIL import Image, ImageDraw
import numpy as np
from celery import Celery

# Import our custom ML modules for restoration, detection, text extraction, and metadata parsing
from modules.restoration import prepare_image_for_detection
from modules.detector import detect_license_plates
from modules.ocr_engine import extract_text
from modules.nafnet import process_with_nafnet
from modules.rto_metadata import parse_rto_metadata
from modules.dark_ir import process_with_darkir
from modules.dehaze import process_with_dehaze
from modules.derain import process_with_derain
from modules.classifier import classify_restoration_module

# Get Redis connection URL from environment variable, default to localhost:6379 for development
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize the Celery application, setting both broker and backend to Redis
celery_app = Celery(
    "tasks",
    broker=redis_url,
    backend=redis_url
)
# Configure Celery results to expire after 1 hour (3600 seconds) to free up Redis memory
celery_app.conf.result_expires = 3600

# Helper function to convert a PIL Image or Numpy array into a Base64-encoded Data URL
def img_to_b64(img):
    if img is None:
        return None
    # Convert numpy array to PIL Image if necessary
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    # Create an in-memory byte buffer
    buffered = io.BytesIO()
    # Ensure image is in RGB mode before saving as JPEG
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Save the image to the buffer in JPEG format
    img.save(buffered, format="JPEG")
    # Return the Base64 string formatted as a Data URL for easy frontend rendering
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

# Define the Celery background task for Auto-Adaptive License Plate Recognition
@celery_app.task
def process_alpr_task(file_path, true_label=None):
    try:
        # Load the uploaded original image from disk
        original_image = Image.open(file_path)
        # Explicitly force Pillow to load the image data into memory before conversion.
        # This prevents the "AttributeError: 'PngImageFile' object has no attribute '_im'" error.
        original_image.load()
        original_image = original_image.convert('RGB')
        
        # Track pipeline telemetry & drift
        import time
        import cv2
        from modules.mlops_metrics import calculate_mlops_scores
        start_inference_time = time.time()
        
        try:
            cv_img = cv2.imread(file_path)
            optimal_module = classify_restoration_module(cv_img)
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
            brightness = float(gray_img.mean())
        except Exception:
            optimal_module = 'normal'
            blur_score = 0.0
            brightness = -1.0

        # --- ML Logic Start ---
        original_image = process_with_nafnet(original_image)
        # Only apply the selected restoration module
        nafnet_image = None
        darkir_image = None
        dehaze_image = None
        derain_image = None

        if optimal_module == 'darkir':
            detection_image = process_with_darkir(original_image)
            darkir_image = detection_image
            detection_source = "Routed to DarkIR Engine"
        elif optimal_module == 'dehaze':
            detection_image = process_with_dehaze(original_image)
            dehaze_image = detection_image
            detection_source = "Routed to DeHaze Engine"
        elif optimal_module == 'derain':
            detection_image = process_with_derain(original_image)
            derain_image = detection_image
            detection_source = "Routed to DeRain Engine"
        else:
            detection_image = original_image
            nafnet_image = detection_image
            detection_source = "Routed to NAFNet Engine"

        restoration_msg = f"Mathematically Classified & Routed to: {optimal_module}"
            
        # 2. License Plate Detection
        boxes = detect_license_plates(detection_image)
        bbox_count = len(boxes)
        
        # Variables to store the final annotated image and the bounding box of the best plate
        annotated_img = None
        best_box = None
        
        # If at least one plate was detected
        if len(boxes) > 0:
            # Create a copy of the final detection image for drawing bounding boxes
            annotated_img = detection_image.copy()
            draw = ImageDraw.Draw(annotated_img)
            # Select the bounding box with the highest confidence score
            best_box = max(boxes, key=lambda x: x['score'])
            px1, py1, px2, py2 = best_box['box']
            # Draw a thick red rectangle around the detected plate
            draw.rectangle([px1, py1, px2, py2], outline="red", width=3)
        
        # Initialize default OCR variables in case detection or OCR fails completely
        text = "Failed OCR"
        conf = 0.0
        plate_crop_padded = None
        sr_plate = None
        rto_meta = {"state": "Unknown", "district_code": "Unknown"}
        
        # If we found a plate, proceed to crop and perform OCR
        if best_box:
            # Get the dimensions of the detection image
            h, w = np.array(detection_image).shape[:2]
            # Calculate a 5% margin to capture a slightly larger context area around the plate
            y_margin = int((py2 - py1) * 0.05)
            x_margin = int((px2 - px1) * 0.05)
            
            # Ensure the cropped coordinates don't go out of bounds of the image
            new_py1 = max(0, py1 - y_margin)
            new_py2 = min(h, py2 + y_margin)
            new_px1 = max(0, px1 - x_margin)
            new_px2 = min(w, px2 + x_margin)

            # Crop out the raw license plate region using Numpy indexing
            plate_crop_raw = np.array(detection_image)[new_py1:new_py2, new_px1:new_px2]
            
            # Test-Time Augmentation (TTA) multi-scale padding arrays
            padding_scales = [0.0, 0.05, 0.10, 0.15]
            predictions = []

            for pad_scale in padding_scales:
                # Calculate padding iteratively based on the crop's height and width
                pad_y = int(plate_crop_raw.shape[0] * pad_scale)
                pad_x = int(plate_crop_raw.shape[1] * pad_scale)
                new_h = plate_crop_raw.shape[0] + 2 * pad_y
                new_w = plate_crop_raw.shape[1] + 2 * pad_x
                # Skip overly small crops
                if new_h < 20 or new_w < 20:
                    continue
                    
                # Apply constant white padding (255) around the license plate
                padded = np.pad(plate_crop_raw, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=255)
                try:
                    # Run the padded crop through NAFNet to deblur and super-resolve the image
                    deblurred = process_with_nafnet(plate_crop_raw)
                    sr_result = deblurred 
                    
                    # Convert to RGB just in case
                    sr_result = sr_result.convert("RGB")
                    
                    # Extract the plate text and confidence score via EasyOCR
                    extracted_txt, extracted_conf = extract_text(sr_result)
                    # Aggregate the prediction tuple
                    predictions.append((extracted_txt, extracted_conf, sr_result, padded))
                except Exception:
                    # Ignore failures on individual augmented crops
                    continue
            
            from collections import Counter
            # Select the best OCR result using majority voting
            if len(predictions) > 0:
                # Extract all text strings from the predictions
                texts = [p[0] for p in predictions]
                # Find the most common text string (voting)
                majority_text = Counter(texts).most_common(1)[0][0]
                # Filter out the runs that generated the majority text, then pick the one with highest confidence
                best_run = max([p for p in predictions if p[0] == majority_text], key=lambda x: x[1])
                # Unpack the best result into local variables
                text, conf, sr_plate, plate_crop_padded = best_run
            else:
                # Fallback if all OCR augmentations failed
                text, conf = "Failed TTA OCR", 0.0
                sr_plate = Image.fromarray(plate_crop_raw)
                plate_crop_padded = plate_crop_raw
            
            # Lastly, parse the text string to infer State and District info
            rto_meta = parse_rto_metadata(text)
            
        # -------------------------------------------------------------
        # ACTIVE LEARNING DATA ROUTING (DELTA BATCH VS QUARANTINE)
        # -------------------------------------------------------------
        if conf > 0.0 and conf < 0.50:
            import shutil
            import csv
            base_filename = os.path.basename(file_path)
            if true_label is not None:
                # Store specifically for Automated MLOps Delta Training
                os.makedirs("data/delta_batch", exist_ok=True)
                dest = os.path.join("data/delta_batch", base_filename)
                shutil.copy(file_path, dest)
                # Append to delta tracking metadata (add header if missing)
                csv_path = "data/delta_metadata.csv"
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(["filename", "text"])
                    writer.writerow([base_filename, true_label])
            else:
                # Quaratine for Human Review (Unsupervised website traffic)
                os.makedirs("data/needs_review", exist_ok=True)
                dest = os.path.join("data/needs_review", base_filename)
                shutil.copy(file_path, dest)
        
        import re
        invalid_format = True
        if text != "Failed OCR" and text != "Failed TTA OCR":
            # Simple format heuristic for Indian Plates (Must end in roughly 4 digits and start with chars)
            if re.search(r'[A-Za-z]+.*[0-9]{3,4}$', text):
                invalid_format = False
                
        perf_scores = calculate_mlops_scores(text, true_label)
        
        # Proxy drift score: Penalize high blur, low brightness, low confidence
        if blur_score > 0 and brightness > 0:
            drift_score = 100.0 / max(1.0, blur_score) + abs(brightness - 128) + ((1.0 - conf) * 100)
        else:
            drift_score = 0.0
            
        result = {
            "restoration_msg": restoration_msg,
            "detection_source": detection_source,
            "original_image": img_to_b64(original_image),
            "nafnet_image": img_to_b64(nafnet_image) if nafnet_image is not None else None,
            "darkir_image": img_to_b64(darkir_image) if darkir_image is not None else None,
            "dehaze_image": img_to_b64(dehaze_image) if dehaze_image is not None else None,
            "derain_image": img_to_b64(derain_image) if derain_image is not None else None,
            "detection_used": img_to_b64(detection_image),
            "annotated_image": img_to_b64(annotated_img),
            "plate_crop": img_to_b64(plate_crop_padded),
            "plate_upscaled": img_to_b64(sr_plate),
            "extracted_text": text,
            "confidence": float(conf),
            "rto_metadata": rto_meta,
            "blur_score": blur_score,
            "brightness": brightness,
            "invalid_format": invalid_format,
            "drift_score": drift_score,
            "performance_scores": perf_scores,
            "bbox_count": bbox_count,
            "inference_time": float(time.time() - start_inference_time),
            "restoration_used": optimal_module
        }
        # Return the final task payload, which is stored in Redis for the API layer to fetch
        return result
    finally:
        # Garbage collection & Cleanup: ensure the temporary file is deleted from disk to save space
        if os.path.exists(file_path):
            os.remove(file_path)
