import pandas as pd
import requests
import time
import os
from tqdm import tqdm

def simulate_live_traffic():
    metadata_path = "data/metadata.csv"
    img_dir = "data/images"
    
    if not os.path.exists(metadata_path):
        print("Metadata not found. Wait for dataset extraction to complete.")
        return
        
    df = pd.read_csv(metadata_path, encoding='utf-8', skipinitialspace=True)
    # Strip any stray \r characters from all string columns (Windows CRLF CSVs)
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    # Exclude the first 200 items (reserved for TrOCR training) and limit to 50 items
    sim_df = df.iloc[0:800]
    
    print(f"Loaded {len(sim_df)} images for live MLOps traffic simulation.")
    print("Hitting FastAPI Server at http://localhost:8000/process-image")
    
    for idx, row in tqdm(sim_df.iterrows(), total=len(sim_df)):
        filename = row['filename'].strip()   # Strip \r\n from Windows-encoded CSVs
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            continue
            
        with open(img_path, 'rb') as f:
            files = {'file': (filename, f, 'image/jpeg')}
            # Send the ground truth label to facilitate Delta Tracking
            data = {'label': row['text']} 
            try:
                # 1. Post image to Celery queue via FastAPI
                response = requests.post("http://localhost:8000/process-image", files=files, data=data)
                if response.status_code != 200:
                    continue
                task_id = response.json().get("task_id")
                
                # 2. Poll for Status to trigger Prometheus logging block in FastAPI
                while True:
                    status_res = requests.get(f"http://localhost:8000/status/{task_id}")
                    if status_res.status_code == 200:
                        data = status_res.json()
                        if data.get("status") in ["SUCCESS", "FAILURE"]:
                            break
                    time.sleep(0.5)
                    
                # Pause 1 second between requests to visualize gradual events on Grafana
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"\\nSimulation Error on {filename}: {e}")

if __name__ == "__main__":
    simulate_live_traffic()
