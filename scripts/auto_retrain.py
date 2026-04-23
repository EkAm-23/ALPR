import sys
import time
import json
import subprocess
import os
import datetime

# ──────────────────────────────────────────────────────────────────────
# MLOps Configuration
# ──────────────────────────────────────────────────────────────────────

# Directory name where the LoRA adapter weights are saved by the trainer.
# These files (adapter_model.safetensors + adapter_config.json) are tiny (~6MB)
# compared to the full model, and are injected into the containers at runtime.
LORA_ADAPTER_DIR  = "trocr_lora_adapters"

# The target path *inside* the containers where the adapters are mounted.
# This path matches the lora_weights Docker volume mount in docker-compose.yml.
CONTAINER_LORA_PATH = "/app/lora_adapters"

# File that tracks the current semantic version of the LoRA model.
# Format: vMAJOR.MINOR.PATCH  (e.g. v1.0.3)
# MAJOR = base model generation, MINOR = LoRA architecture revision,
# PATCH = auto-incremented per CT cycle.
MODEL_VERSION_FILE = "model_version.txt"

# File that counts total continuous-training cycles completed.
CT_CYCLE_FILE = "data/.ct_cycle_count"

# Training history log — one JSON record per CT cycle.
# Stored at DEP/logs/training_history.jsonl (JSON Lines format)
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "training_history.jsonl"
)


# ──────────────────────────────────────────────────────────────────────
# Version Management Helpers
# ──────────────────────────────────────────────────────────────────────

def read_model_version(backend_dir):
    """Read the current model version string (e.g. 'v1.0.0')."""
    path = os.path.join(backend_dir, MODEL_VERSION_FILE)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "v1.0.0"


def bump_patch_version(version):
    """Increment the patch component: 'v1.0.3' → 'v1.0.4'."""
    parts = version.lstrip("v").split(".")
    parts[2] = str(int(parts[2]) + 1)
    return "v" + ".".join(parts)


def write_model_version(backend_dir, version):
    """Persist the new version string to model_version.txt."""
    path = os.path.join(backend_dir, MODEL_VERSION_FILE)
    with open(path, "w") as f:
        f.write(version + "\n")


def get_ct_cycle(backend_dir):
    """Return the number of CT cycles completed so far (0 if first run)."""
    path = os.path.join(backend_dir, CT_CYCLE_FILE)
    if os.path.exists(path):
        with open(path, "r") as f:
            return int(f.read().strip())
    return 0


def increment_ct_cycle(backend_dir):
    """Increment the CT cycle counter and return the new value."""
    cycle = get_ct_cycle(backend_dir) + 1
    path = os.path.join(backend_dir, CT_CYCLE_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(cycle))
    return cycle


def log_training_event(version, ct_cycle, start_time, end_time):
    """Append a single training record to logs/training_history.jsonl."""
    duration_secs = (end_time - start_time).total_seconds()
    record = {
        "ct_cycle":      ct_cycle,
        "model_version": version,
        "start_time":    start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time":      end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_mins": round(duration_secs / 60, 2),
        "duration_secs": round(duration_secs, 1),
    }
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  📋 Training log updated: {LOG_FILE}")


# ──────────────────────────────────────────────────────────────────────
# Delta Queue Monitor
# ──────────────────────────────────────────────────────────────────────

def check_delta_batch_size():
    """Checks the number of explicitly labeled images waiting in the active learning queue."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(backend_dir, "data/delta_metadata.csv")

    if not os.path.exists(csv_path):
        return 0

    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) <= 1:
            return 0
        return len(data) - 1  # Subtract 1 for the header


# ──────────────────────────────────────────────────────────────────────
# Continuous Training Pipeline
# ──────────────────────────────────────────────────────────────────────

def trigger_retraining():
    """
    Kicks off the LoRA fine-tuning pipeline, versions the result with DVC,
    then hot-swaps the adapter weights into the running Docker containers.

    Flow:
      Step 1: Run train_trocr_lora.py  →  produces trocr_lora_adapters/
      Step 2: DVC add + git commit     →  versions new weights, bumps model version
      Step 3: Inject adapters          →  docker cp into worker + backend containers
      Step 4: Restart containers       →  ocr_engine.py merges new adapters on load
      Step 5: Purge delta queue        →  resets the CSV and delta_batch/ directory
    """
    print(f"\n[{datetime.datetime.now()}] 🚨 DATA DRIFT DETECTED!")
    print("Initiating LoRA Continuous Training (CT) Pipeline...")

    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_dir = os.path.join(backend_dir, "ALPR-app")  # Where docker-compose.yml lives
        delta_csv      = "data/delta_metadata.csv"
        delta_csv_path = os.path.join(backend_dir, delta_csv)

        if not os.path.exists(delta_csv_path):
            print("No Delta Batch CSV found yet! Waiting for more data...")
            return

        # ── Step 1/5: LoRA fine-tuning ────────────────────────────────
        # Saves lightweight adapter weights only — NOT the full model.
        # Outputs: trocr_lora_adapters/adapter_model.safetensors
        #          trocr_lora_adapters/adapter_config.json
        print("\n[Step 1/5] Running LoRA fine-tuning...")
        training_start = datetime.datetime.now()
        # Use sys.executable so we call the same Python interpreter (conda env)
        # that is running this daemon — not the system `python3`.
        subprocess.run(
            [
                sys.executable, "training/train_trocr_lora.py",
                "--dataset",    delta_csv,
                "--img_dir",    "data/delta_batch",
                "--output",     LORA_ADAPTER_DIR,
                "--epochs",     "5",
                "--lr",         "2e-5",   # Low LR preserves base model knowledge
                "--lora_r",     "4",      # Smaller rank = fewer params = less overfitting
                "--lora_alpha", "8",      # Alpha ~ 2x rank
            ],
            cwd=backend_dir,
            check=True
        )
        training_end = datetime.datetime.now()
        duration_min = round((training_end - training_start).total_seconds() / 60, 2)
        print(f"  ✅ LoRA training complete. Duration: {duration_min} min")

        # ── Step 2/5: DVC versioning + git commit ─────────────────────
        # Track the new adapter weights with DVC and commit the pointer
        # file to git so every CT cycle is reproducible and reversible.
        ct_cycle       = increment_ct_cycle(backend_dir)
        old_version    = read_model_version(backend_dir)
        new_version    = bump_patch_version(old_version)
        write_model_version(backend_dir, new_version)

        # Log the training event now that we know the final version and duration
        log_training_event(new_version, ct_cycle, training_start, training_end)

        print(f"\n[Step 2/5] Versioning weights with DVC "
              f"({old_version} → {new_version}, CT Cycle #{ct_cycle})...")

        # Tell DVC to hash the new adapter files and update the .dvc pointer
        subprocess.run(["dvc", "add", LORA_ADAPTER_DIR], cwd=backend_dir, check=True)

        # Commit the updated .dvc pointer + version file + training log to git
        commit_msg = f"model: TrOCR LoRA {new_version} - CT Cycle #{ct_cycle}"
        subprocess.run(
            ["git", "add", f"{LORA_ADAPTER_DIR}.dvc", MODEL_VERSION_FILE, LOG_FILE],
            cwd=backend_dir, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=backend_dir, check=True
        )
        print(f"  ✅ DVC tracking committed: '{commit_msg}'")

        # ── Step 3/5: Inject adapter weights into containers ──────────
        # We copy only the tiny adapter directory via the Docker socket.
        # This avoids a full `docker build` and any VirtioFS deadlocks.
        adapter_local_path = os.path.join(backend_dir, LORA_ADAPTER_DIR)
        print(f"\n[Step 3/5] Copying LoRA adapters → containers ({CONTAINER_LORA_PATH})...")
        # Trailing /. on the source copies the *contents* of the directory
        # directly into CONTAINER_LORA_PATH, avoiding a nested sub-directory.
        subprocess.run(
            ["docker", "cp", adapter_local_path + "/.", f"alpr-app-worker-1:{CONTAINER_LORA_PATH}"],
            check=True
        )
        subprocess.run(
            ["docker", "cp", adapter_local_path + "/.", f"alpr-app-backend-1:{CONTAINER_LORA_PATH}"],
            check=True
        )
        print("  ✅ Adapter weights injected.")

        # ── Step 4/5: Restart containers to reload ocr_engine.py ──────
        # docker compose must run from the directory containing docker-compose.yml
        print("\n[Step 4/5] Restarting worker and backend containers...")
        subprocess.run(
            ["docker", "compose", "restart", "worker", "backend"],
            check=True, cwd=compose_dir
        )
        print("  ✅ Containers restarted with new LoRA weights.")

        # ── Step 5/5: Purge the delta batch to reset the queue ─────────
        print("\n[Step 5/5] Purging delta batch queue...")
        os.remove(delta_csv_path)
        import shutil
        delta_batch_dir = os.path.join(backend_dir, "data/delta_batch")
        if os.path.exists(delta_batch_dir):
            shutil.rmtree(delta_batch_dir)
            os.makedirs(delta_batch_dir, exist_ok=True)
        print("  ✅ Delta memory purged.")

        print(f"\n✅ LoRA MLOps loop closed. Model is now {new_version}\n")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed during execution. Error: {e}")


if __name__ == "__main__":
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    THRESHOLD = 100

    print("================================================")
    print("  MLOps LoRA Continuous Training Daemon        ")
    print("================================================")
    print(f"  Model Version : {read_model_version(backend_dir)}")
    print(f"  CT Cycles done: {get_ct_cycle(backend_dir)}")
    print(f"  Drift Threshold: {THRESHOLD} images")
    print("Monitoring Delta Batch Queue Directory...\n")

    while True:
        queue_size = check_delta_batch_size()
        print(f"[{datetime.datetime.now().time()}] Delta Queue: {queue_size}/{THRESHOLD} images")

        if queue_size >= THRESHOLD:
            trigger_retraining()
            print("Sleeping for 10 minutes (cooldown) before monitoring again...")
            time.sleep(600)  # Cooldown period so we don't spam hardware on rapid drift
        else:
            time.sleep(30)  # Poll every 30 seconds when below threshold
