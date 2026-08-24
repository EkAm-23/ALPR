# 🚗 ALPR — Automatic License Plate Recognition

> **An end-to-end MLOps system for Indian license plate recognition**, featuring adaptive image restoration, deep-learning OCR with LoRA fine-tuning, a self-healing continuous training loop, and a fully monitored microservice deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [ML Pipeline](#ml-pipeline)
   - [Stage 1 — Image Restoration (Adaptive Routing)](#stage-1--image-restoration-adaptive-routing)
   - [Stage 2 — License Plate Detection](#stage-2--license-plate-detection)
   - [Stage 3 — OCR with TTA](#stage-3--ocr-with-tta)
4. [Models Used](#models-used)
5. [MLOps — Continuous Training Loop](#mlops--continuous-training-loop)
6. [Monitoring & Observability](#monitoring--observability)
7. [Services & Ports](#services--ports)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Project Structure](#project-structure)
10. [Running Locally](#running-locally)

---

## Overview

This project was built as a third-year engineering design project. It goes beyond a standard "run a model and show output" approach — the system is **production-architecture grade**, designed around real MLOps principles:

- **Adaptive pre-processing**: incoming images are mathematically classified and routed to the optimal restoration model (DarkIR for night, DeHaze, DeRain, or NAFNet).
- **Deep OCR with LoRA**: Microsoft TrOCR fine-tuned with LoRA adapters trained on domain-specific Indian plate data.
- **Active learning**: low-confidence predictions are automatically queued for re-training.
- **Continuous training daemon**: when the delta queue reaches a threshold, the system autonomously re-trains, versions, and hot-swaps the new model weights into the live containers — **without downtime**.
- **Full observability**: Prometheus scrapes custom ML metrics (OCR confidence, data drift score, inference latency, queue sizes); Grafana visualises them.

---

## System Architecture

```
                         ┌─────────────────────────────────┐
                         │          React Frontend          │
                         │       (Vite + TypeScript)        │
                         └──────────────┬──────────────────┘
                                        │ HTTP (port 8080)
                                   ┌────▼────┐
                                   │  Nginx  │  ← Reverse Proxy
                                   └────┬────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │     FastAPI  (port 8000)      │
                         │    "ALPR API Gateway"         │
                         │  /process-image  POST         │
                         │  /status/{id}    GET          │
                         │  /vehicle/{plate} GET         │
                         │  /metrics        GET          │
                         └──────────┬───────────────────┘
                                    │ Celery task dispatch
                              ┌─────▼──────┐
                              │   Redis    │  ← Message Broker + Result Backend
                              └─────┬──────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │     Celery Worker (×4 processes)  │
                    │                                   │
                    │  1. Classify → Route image        │
                    │  2. Restore  → NAFNet/DarkIR/etc. │
                    │  3. Detect   → YOLOv8             │
                    │  4. TTA OCR  → TrOCR + LoRA       │
                    │  5. Active Learning routing        │
                    └───────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │      MLOps Daemon  (host process)    │
                    │         scripts/auto_retrain.py      │
                    │                                      │
                    │  Polls delta queue every 30s         │
                    │  On threshold → LoRA re-train        │
                    │              → DVC version           │
                    │              → docker cp hot-inject  │
                    │              → container restart     │
                    └──────────────────────────────────────┘

         ┌──────────────────────┐    ┌─────────────────────┐
         │  Prometheus (9090)   │ ←  │ FastAPI /metrics     │
         └──────────┬───────────┘    └─────────────────────┘
                    │
         ┌──────────▼───────────┐
         │   Grafana (3000)     │  ← Dashboards & Alerts
         └──────────────────────┘
```

---

## ML Pipeline

Every image submitted to the API goes through a 3-stage ML pipeline, executed asynchronously by the Celery worker.

### Stage 1 — Image Restoration (Adaptive Routing)

The system **does not blindly apply one restoration model**. Instead, it mathematically analyses each image and routes it:

| Condition Detected | Routing Decision | Model Used |
|---|---|---|
| Mean brightness < 65 | Dark / night scene | **DarkIR** |
| High dark channel score + low saturation + low contrast | Haze / fog | **DeHaze** |
| Low Laplacian variance + low contrast | Rain streaks / blur | **DeRain** |
| Normal / clear image | Standard deblur | **NAFNet** |

The classifier (`modules/classifier.py`) uses **Dark Channel Prior (DCP)**, Laplacian variance, and HSV saturation — all computed with OpenCV, with zero inference overhead.

After restoration, **NAFNet is always applied to the extracted plate crop** as a super-resolution / deblurring pass before OCR.

### Stage 2 — License Plate Detection

A fine-tuned **YOLOv8** model (`yolov8_plate.pt`) detects bounding boxes of license plates in the restored image. Multiple plates are handled — the one with the **highest confidence score** is selected. A red bounding box is drawn on the annotated output image.

### Stage 3 — OCR with TTA

**Test-Time Augmentation (TTA)** is applied to improve OCR robustness:

1. The detected plate is cropped with a 5% margin.
2. The crop is run through **4 different white-padding scales** (0%, 5%, 10%, 15%).
3. Each padded crop is deblurred by NAFNet, then passed through **Microsoft TrOCR** (fine-tuned with LoRA adapters if available).
4. The 4 predictions are **majority-voted** — the most common text wins; the highest-confidence run of that text is chosen as the final result.

Post-processing applies **strict Indian plate formatting** (2-2-2-4 character block: `MH 12 AB 1234`), with character-level correction of common OCR confusions (e.g., `0↔O`, `1↔I`, `5↔S`).

Finally, the **RTO metadata parser** extracts the state and district from the plate prefix.

---

## Models Used

| Model | Role | Source |
|---|---|---|
| **YOLOv8** (custom fine-tuned) | License plate detection | Ultralytics — fine-tuned on Indian plate dataset |
| **Microsoft TrOCR** (`trocr-base-printed`) | OCR — plate text extraction | Hugging Face `microsoft/trocr-base-printed` |
| **LoRA Adapters** (PEFT) | Domain-specific TrOCR fine-tuning | Trained locally via `training/train_trocr_lora.py` |
| **NAFNet** | Image deblurring / sharpening | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) |
| **DarkIR** | Low-light image enhancement | [DarkIR paper](https://arxiv.org/abs/2412.13507) |
| **DeHaze** | Haze / fog removal | Classical DCP + DNN approach |
| **DeRain** | Rain streak removal | CNN-based deraining |

> **Model weights are tracked with DVC** and are not stored in git. See the `.dvc` pointer files.

---

## MLOps — Continuous Training Loop

This is the core MLOps contribution of the project. The system implements a **fully automated, closed-loop continuous training pipeline**.

### How it works

```
  User submits image with optional ground-truth label
                   │
                   ▼
         Celery processes image
                   │
           OCR confidence < 0.50?
                   │
         ┌─────────┴──────────┐
        YES                   NO
         │                    │  → Normal response
         │
   Ground truth label provided?
         │
   ┌─────┴──────────┐
  YES               NO
   │                │
   ▼                ▼
Delta Batch      Quarantine
(auto-training)  (needs human review)
   │
   ▼
Delta queue size ≥ 100 images?
   │
   ▼  (auto_retrain.py daemon polls every 30s)
  YES → Trigger CT Pipeline:
         Step 1: LoRA fine-tune  (train_trocr_lora.py)
         Step 2: DVC version     (trocr_lora_adapters.dvc)
         Step 3: docker cp       (hot-inject weights into containers)
         Step 4: docker restart  (containers reload with new weights)
         Step 5: Purge delta queue
```

### LoRA Fine-Tuning Details

- **Base model**: `microsoft/trocr-base-printed` (frozen)
- **Trainable parameters**: Only the low-rank adapter matrices (rank `r=4`, alpha `α=8`)
- **Adapter size**: ~6 MB (vs. ~400 MB full model) — fits easily in a `docker cp`
- **Hot-injection**: `ocr_engine.py` merges adapters via `PeftModel.merge_and_unload()` on container restart — zero latency overhead at inference
- **Versioning**: Every CT cycle bumps `model_version.txt` (e.g. `v1.0.0 → v1.0.1`) and commits a DVC pointer to git — full reproducibility

### Training History

Every CT cycle is logged to `logs/training_history.jsonl`:
```json
{
  "ct_cycle": 3,
  "model_version": "v1.0.3",
  "start_time": "2026-08-20 14:32:01",
  "end_time": "2026-08-20 14:46:18",
  "duration_mins": 14.28
}
```

---

## Monitoring & Observability

The FastAPI backend exposes a `/metrics` endpoint scraped by Prometheus. Custom metrics are emitted after every inference:

| Metric | Type | Description |
|---|---|---|
| `alpr_ocr_confidence` | Gauge | Raw TrOCR confidence score |
| `alpr_ocr_length` | Histogram | Length of predicted plate string |
| `alpr_image_blur_score` | Gauge | Laplacian variance — tracks camera focus degradation |
| `alpr_image_brightness` | Gauge | Mean pixel intensity — tracks day/night shift |
| `alpr_data_drift_score` | Gauge | Compound drift proxy (blur + brightness + confidence) |
| `alpr_inference_latency_seconds` | Histogram | Full PyTorch pipeline latency |
| `alpr_bbox_count` | Histogram | Plates detected per frame |
| `alpr_exact_match` | Gauge | Exact match vs. ground truth (simulated mode) |
| `alpr_character_accuracy` | Gauge | Character-level accuracy vs. ground truth |
| `alpr_edit_distance` | Histogram | Levenshtein distance vs. ground truth |
| `alpr_invalid_format_total` | Counter | Predictions violating the `XX-00-XX-0000` plate format |
| `alpr_delta_queue_size` | Gauge | Images waiting for CT — triggers re-training |
| `alpr_quarantine_queue_size` | Gauge | Images quarantined for human review |

All metrics are visualised in **Grafana** at `http://localhost:3000`.

---

## Services & Ports

| Service | Port | Description |
|---|---|---|
| Frontend (Nginx) | `8080` | React UI served through Nginx reverse proxy |
| Backend (FastAPI) | `8000` | REST API + Prometheus metrics |
| Prometheus | `9090` | Metrics scraper |
| Grafana | `3000` | Monitoring dashboards |
| Redis | `6379` | Internal only (message broker + result backend) |

---

## CI/CD Pipeline

The repository uses **GitHub Actions** with a two-job pipeline (`.github/workflows/cicd.yml`).

### Job 1 — Lint & Unit Tests (GitHub-hosted runner)
- Triggers on every push and pull request to `main`
- Runs on a clean `ubuntu-latest` environment
- Installs lightweight dependencies (ML libs are mocked in tests)
- Executes `pytest tests/ -v`

### Job 2 — Deploy to Local Stack (self-hosted runner)
- Triggers **only on direct pushes to `main`** — never on PRs
- Executes on a self-hosted runner (local Mac)
- Runs `docker compose up -d --build`
- Health-checks that `backend` and `worker` containers are `Up` before marking success

> **Security note**: The deploy job's `if: github.event_name == 'push'` guard ensures fork pull requests from external contributors can never trigger code execution on the self-hosted runner.

---

## Project Structure

```
DEP/
├── ALPR-app/
│   ├── backend/
│   │   ├── main.py              # FastAPI app — endpoints + Prometheus metrics
│   │   ├── tasks.py             # Celery task — full ML inference pipeline
│   │   ├── modules/
│   │   │   ├── classifier.py    # Adaptive image routing (DCP + brightness)
│   │   │   ├── nafnet.py        # NAFNet deblur/sharpen wrapper
│   │   │   ├── dark_ir.py       # DarkIR low-light enhancement
│   │   │   ├── dehaze.py        # Dehazing module
│   │   │   ├── derain.py        # Deraining module
│   │   │   ├── detector.py      # YOLOv8 plate detection
│   │   │   ├── ocr_engine.py    # TrOCR + LoRA adapter injection + TTA
│   │   │   ├── rto_metadata.py  # State/district parser from plate text
│   │   │   ├── vehicle_lookup.py# Mock RTO vehicle registration lookup
│   │   │   └── mlops_metrics.py # Exact match, char accuracy, edit distance
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/                # React + TypeScript (Vite)
│   ├── nginx/                   # Nginx reverse proxy config
│   ├── docker-compose.yml       # Full stack definition (7 services)
│   └── prometheus.yml           # Prometheus scrape config
│
├── training/
│   └── train_trocr_lora.py      # LoRA fine-tuning script (PEFT)
│
├── scripts/
│   ├── auto_retrain.py          # MLOps CT daemon — polls queue, triggers training
│   ├── prepare_dataset.py       # Dataset preparation utilities
│   └── simulate_traffic.py      # Load simulation for Prometheus/Grafana testing
│
├── tests/                       # Pytest unit tests (CI-compatible, mocked ML)
├── logs/
│   └── training_history.jsonl   # Append-only CT cycle log
│
├── trocr_lora_adapters.dvc      # DVC pointer to versioned LoRA weights
├── model_version.txt            # Current model version (e.g. v1.0.3)
├── .github/workflows/cicd.yml   # GitHub Actions CI/CD
└── .dvc/                        # DVC configuration
```

---

## Running Locally

### Prerequisites

- Docker Desktop
- Python 3.10+ with a virtual environment
- DVC (`pip install dvc`)

### 1. Pull model weights (DVC)

```bash
dvc pull
```

> This fetches the YOLOv8 weights and LoRA adapter files tracked by DVC.

### 2. Start the full stack

```bash
cd ALPR-app
docker compose up -d --build
```

### 3. Access the app

| URL | Service |
|---|---|
| http://localhost:8080 | Frontend UI |
| http://localhost:8000/docs | FastAPI Swagger docs |
| http://localhost:9090 | Prometheus |
| http://localhost:3000 | Grafana (admin / admin) |

### 4. Start the MLOps Continuous Training Daemon

```bash
# In a separate terminal, from the repo root
python scripts/auto_retrain.py
```

The daemon polls the delta queue every 30 seconds and automatically triggers the full LoRA re-training → versioning → hot-injection pipeline when 100 labeled low-confidence images accumulate.

### 5. Simulate traffic (optional)

```bash
python scripts/simulate_traffic.py
```

Sends synthetic labeled requests to the API to rapidly populate the delta queue and trigger a CT cycle for demonstration.

---

## Active Learning Flow

When submitting an image via the UI, an **optional ground-truth label field** is available. This enables the active learning loop:

- **With label + low confidence** → routed to `data/delta_batch/` for automated CT
- **No label + low confidence** → quarantined in `data/needs_review/` for human inspection
- **High confidence** → inference only, no data stored

This design ensures that unsupervised production traffic never silently corrupts the training set.
