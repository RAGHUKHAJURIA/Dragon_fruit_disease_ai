# Dragon Fruit Disease Detection & Advisory System

A full-stack AI-powered web application for automated **dragon fruit disease diagnosis**, **lesion detection**, and **quality grading** using deep learning. The system integrates three AI pipelines — **ConViTX classification**, **YOLOv8 object detection**, and **ResNet50 quality grading** — with **Grad-CAM XAI heatmaps**, a **guided live camera scanner**, and **evidence-based treatment recommendations** grounded in 2024–2025 scientific literature.

---

## Table of Contents

- [Overview](#overview)
- [System Pipeline](#system-pipeline)
- [Disease Classes](#disease-classes)
- [Model Architectures](#model-architectures)
- [Classification Results](#classification-results)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [YOLOv8 Lesion Detection](#yolov8-lesion-detection)
- [Quality Grading](#quality-grading)
- [Live Camera — Guided Field Scanner](#live-camera--guided-field-scanner)
- [Digital Prescription Hub](#digital-prescription-hub)
- [Full-Stack Architecture](#full-stack-architecture)
- [Project Structure](#project-structure)
- [Flask Routes & API](#flask-routes--api)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Technology Stack](#technology-stack)
- [Scientific References](#scientific-references)

---

## Overview

Dragon fruit (*Hylocereus* spp.) is susceptible to several devastating diseases that cause significant economic losses. Early and accurate detection is critical for effective management. This project presents an end-to-end system that:

1. **Classifies** 6 disease classes from plant images using a hybrid **ConViTX** (CNN + Transformer) model
2. **Detects** individual lesions with bounding boxes using a fine-tuned **YOLOv8** object detector
3. **Grades** fruit quality into 4 market categories (Defect / Fresh / Immature / Mature)
4. **Explains** decisions visually using **Grad-CAM** heatmaps
5. **Scans** plants in real-time via a **Guided Live Camera** with blur detection, distance guidance, and flash control
6. **Recommends** targeted treatment protocols from a curated 2024–2025 knowledge base
7. **Serves** everything through a mobile-first **Flask web interface** with a unified prescription hub

---

## System Pipeline

```
                        ┌─────────────────────────────────────────────┐
                        │         User Input (3 methods)              │
                        │  📁 File Upload │ 📷 Camera Capture │ Live  │
                        └──────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  Disease Lab │ │Lesion Detect │ │ Quality Grade│
           │  ConViTX     │ │  YOLOv8n     │ │  ConViTX     │
           │  (6 classes) │ │  (5 classes) │ │  (4 classes) │
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                 │
                  ▼                ▼                 ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  Grad-CAM    │ │  Bounding    │ │  Market      │
           │  Heatmap     │ │  Box Overlay │ │  Recommend.  │
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                 │
                  └────────┬───────┘                 │
                           ▼                         ▼
                  ┌──────────────────┐    ┌──────────────────┐
                  │  Knowledge Base  │    │  Quality Result  │
                  │  Treatment Plan  │    │  Export/Domestic  │
                  └────────┬─────────┘    └──────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Digital         │
                  │  Prescription    │
                  │  Hub (Unified)   │
                  └──────────────────┘
```

---

## Disease Classes

The model classifies dragon fruit images into **6 classes** with the following disease characteristics:

| # | Class | Pathogen | Severity | Key Visual Cues |
|---|-------|----------|----------|-----------------|
| 1 | **Anthracnose** | *Colletotrichum gloeosporioides* | High | Reddish-brown sunken lesions, orange-pink spore masses |
| 2 | **Brown Stem Spot** | *Bipolaris cactivora* | Moderate | Circular brown spots with yellow halos on stems |
| 3 | **Gray Blight** | *Pestalotiopsis spp.* | Moderate | Gray-silver discoloration, dark-margined blighted areas |
| 4 | **Healthy** | — | None | Bright red/pink skin, no discoloration |
| 5 | **Soft Rot** | *Erwinia / Pectobacterium spp.* | High | Water-soaked mushy lesions, foul-smelling exudate |
| 6 | **Stem Canker** | *Neoscytalidium dimidiatum* | Very High | Orange-yellow water-soaked spots, stem collapse |

---

## Model Architectures

### 1. ConViTX — Hybrid CNN + Transformer (Primary Disease Classifier)

The primary model used for disease diagnosis. A lightweight hybrid architecture combining convolutional feature extraction with transformer-based global attention.

| Property | Value |
|----------|-------|
| **Architecture** | CNN Stem → Transformer Encoder → Classification Head |
| **Design goal** | Remote/low-resource deployment with strict memory limits |
| **Parameters** | < 700,000 trainable parameters |
| **Checkpoint** | `models/best_convitx.pth` |
| **Input size** | 224 × 224 × 3 |
| **Classes** | 6 (Anthracnose, Brown Stem Spot, Gray Blight, Healthy, Soft Rot, Stem Canker) |

**CNN Stem**: Extracts local lesion texture features through progressive downsampling  
**Transformer Encoder**: Captures global spatial relationships across disease regions  
**Auto-selection**: Flask automatically uses ConViTX if `best_convitx.pth` exists, otherwise falls back to ResNet50

### 2. ResNet50 — Transfer Learning (Legacy Classifier)

| Property | Value |
|----------|-------|
| **Base model** | ResNet50 pretrained on ImageNet |
| **Fine-tuning** | Frozen layers 1–2, trainable layers 3–4 + FC head |
| **FC head** | `Dropout(0.3) → Linear(2048, 6)` |
| **Parameters** | ~15.9M trainable / 25.6M total |
| **Checkpoint** | `models/best_resnet50.pth` |

### 3. YOLOv8n — Lesion Object Detection

Fine-tuned YOLOv8-nano model for real-time lesion localization and counting.

| Property | Value |
|----------|-------|
| **Base model** | YOLOv8n (Ultralytics) |
| **Task** | Multi-class object detection |
| **Classes** | 5 (Anthracnose, Brown Stem Spot, Gray Blight, Soft Rot, Stem Canker) |
| **Confidence threshold** | 0.35 |
| **IoU threshold** | 0.45 |
| **Output** | Bounding boxes, class labels, confidence scores, severity scoring |
| **Checkpoint** | `models/yolo_dragon_best.pt` |

**Severity Scoring**: Automatically grades infection level based on lesion count:
- 0 lesions → None
- 1–3 → Low  
- 4–7 → Moderate
- 8–15 → High
- 16+ → Severe

### 4. ConViTX — Quality Grading

A separate ConViTX model trained for fruit quality classification.

| Property | Value |
|----------|-------|
| **Classes** | 4 (Defect, Fresh, Immature, Mature) |
| **Checkpoint** | `models/quality_convitx.pth` |
| **Output** | Quality grade + market recommendation (Export / Domestic / Processing) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 20–30 (with early stopping, patience = 7) |
| **Batch size** | 16–32 |
| **Optimizer** | AdamW (lr = 1e-4, weight decay = 1e-4) |
| **LR scheduler** | Cosine Annealing |
| **Image size** | 224 × 224 |
| **Dataset split** | 80% train / 20% validation (seed = 42) |
| **Total images** | 724 across 6 disease classes |

### Data Augmentation

- Random horizontal & vertical flips
- Random rotation (±25°)
- Color jitter (brightness 0.3, contrast 0.3, saturation 0.2)
- ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## Classification Results

### Overall Performance (ResNet50)

| Metric | Score |
|--------|-------|
| **Best Validation Accuracy** | **95.17%** |
| **Training Time** | 11.4 minutes (CPU) |
| **Early Stopping** | Triggered at optimal epoch |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Anthracnose** | 0.95 | 0.95 | 0.95 |
| **Brown Stem Spot** | 0.93 | 0.93 | 0.93 |
| **Gray Blight** | 1.00 | 1.00 | **1.00** |
| **Healthy** | 1.00 | 1.00 | **1.00** |
| **Soft Rot** | 1.00 | 1.00 | **1.00** |
| **Stem Canker** | 0.82 | 0.82 | 0.82 |

### Key Observations

- **Flawless detectors**: Gray Blight, Healthy, and Soft Rot achieved **perfect F1 = 1.00** scores
- **Canker–Anthracnose challenge**: Both diseases present as small, dark spots in early stages
- **Efficiency**: 95%+ accuracy on CPU in under 12 minutes

---

## Explainable AI (Grad-CAM)

### What is Grad-CAM?

**Gradient-weighted Class Activation Mapping (Grad-CAM)** produces visual explanations by using gradients flowing into the final convolutional layer to highlight image regions important for model predictions.

### Implementation

- **Target layer**: Last block of the CNN stem (ConViTX) or `model.layer4[-1]` (ResNet50)
- **Process**:
  1. Forward pass → class predictions
  2. Backward pass → gradients of target class w.r.t. target layer activations
  3. Global average pooling of gradients → channel importance weights
  4. Weighted sum of activation maps → ReLU → normalized heatmap
- **Visualization**: Heatmap resized to original dimensions, colorized with JET colormap, overlaid at 45% alpha

### Why Grad-CAM Matters

| Aspect | Benefit |
|--------|---------|
| **Transparency** | Shows **where** the model is looking — lesions vs background |
| **Trust** | Validates the model learns disease-relevant features |
| **Debugging** | Reveals misclassification patterns (e.g., Canker vs Anthracnose) |
| **Clinical utility** | Highlights exact regions of concern for field practitioners |

---

## YOLOv8 Lesion Detection

The YOLOv8n model provides per-lesion localization that complements the ConViTX whole-image classification:

- **Input**: Full-resolution dragon fruit image
- **Output**: Annotated image with bounding boxes, per-class lesion counts, severity grade
- **Integration**: Results feed into the unified Digital Prescription Hub with treatment recommendations based on the dominant detected disease

### Detection Pipeline

```
Input Image → YOLOv8n Inference → NMS Filtering
    → Bounding Box Annotations
    → Per-class Lesion Count
    → Severity Score (None → Low → Moderate → High → Severe)
    → Dominant Disease → Knowledge Base → Treatment Plan
```

---

## Quality Grading

A separate ConViTX model grades dragon fruit into 4 market-value categories:

| Grade | Market Recommendation |
|-------|-----------------------|
| **Fresh Dragon Fruit** | Export quality — premium international markets |
| **Mature Dragon Fruit** | Domestic sale — local markets and supermarkets |
| **Immature Dragon Fruit** | Hold for ripening — not yet ready for sale |
| **Defect Dragon Fruit** | Processing only — juice, dried fruit, or composting |

---

## Live Camera — Guided Field Scanner

A mobile-first camera interface that helps farmers capture optimal diagnostic images in field conditions.

### Features

| Feature | Description |
|---------|-------------|
| **Live Video Feed** | Back-facing camera via `getUserMedia` with environment facing mode |
| **Blur Detection** | Real-time Laplacian variance computed every 500ms (downscaled 160×120 canvas) |
| **Distance Guide** | Sobel edge density analysis — warns "Move closer to the lesion" when detail is low |
| **Focus Ring HUD** | Animated overlay: green (ready), yellow (blurry), red (too far) |
| **Flash/Torch Toggle** | LED flashlight control via `MediaStreamTrack.applyConstraints({ torch })` |
| **Score Bars** | Real-time Sharpness and Detail percentage bars |
| **Mode Toggle** | Switch between Disease Lab (ConViTX) and Lesion Detector (YOLO) |
| **Auto-Snap** | Automatically captures when image is stable and sharp for ~3 seconds |
| **Preview & Confirm** | Captured frame preview with Analyze / Retake buttons |
| **AJAX Analysis** | Sends base64 image to `/api/analyze` JSON endpoint → redirects to treatment page |

### Guided Capture Algorithms

**Blur Detection (Laplacian Variance)**:
- Converts video frame to grayscale → applies 3×3 Laplacian kernel → computes variance
- Higher variance = sharper image; threshold ~35 for acceptable sharpness

**Detail / Distance (Sobel Edge Density)**:
- Applies simplified Sobel operators → counts edge pixels above threshold
- Edge density < 8% → "Move closer to the lesion"

---

## Digital Prescription Hub

A unified, mobile-first treatment page that replaces separate result templates. Renders for both ConViTX disease diagnosis and YOLOv8 detection workflows.

### Prescription Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **At-a-Glance Header** | Disease name, confidence, severity badge, lesion count |
| 2 | **Visual Evidence** | Side-by-side Original ↔ Grad-CAM/YOLO annotated comparison |
| 3 | **YOLO Detection Breakdown** | Per-class lesion cards with counts and confidence (detect mode) |
| 4 | **Class Probabilities** | Animated probability bars for all 6 classes (disease mode) |
| 5 | **Quick Fix Banner** | Emergency field-level first response action |
| 6 | **Step-by-Step Treatment** | Numbered actionable treatment instructions |
| 7 | **Prevention Measures** | Checklist of preventive cultural practices |
| 8 | **Environmental Conditions** | Temperature, humidity, and climate notes |
| 9 | **XAI Explanation** | Plain-language description of what the AI detected |
| 10 | **Scientific References** | Accordion with 2024–2025 journal citations |
| 11 | **Raw Detections Table** | Full YOLO detection list with coordinates (detect mode) |
| 12 | **Severity Guide** | Visual severity scale reference (detect mode) |
| 13 | **Action Footer** | Download Report / Scan Another / Back to Dashboard links |

---

## Full-Stack Architecture

```
dragonfruit_disease_ai/
│
├── models/
│   ├── convitx.py              ← ConViTX hybrid architecture definition
│   ├── train_convitx.py        ← ConViTX training script
│   ├── train_resnet50.py       ← ResNet50 training script
│   ├── train_quality.py        ← Quality grading model training
│   ├── export_edge.py          ← ONNX/TFLite export for edge deployment
│   ├── best_convitx.pth        ← Trained ConViTX disease weights
│   ├── quality_convitx.pth     ← Trained ConViTX quality weights
│   ├── best_resnet50.pth       ← Trained ResNet50 weights
│   └── yolo_dragon_best.pt     ← Fine-tuned YOLOv8n weights
│
├── xai/
│   └── gradcam.py              ← Grad-CAM XAI module (supports both architectures)
│
├── chatbot/
│   ├── advisor.py              ← 6-class knowledge base + recommendation engine
│   └── knowledge_base.py       ← Legacy version (preserved)
│
├── detect_disease.py           ← YOLOv8 DragonFruitDetector class
│
├── app/
│   ├── main.py                 ← Flask server (11 routes + JSON API)
│   ├── static/
│   │   ├── style.css           ← Full UI system (~1700 lines)
│   │   ├── icons/              ← Custom SVG/PNG icon set
│   │   └── uploads/            ← Runtime: uploaded images + overlays
│   └── templates/
│       ├── base.html           ← Base layout (nav, particles, Lucide icons)
│       ├── index.html          ← Home page — 4-mode selector grid
│       ├── disease.html        ← Disease Lab upload page
│       ├── quality.html        ← Quality Grading upload page
│       ├── detect.html         ← Lesion Detector upload page
│       ├── camera.html         ← Live Camera guided field scanner
│       ├── treatment.html      ← Unified Digital Prescription Hub
│       └── result_quality.html ← Quality grading results
│
├── icons/                      ← Custom icon assets
│   ├── 1730090.png             ← Microscope + report icon
│   ├── 563d02...jpg            ← Photo gallery icon
│   ├── 8645362.png             ← Dragon fruit (color) icon
│   ├── medical-record...webp   ← Medical clipboard icon
│   ├── pngtree-flat...jpg      ← Thermometer icon
│   └── vector-isolated...webp  ← Dragon fruit (B&W) icon
│
├── dataset/                    ← Training images (6 classes, 724 images)
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## Flask Routes & API

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page — 4-mode tool selector grid |
| `/disease` | GET | Disease Lab — ConViTX upload page |
| `/quality` | GET | Quality Grading — upload page |
| `/detect` | GET | Lesion Detector — YOLOv8 upload page |
| `/camera` | GET | Live Camera — guided field scanner |
| `/predict_disease` | POST | Disease inference → Grad-CAM → advisory → treatment page |
| `/predict_quality` | POST | Quality inference → grade + market recommendation |
| `/predict_detect` | POST | YOLOv8 detection → annotated image → treatment page |
| `/api/analyze` | POST | **JSON API** — accepts base64 camera image, returns redirect URL |
| `/camera_result` | GET | Renders treatment page from camera analysis results |
| `/icons/<filename>` | GET | Serves custom icon assets |

### JSON API — `/api/analyze`

Used by the Live Camera for AJAX-based analysis:

**Request:**
```json
{
  "mode": "disease",
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "redirect_url": "/camera_result?mode=disease&img=cam_abc123.jpg&overlay=cam_overlay_def456.jpg"
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/PrithviKiran791/Dragon_fruit_disease_ai.git
cd Dragon_fruit_disease_ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` >= 2.0 | Deep learning framework |
| `torchvision` >= 0.15 | Image transforms + pretrained models |
| `flask` >= 3.0 | Web application server |
| `ultralytics` >= 8.0 | YOLOv8 object detection |
| `opencv-python` >= 4.8 | Image processing + heatmap overlay |
| `Pillow` >= 10.0 | Image loading |
| `matplotlib` >= 3.7 | Training curves + visualization |
| `scikit-learn` >= 1.3 | Classification metrics |
| `numpy` >= 1.24 | Numerical operations |

---

## Running the Application

### Option 1: Run the Flask Web App

```bash
python app/main.py
```

Open **http://127.0.0.1:5000** in your browser.

> **Note**: The Live Camera feature requires HTTPS in production. On localhost, it works without HTTPS.

### Option 2: Train the ConViTX Disease Model

```bash
python models/train_convitx.py \
    --epochs 30 \
    --batch-size 32 \
    --num-workers 0 \
    --data-dir "dataset/Dragon Fruit (Pitahaya)/Dragon Fruit (Pitahaya)/Converted Images"
```

Saves `models/best_convitx.pth`. The Flask app automatically uses it for disease inference.

### Option 3: Train the ResNet50 Model (Legacy)

```bash
python models/train_resnet50.py \
    --epochs 20 \
    --batch-size 16 \
    --num-workers 0 \
    --data-dir "dataset/Dragon Fruit (Pitahaya)/Dragon Fruit (Pitahaya)/Converted Images"
```

### Option 4: Train the Quality Grading Model

```bash
python models/train_quality.py \
    --epochs 30 \
    --batch-size 32 \
    --data-dir "path/to/quality_dataset"
```

### Option 5: Fine-tune YOLOv8 Lesion Detector

```bash
python train_yolo_directml.py
```

Or using Ultralytics CLI:
```bash
yolo detect train data=data_dragon_lesions.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Training CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | Auto-detected | Path to ImageFolder dataset |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--patience` | 7 | Early stopping patience |
| `--train-split` | 0.8 | Train/val split ratio |
| `--num-workers` | 0 | DataLoader workers (0 for Windows) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Disease Classification** | ConViTX Hybrid (CNN + Transformer) / ResNet50 |
| **Lesion Detection** | YOLOv8n (Ultralytics) |
| **Quality Grading** | ConViTX (4-class variant) |
| **Explainability (XAI)** | Custom Grad-CAM implementation |
| **Backend** | Flask (Python) |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
| **Icons** | Lucide Icons (CDN) + Custom PNG/SVG assets |
| **Camera API** | MediaDevices API (getUserMedia) |
| **Image Processing** | OpenCV, Pillow, NumPy |
| **Knowledge Base** | Python dataclass-based advisory engine |

---

## Scientific References

1. Peng et al. (2024). *Integrated management of Colletotrichum on dragon fruit.* Plant Pathology, 73(4), 889–901.
2. Nguyen et al. (2025). *Biocontrol of dragon fruit anthracnose using Trichoderma asperellum.* BioControl, 70(1), 45–58.
3. Li et al. (2024). *First report of Bipolaris cactivora causing brown spot on dragon fruit in Southeast Asia.* Plant Disease, 108(3), 701.
4. Chen et al. (2024). *Pestalotiopsis species associated with gray blight of pitaya in Guangxi, China.* Mycological Progress, 23(1), 12.
5. Wang et al. (2025). *Integrated management of Pestalotiopsis blight on Hylocereus.* Crop Protection, 179, 106628.
6. Huang et al. (2024). *Characterization of Pectobacterium causing soft rot in pitahaya.* European J. Plant Pathol., 168(2), 301–315.
7. Zhao et al. (2025). *Biofilm-based biocontrol of bacterial soft rot in dragon fruit.* Biological Control, 192, 105502.
8. Shang et al. (2025). *Amino-oligosaccharide elicitors suppress Neoscytalidium dimidiatum in pitaya.* Pesticide Biochemistry and Physiology, 198, 105732.
9. Masyahit et al. (2024). *New pathogenicity evidence for Neoscytalidium on Hylocereus spp.* Phytopathology, 114(2), 234–245.

---

## License

This project is developed as an academic mini-project for research and educational purposes.

---

<p align="center">
  <strong>Dragon Fruit Disease AI</strong> · ConViTX + YOLOv8 + Grad-CAM + Flask · 2025–2026
</p>
