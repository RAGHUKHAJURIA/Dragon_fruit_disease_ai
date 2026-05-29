# Dragon Fruit Disease AI

AI-powered toolkit for dragon-fruit image analysis: disease classification, lesion detection, quality grading, VQA, explainability, and a Flask web UI.

Table of Contents
- Quick Start
- Features
- Installation
- Run the app
- APIs & Routes
- Model artifacts
- Development & Training
- Project layout
- License

Quick Start
1. Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

2. (Optional) copy `.env.example` to `.env` and set `GEMINI_API_KEY` if you want Gemini chat integration.

3. Start the Flask app:

```bash
python app/main.py
```

Open http://127.0.0.1:5000 in your browser.

Features
- Disease classification (6 classes) with ConViTX-based models and Grad-CAM explainability (TTA supported).
- Lesion detection using YOLOv8 with lesion boxes and severity estimate.
- Quality grading (4-class) for produce sorting.
- Visual Question Answering (VQA) for image+question inference.
- Advisory chatbot (`/api/chat`) using Gemini (when configured) with a deterministic fallback in `chatbot/advisor.py`.

Installation
1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate and install:

Windows PowerShell:
```powershell
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app
- Flask (primary):

```bash
python app/main.py
```

- Streamlit (legacy demo):

```bash
streamlit run app/app.py
```

APIs & Routes (selected)
- Pages:
  - `GET /` — Home
  - `GET /disease`, `/quality`, `/detect`, `/camera`, `/vqa`
- Form endpoints:
  - `POST /predict_disease` — runs disease classifier + Grad-CAM + advisory
  - `POST /predict_quality`
  - `POST /predict_detect`
  - `POST /predict_vqa`
- JSON APIs:
  - `POST /api/analyze` — camera/base64 analysis flow
  - `POST /api/vqa` — VQA JSON payload
  - `POST /api/vqa_query` — VQA using cached visual features (uid)
  - `POST /api/chat` — advisory chatbot

Example payloads
- `/api/chat`:

```json
{
  "message": "How do I treat stem canker?",
  "disease": "Stem_Canker",
  "confidence": 0.91,
  "history": []
}
```

- `/api/vqa`:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "question": "What disease is visible?",
  "lang": "en"
}
```

Model artifacts (place under `models/`)
- Disease: `best_convitx_pretrained.pth` (preferred), `best_convitx.pth` (fallback)
- Quality: `quality_convitx.pth` and `models/quality_classes.txt`
- YOLO detector: `yolo_dragon_best.pt`
- VQA: `best_vqa.pth`, `vqa_vocab.json`, `best_vqa_config.json`

The app lazy-loads models and reports clear errors if any required files are missing.

Development & Training (selected commands)
- Train disease models:

```bash
python models/train_convitx.py
python models/train_convitx_pretrained.py
```

- Train quality model:

```bash
python models/train_quality.py
```

- Prepare YOLO dataset and validate:

```bash
python setup_yolo_folders.py
python prepare_yolo_dataset.py
python validate_yolo_dataset.py
```

- Train YOLO (DirectML path used on Windows):

```bash
python train_yolo_directml.py
```

- VQA training/export:

```bash
python models/train_vqa.py
python models/export_vqa_edge.py
```

Edge export (examples)
- Disease ONNX / quantized export:

```bash
python models/export_edge.py --model-path models/best_convitx.pth --output-dir models
```

- VQA edge export:

```bash
python models/export_vqa_edge.py
```

Project layout (high level)

See the main modules and scripts in the repository root — app, chatbot, models, xai, dataset helpers, and training scripts.

Notes
- This repo contains active production code alongside legacy/demo scripts (e.g. `app/app.py`). Prefer `app/main.py` as the runtime entrypoint.
- `README copy.md` is a stale duplicate and can be ignored.
- Keep large model artifacts out of Git; use external storage for weight files.

License
Academic / research mini-project. Reuse with attribution and appropriate validation before deployment.
