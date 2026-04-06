"""
Dragon Fruit Disease Advisory Chatbot — Streamlit Web App

Pipeline:
  User uploads image → Preprocessing → CNN (EfficientNet/ResNet)
  → Disease prediction → Grad-CAM heatmap → Knowledge DB → Chatbot response
"""

import os
import sys
import io
import torch
import timm
import numpy as np
import streamlit as st
from PIL import Image

# Make sibling modules importable when running from app/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from xai.gradcam import run_gradcam, get_target_layer_efficientnet, infer_transforms
from chatbot.knowledge_base import generate_recommendation, DISEASE_KNOWLEDGE

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT, "models", "best_model.pth")
MODEL_NAME   = "efficientnet_b3"
CLASS_NAMES  = ["Healthy", "Anthracnose", "Stem_Canker", "Fruit_Rot", "Brown_Spot"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── MODEL LOADER (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        st.success("✅ Loaded trained model weights.")
    else:
        st.warning("⚠️ No trained weights found — using random weights (for demo only).")
    model.eval().to(DEVICE)
    return model

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dragon Fruit Disease Advisory",
    page_icon="🌵",
    layout="wide",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f0f1a; color: #e8e8f0; }
    .main { background-color: #0f0f1a; }
    .stApp { font-family: 'Inter', sans-serif; }
    .disease-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin: 0.5rem 0;
    }
    .severity-high    { color: #ff6b6b; font-weight: bold; }
    .severity-moderate{ color: #ffa94d; font-weight: bold; }
    .severity-low     { color: #69db7c; font-weight: bold; }
    .severity-none    { color: #63e6be; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── APP UI ──────────────────────────────────────────────────────────────────
st.title("🌵 Dragon Fruit Disease Advisory Chatbot")
st.markdown(
    "Upload a dragon fruit plant or fruit image. "
    "The AI will analyze it, highlight infected regions with **Grad-CAM XAI**, "
    "and provide evidence-based treatment recommendations."
)
st.divider()

# ── Sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ System Info")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Classes:** {len(CLASS_NAMES)}")
    for c in CLASS_NAMES:
        st.markdown(f"  • {c}")
    st.divider()
    st.markdown("**XAI Method:** Grad-CAM")
    st.markdown("**Literature:** 2024–2025 research")

# ── Main: upload & analyse ────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a dragon fruit image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    analyse_btn = st.button("🔍 Analyse Disease", type="primary", disabled=not uploaded_file)

# ── Analysis pipeline ─────────────────────────────────────────────────────────
if uploaded_file and analyse_btn:
    with st.spinner("Loading model..."):
        model = load_model()
        target_layer = get_target_layer_efficientnet(model)

    # Save temp image for Grad-CAM
    tmp_path = os.path.join(RESULTS_DIR, "tmp_input.jpg")
    pil_img.save(tmp_path)

    with st.spinner("Running Grad-CAM XAI analysis..."):
        gradcam_result = run_gradcam(
            model        = model,
            target_layer = target_layer,
            image_path   = tmp_path,
            class_names  = CLASS_NAMES,
            save_path    = os.path.join(RESULTS_DIR, "gradcam_output.png"),
        )

    predicted_class = gradcam_result["predicted_class"]
    confidence      = gradcam_result["confidence"]
    overlay_img     = gradcam_result["overlay"]

    # Generate chatbot recommendation
    response = generate_recommendation(
        predicted_class = predicted_class,
        confidence      = confidence,
    )

    # ── Display Results ───────────────────────────────────────────────────────
    with col_results:
        st.subheader("📊 Analysis Results")

        # Severity color mapping
        sev_class = {
            "None": "severity-none", "Low": "severity-low",
            "Moderate": "severity-moderate", "High": "severity-high",
            "Very High": "severity-high",
        }.get(response.severity, "severity-none")

        st.markdown(
            f'<div class="disease-card">'
            f'<h3>🌿 Detected: <code>{predicted_class}</code></h3>'
            f'<p>Confidence: <strong>{confidence:.1%}</strong></p>'
            f'<p>Severity: <span class="{sev_class}">{response.severity}</span></p>'
            f'<p>Pathogen: <em>{response.pathogen or "None"}</em></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Grad-CAM overlay
        st.subheader("🔬 Grad-CAM Heatmap (XAI)")
        col_orig, col_overlay = st.columns(2)
        with col_orig:
            st.image(pil_img, caption="Original", use_column_width=True)
        with col_overlay:
            st.image(overlay_img, caption="Grad-CAM Overlay", use_column_width=True)

        # Probability bar chart
        st.subheader("📈 Class Probabilities")
        probs_dict = gradcam_result["probabilities"]
        st.bar_chart(probs_dict)

        # Chatbot advisory
        st.subheader("💬 Disease Advisory")
        st.info(response.description)

        with st.expander("🔬 XAI Explanation", expanded=True):
            st.markdown(response.xai_explanation)

        with st.expander("💊 Treatment Plan", expanded=True):
            for i, step in enumerate(response.treatment_steps, 1):
                st.markdown(f"**{i}.** {step}")

        with st.expander("🛡️ Prevention Measures"):
            for tip in response.prevention_tips:
                st.markdown(f"- {tip}")

        with st.expander("🌦️ Environmental Conditions"):
            st.markdown(response.environmental_note)

        if response.literature:
            with st.expander("📚 Scientific Literature (2024–2025)"):
                for ref in response.literature:
                    st.markdown(f"- {ref}")

        # Download report button
        report_text = response.format_text()
        st.download_button(
            "📄 Download Advisory Report",
            data       = report_text,
            file_name  = f"dragon_fruit_report_{predicted_class}.md",
            mime       = "text/markdown",
        )

elif not uploaded_file:
    with col_results:
        st.info("👆 Upload a dragon fruit image on the left to begin analysis.")

        # Show sample disease info cards
        st.subheader("📖 Known Disease Classes")
        for disease, info in DISEASE_KNOWLEDGE.items():
            if disease == "Healthy":
                continue
            with st.expander(f"🔴 {disease}"):
                st.markdown(f"**Pathogen:** {info['pathogen']}")
                st.markdown(f"**Severity:** {info['severity']}")
                st.markdown("**Visual Signs:**")
                for v in info["visual_cues"]:
                    st.markdown(f"  - {v}")
