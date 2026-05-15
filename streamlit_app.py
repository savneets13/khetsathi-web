"""
KhetSathi Web — Crop Disease Detection
========================================

A simple Streamlit web app that uses the trained MobileNetV2 model
to identify crop diseases from leaf photos. Designed for smartphone
browsers — works on any device with internet.

Deployment: Streamlit Community Cloud (free)
Repository: GitHub
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="KhetSathi — Crop Disease Detector",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for earthy agricultural theme
st.markdown("""
<style>
    .stApp {
        background-color: #FAF7F0;
    }
    h1, h2, h3 {
        color: #2D5016;
        font-family: 'Georgia', serif;
    }
    .stButton>button {
        background-color: #2D5016;
        color: #FAF7F0;
        border: none;
        padding: 0.6rem 1.4rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #87A96B;
    }
    .severity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .severity-healthy {
        background-color: #C8D5B9;
        color: #2D5016;
    }
    .severity-moderate {
        background-color: #FFE4B5;
        color: #8B6914;
    }
    .severity-severe {
        background-color: #FFB6B0;
        color: #8B2500;
    }
    .info-card {
        background: white;
        border-left: 4px solid #87A96B;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_PATH = "KhetSathiModel.tflite"
DISEASE_INFO_PATH = "disease_info.json"

# Class labels in the order the model outputs them
CLASS_LABELS = [
    'Bell Pepper — Bacterial Spot',
    'Bell Pepper — Healthy',
    'Potato — Early Blight',
    'Potato — Late Blight',
    'Potato — Healthy',
    'Tomato — Bacterial Spot',
    'Tomato — Late Blight',
    'Tomato — Leaf Mold',
    'Tomato — Spider Mites',
    'Tomato — Target Spot',
    'Tomato — Yellow Leaf Curl Virus',
    'Tomato — Mosaic Virus',
    'Tomato — Healthy'
]

IMG_SIZE = 224

# ============================================================================
# MODEL LOADING (cached for speed)
# ============================================================================

@st.cache_resource
def load_tflite_model(path):
    """Load TFLite model and return interpreter ready for inference."""
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_disease_info(path):
    """Load the disease information JSON."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        # Build lookup by display name + crop
        return {f"{d['crop']} — {d['display_name']}": d for d in data['diseases']}
    return {}

# ============================================================================
# INFERENCE
# ============================================================================

def predict(image: Image.Image, interpreter) -> tuple:
    """
    Run inference on a PIL image.
    Returns (top_class_label, all_probabilities_dict).
    """
    # Resize and convert to RGB
    img = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocessing: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Get input/output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get top class
    top_idx = int(np.argmax(output))
    top_label = CLASS_LABELS[top_idx]

    # Build probability dictionary
    probs = {CLASS_LABELS[i]: float(output[i]) for i in range(len(CLASS_LABELS))}

    return top_label, probs

# ============================================================================
# UI
# ============================================================================

# Header
st.markdown("# 🌱 KhetSathi")
st.markdown("### Crop Disease Detector for Indian Farmers")
st.markdown(
    "*Snap a photo of a tomato, potato, or bell pepper leaf to identify diseases. "
    "Works for 13 common conditions across these crops.*"
)
st.markdown("---")

# Load model and info
try:
    interpreter = load_tflite_model(MODEL_PATH)
    disease_info = load_disease_info(DISEASE_INFO_PATH)
    model_ready = True
except Exception as e:
    st.error(f"⚠️ Model failed to load: {e}")
    st.info("Make sure KhetSathiModel.tflite is in the same directory as this script.")
    model_ready = False

if model_ready:
    # Image input
    st.markdown("## 📸 Provide a Leaf Photo")

    tab1, tab2 = st.tabs(["Upload Photo", "Take Photo"])

    image = None

    with tab1:
        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Best results: clear photo of a single leaf, well-lit, plain background"
        )
        if uploaded is not None:
            image = Image.open(uploaded)

    with tab2:
        captured = st.camera_input("Take a photo with your camera")
        if captured is not None:
            image = Image.open(captured)

    # If we have an image, show it and predict
    if image is not None:
        st.markdown("---")
        st.markdown("## 🔍 Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Your image:**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("**Prediction:**")
            with st.spinner("Analyzing leaf..."):
                top_label, probs = predict(image, interpreter)

            # Look up disease info
            info = disease_info.get(top_label)

            # Display top prediction
            st.markdown(f"### {top_label}")

            top_prob = probs[top_label]
            st.markdown(f"**Confidence:** {top_prob*100:.1f}%")
            st.progress(min(top_prob, 1.0))

            # Severity badge
            if info:
                severity = info.get('severity', 'moderate')
                severity_class = f"severity-{severity}" if severity in ['healthy', 'moderate', 'severe'] else 'severity-moderate'
                severity_label = "🟢 Healthy" if severity == "none" else f"🟡 {severity.title()}" if severity == "moderate" else f"🔴 {severity.title()}"
                if severity == "none":
                    severity_class = "severity-healthy"
                st.markdown(
                    f'<span class="severity-badge {severity_class}">{severity_label}</span>',
                    unsafe_allow_html=True
                )

        # Detailed info
        if info:
            st.markdown("---")
            st.markdown(f"## About {info['display_name']}")

            if info.get('pathogen'):
                st.markdown(f"**Pathogen:** *{info['pathogen']}*")

            # Symptoms
            st.markdown("### 🔬 Symptoms")
            st.markdown(f'<div class="info-card">{info["symptoms"]}</div>', unsafe_allow_html=True)

            # Treatment (only if not healthy)
            if not info.get('is_healthy', False):
                st.markdown("### 💊 Treatment")
                st.markdown(f'<div class="info-card">{info["treatment"]}</div>', unsafe_allow_html=True)

            # Prevention
            st.markdown("### 🛡️ Prevention")
            st.markdown(f'<div class="info-card">{info["prevention"]}</div>', unsafe_allow_html=True)

            # Favorable conditions
            if info.get('favorable_conditions'):
                st.markdown("### 🌤️ Favorable Conditions")
                st.markdown(f'<div class="info-card">{info["favorable_conditions"]}</div>', unsafe_allow_html=True)

        # Top 3 predictions
        st.markdown("---")
        with st.expander("📊 See top 3 predictions"):
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for label, prob in sorted_probs:
                st.markdown(f"**{label}** — {prob*100:.2f}%")
                st.progress(min(prob, 1.0))

    else:
        # Instructions when no image yet
        st.info(
            "👆 Upload a photo or take one with your camera to get started.\n\n"
            "**Best results when:**\n"
            "- The leaf fills most of the frame\n"
            "- Good natural lighting\n"
            "- Plain background (paper, fabric, or sky)\n"
            "- Sharp focus on the leaf"
        )

# Footer
st.markdown("---")
st.markdown(
    "<small style='color: #87A96B;'>"
    "🌾 KhetSathi — BCA Major Project | Crop Disease Detection using Deep Learning | "
    "Model: MobileNetV2 (Transfer Learning) | Test Accuracy: 95.32%"
    "</small>",
    unsafe_allow_html=True
)
