
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="FAKTRION033 | AI Media Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

PHOTO_MODELS = [
    "CLIP ViT-B/32",
    "Swin-Tiny",
    "ConvNeXt-Tiny",
    "DINOv2 ViT-S/14",
    "EfficientNet-B0"
]

VIDEO_MODELS = [
    "UNITE-inspired SigLIP Transformer",
    "CLIP ViT-B/32 + Temporal Pooling",
    "DINOv2 + Temporal Attention",
    "Video Swin Transformer-Tiny",
    "VideoMAE-Base (Fine-tuned)"
]

st.markdown("""
<style>

/* Main application */
.stApp {
    background:
        radial-gradient(
            ellipse at 80% 0%,
            rgba(124,58,237,0.13),
            transparent 42%
        ),
        #09090F;
}

/* Main content */
.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #10101B;
    border-right: 1px solid #25253A;
}

/* Hero section */
.hero {
    padding: 35px 32px;
    background: linear-gradient(
        120deg,
        #1C1635,
        #11111D
    );
    border: 1px solid #343052;
    border-radius: 22px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: clamp(30px, 5vw, 48px);
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.hero-title span {
    color: #A78BFA;
}

.hero-subtitle {
    color: #A1A1AA;
    font-size: 15px;
    line-height: 1.8;
}

/* Section headings */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #F5F3FF;
    margin-bottom: 8px;
}

.section-desc {
    color: #A1A1AA;
    font-size: 14px;
    margin-bottom: 15px;
}

/* Cards */
.info-card {
    background: #151522;
    border: 1px solid #2B2B3D;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 15px;
}

.info-label {
    color: #A1A1AA;
    font-size: 12px;
    margin-bottom: 7px;
}

.info-value {
    color: #F4F4F5;
    font-size: 18px;
    font-weight: 700;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #7C3AED,
        #9333EA
    );
    border: none;
    border-radius: 12px;
    min-height: 50px;
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover {
    background: #6D28D9;
}

/* Inputs */
[data-testid="stFileUploader"] {
    border-radius: 15px;
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
}

/* Footer */
.footer {
    color: #71717A;
    text-align: center;
    font-size: 12px;
    padding: 28px 0 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 🔍 FAKTRION033")
    st.caption("AI & Real Media Detection")

    st.divider()

    st.markdown("### Detection Mode")

    media_type = st.radio(
        "Choose Media Type",
        ["Photo Detection", "Video Detection"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### About")
    st.write(
        "A deep learning application for detecting "
        "synthetic and authentic visual content."
    )

    st.caption("10 supported models · 2 media types")

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <div class="hero-title">
        FAKTRION<span>033</span>
    </div>
    <div class="hero-subtitle">
        Advanced AI-Generated & Real Media Detection
        <br>
        Powered by Deep Learning and Computer Vision
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- MEDIA MODE ----------------

is_photo = media_type == "Photo Detection"

if is_photo:

    st.markdown(
        '<div class="section-title">📷 Photo Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-desc">'
        'Upload an image and select a deep learning model.'
        '</div>',
        unsafe_allow_html=True
    )

    models = PHOTO_MODELS

    file_types = [
        "jpg", "jpeg", "png", "webp"
    ]

else:

    st.markdown(
        '<div class="section-title">🎬 Video Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-desc">'
        'Upload a video and choose a video detection model.'
        '</div>',
        unsafe_allow_html=True
    )

    models = VIDEO_MODELS

    file_types = [
        "mp4", "mov", "avi", "mkv"
    ]

# ---------------- MODEL SELECTION ----------------

selected_model = st.selectbox(
    "Select Deep Learning Model",
    models,
    index=0
)

st.markdown(
    f"""
    <div class="info-card">
        <div class="info-label">SELECTED MODEL</div>
        <div class="info-value">{selected_model}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Photo" if is_photo else "Upload Video",
    type=file_types,
    key="photo_upload" if is_photo else "video_upload"
)

if uploaded_file is not None:

    st.markdown("### Media Preview")

    if is_photo:
        st.image(
            uploaded_file,
            caption="Uploaded Photo",
            width="stretch"
        )

    else:
        st.video(uploaded_file)

    st.caption(
        f"Filename: {uploaded_file.name} | "
        f"Size: {uploaded_file.size / (1024*1024):.2f} MB"
    )


# ---------------- DETECTION ----------------

import io
import tempfile
import os
import torch
from PIL import Image

from photo_inference import load_photo_model, predict_photo
from video_inference import (
    extract_video_frames,
    load_video_clip, predict_video_clip,
    load_video_dinov2, predict_video_dinov2,
    load_video_swin, predict_video_swin,
    load_video_unite, predict_video_unite,
    load_video_videomae, predict_video_videomae
)

MODEL_IDS = {'CLIP ViT-B/32': 'photo_clip', 'Swin-Tiny': 'photo_swin', 'ConvNeXt-Tiny': 'photo_convnext', 'DINOv2 ViT-S/14': 'photo_dinov2', 'EfficientNet-B0': 'photo_efficientnet', 'UNITE-inspired SigLIP Transformer': 'video_unite', 'CLIP ViT-B/32 + Temporal Pooling': 'video_clip', 'DINOv2 + Temporal Attention': 'video_dinov2', 'Video Swin Transformer-Tiny': 'video_swin', 'VideoMAE-Base (Fine-tuned)': 'video_videomae'}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path(__file__).resolve().parent / "models"

@st.cache_resource(show_spinner=False, max_entries=2)
def get_detection_model(model_id, device):

    if model_id.startswith("photo_"):
        return load_photo_model(model_id, MODELS_DIR, device)

    loaders = {
        "video_clip": load_video_clip,
        "video_dinov2": load_video_dinov2,
        "video_swin": load_video_swin,
        "video_unite": load_video_unite,
        "video_videomae": load_video_videomae
    }

    return loaders[model_id](MODELS_DIR, device)


def detect_media(model_id, uploaded_file):

    loaded = get_detection_model(model_id, DEVICE)

    if model_id.startswith("photo_"):

        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        )

        encoder, classifier, preprocess, extract = loaded

        return predict_photo(
            image, encoder, classifier,
            preprocess, extract, DEVICE
        )

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        raise ValueError("Unsupported video format.")

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp.write(uploaded_file.getvalue())
            temporary_path = temp.name

        frames, _ = extract_video_frames(
            temporary_path,
            num_frames=8,
            size=224
        )

        if model_id == "video_clip":
            encoder, processor, classifier = loaded
            return predict_video_clip(
                frames, encoder, processor, classifier, DEVICE
            )

        if model_id == "video_dinov2":
            encoder, preprocess, classifier = loaded
            return predict_video_dinov2(
                frames, encoder, preprocess, classifier, DEVICE
            )

        if model_id == "video_swin":
            encoder, classifier = loaded
            return predict_video_swin(
                frames, encoder, classifier, DEVICE
            )

        if model_id == "video_unite":
            encoder, processor, classifier = loaded
            return predict_video_unite(
                frames, encoder, processor, classifier, DEVICE
            )

        if model_id == "video_videomae":
            model, processor = loaded
            return predict_video_videomae(
                frames, model, processor, DEVICE
            )

        raise ValueError("Unsupported video model.")

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


st.markdown("---")

detect = st.button(
    "🔍 Detect AI or Real",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None
)

if detect and uploaded_file is not None:

    model_id = MODEL_IDS[selected_model]

    try:
        with st.spinner("Analyzing media with selected AI model..."):
            result = detect_media(model_id, uploaded_file)

        prediction = result["prediction"]
        confidence = result["confidence"]

        st.markdown("### Detection Result")

        if prediction == "AI":
            st.error(f"AI-Generated Content — {confidence:.2f}%")
        else:
            st.success(f"Real Content — {confidence:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "AI Score",
                f'{result["ai_probability"]:.2f}%'
            )

        with col2:
            st.metric(
                "Real Score",
                f'{result["real_probability"]:.2f}%'
            )

        st.progress(
            min(max(float(confidence) / 100, 0.0), 1.0)
        )

        st.caption(
            "Scores are model softmax outputs, not guaranteed "
            "probabilities of correctness."
        )

    except Exception as error:
        st.error(f"Detection failed: {error}")

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    FAKTRION033 · Deep Learning Research Project
    <br>
    Synthetic vs Real Visual Content Detection
</div>
""", unsafe_allow_html=True)
