
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

st.markdown("---")

detect = st.button(
    "🔍 Detect AI or Real",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None
)

if detect:

    st.info(
        "The detection interface is ready. "
        "Actual model inference will be connected "
        "in the next development steps."
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    FAKTRION033 · Deep Learning Research Project
    <br>
    Synthetic vs Real Visual Content Detection
</div>
""", unsafe_allow_html=True)
