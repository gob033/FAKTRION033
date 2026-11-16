import streamlit as st
from pathlib import Path
import html

st.set_page_config(page_title="FAKTRION033 | AI Authenticity Lab", page_icon="✦", layout="wide", initial_sidebar_state="expanded")

PHOTO_MODELS = ["CLIP ViT-B/32", "Swin-Tiny", "ConvNeXt-Tiny", "DINOv2 ViT-S/14", "EfficientNet-B0"]
VIDEO_MODELS = ["UNITE-inspired SigLIP Transformer", "CLIP ViT-B/32 + Temporal Pooling", "DINOv2 + Temporal Attention", "Video Swin Transformer-Tiny", "VideoMAE-Base (Fine-tuned)"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme:dark}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.stApp {background:radial-gradient(ellipse 65% 45% at 80% -10%,rgba(122,75,255,.19),transparent 75%),radial-gradient(ellipse 40% 30% at -10% 45%,rgba(31,92,181,.10),transparent 80%),#080a11;color:#eaeaf5}
.block-container {max-width:1170px;padding-top:1.6rem;padding-bottom:2.8rem}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#10121f,#0b0d17)!important;border-right:1px solid #25273c}
[data-testid="stSidebar"] .block-container {padding-top:1.6rem}
[data-testid="stHeader"] {background:transparent}
[data-testid="stSidebarNav"] {display:none}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
.brand {display:flex;align-items:center;gap:12px;margin:0 0 8px}
.brand-mark {display:grid;place-items:center;width:40px;height:40px;border-radius:12px;background:linear-gradient(140deg,#a18aff,#5d42d8);box-shadow:0 6px 28px #6549e544;color:white;font-size:23px;font-weight:700}
.brand-name {font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;letter-spacing:-.05em;color:#f7f5ff}
.brand-name span {color:#aa99ff}
.brand-caption {font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:#898aa5;margin:0 0 24px 52px}
.sidebar-label {font-size:11px;color:#8286a4;text-transform:uppercase;letter-spacing:.16em;font-weight:800;margin:18px 0 8px}
.sidebar-note {background:#15182a;border:1px solid #2b2e47;border-radius:14px;padding:16px;color:#a9acc4;font-size:12px;line-height:1.8;margin-top:24px}
.sidebar-note strong {color:#eeeaff}
.eyebrow {font-size:11px;text-transform:uppercase;letter-spacing:.2em;color:#b6a4ff;font-weight:800}
.hero {position:relative;overflow:hidden;padding:40px 42px 34px;border:1px solid #393052;border-radius:24px;background:radial-gradient(circle at 85% 12%,rgba(153,113,255,.23),transparent 36%),linear-gradient(125deg,#19172d 0%,#111423 62%,#121528 100%);box-shadow:0 22px 65px #00000035;margin:10px 0 22px}
.hero:after {content:'';position:absolute;right:-58px;top:-90px;width:310px;height:310px;border-radius:50%;border:1px solid #8d72df33;box-shadow:0 0 0 34px #8d72df0b,0 0 0 75px #8d72df09;pointer-events:none}
.hero h1 {color:#fff;font-size:clamp(34px,5.2vw,58px);line-height:1.08;margin:12px 0 15px;font-weight:700;max-width:740px}
.hero h1 em {font-style:normal;background:linear-gradient(100deg,#c7b5ff,#8f7dff,#75a4ff);background-clip:text;-webkit-background-clip:text;color:transparent}
.hero p {color:#afb1c9;font-size:15px;line-height:1.75;max-width:600px;margin:0}
.hero-pills {display:flex;gap:9px;flex-wrap:wrap;margin-top:24px}
.hero-pill {font-size:11px;font-weight:700;color:#c9c8df;background:#ffffff0a;border:1px solid #ffffff19;border-radius:100px;padding:8px 12px}
.topline {display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}
.live-dot {display:inline-block;width:7px;height:7px;background:#5ee0af;border-radius:50%;box-shadow:0 0 0 4px #5ee0af19;margin-right:7px}
.live-status {color:#b6bfcb;font-size:12px;font-weight:600}
.stats {display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-bottom:26px}
.stat {padding:17px 20px;border:1px solid #292b42;border-radius:16px;background:linear-gradient(155deg,#161827,#11131f)}
.stat-num {font-family:'Space Grotesk',sans-serif;color:#f7f5ff;font-size:26px;font-weight:700;line-height:1.2}
.stat-label {color:#9296ae;font-size:12px;margin-top:6px}
.section-head {margin:0 0 18px}
.section-kicker {font-size:11px;letter-spacing:.17em;text-transform:uppercase;color:#a69aff;font-weight:800;margin-bottom:7px}
.section-title {font-family:'Space Grotesk',sans-serif;color:#f6f4ff;font-size:25px;font-weight:700;letter-spacing:-.035em}
.section-desc {color:#9296b0;font-size:13px;margin-top:5px;line-height:1.7}
.step-label {font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#a9a4c3;font-weight:800;margin-bottom:7px}
.stSelectbox label,.stFileUploader label {font-weight:700!important;color:#e4e3f5!important;font-size:13px!important}
[data-baseweb="select"]>div {background:#151829!important;border:1px solid #353651!important;border-radius:13px!important;min-height:48px}
[data-testid="stFileUploader"] {border:1px solid #373551;border-radius:17px;padding:12px;background:#121522}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #5e548c!important;border-radius:13px!important;background:linear-gradient(120deg,#19162c,#131726)!important;padding:22px 14px!important}
[data-testid="stFileUploaderDropzone"] button {background:#2b2350!important;color:#eee6ff!important;border:1px solid #5a4c93!important;border-radius:9px!important}
[data-testid="stFileUploaderDropzone"] small {color:#aaaac0!important}
[data-testid="stImage"] img {border-radius:14px;border:1px solid #303349}
[data-testid="stVideo"] {border-radius:14px;overflow:hidden}
.stButton>button[kind="primary"] {background:linear-gradient(100deg,#7150ed,#9d65fa)!important;border:1px solid #b19cff33!important;color:white!important;border-radius:13px!important;min-height:55px!important;font-size:15px!important;font-weight:800!important;box-shadow:0 10px 26px #704bed35;transition:filter .2s,transform .2s}
.stButton>button[kind="primary"]:hover {filter:brightness(1.12);transform:translateY(-1px)}
.stButton>button:disabled {opacity:.48!important;box-shadow:none!important}
.model-card {background:#171829;border:1px solid #30314a;border-radius:14px;padding:15px 17px;margin:12px 0 19px}
.model-card small {display:block;color:#9194b2;font-size:10px;letter-spacing:.13em;font-weight:800;margin-bottom:6px}
.model-card strong {font-family:'Space Grotesk',sans-serif;color:#e9e4ff;font-size:15px}
.preview-head {font-size:15px;color:#eae8f9;font-weight:700;margin:18px 0 12px}
.file-meta {color:#9296b2;font-size:12px;margin:10px 0 20px}
.result-panel {background:linear-gradient(120deg,#1a1930,#131829);border:1px solid #49406d;border-radius:20px;padding:24px;margin:18px 0 15px}
.result-panel.ai {border-color:#8d577c;background:linear-gradient(120deg,#2a1b32,#161726)}
.result-panel.real {border-color:#356f69;background:linear-gradient(120deg,#122a2a,#151826)}
.result-eyebrow {font-size:11px;letter-spacing:.15em;color:#a6a6c1;font-weight:800;text-transform:uppercase}
.result-title {font-family:'Space Grotesk',sans-serif;color:#f9f7ff;font-size:clamp(23px,4vw,34px);font-weight:700;margin:8px 0 4px}
.result-desc {font-size:13px;color:#b7b5c9}
.score-row {display:flex;justify-content:space-between;color:#e5e5f6;font-size:13px;font-weight:700;margin:18px 0 8px}
.score-track {height:11px;background:#292a3f;border-radius:20px;overflow:hidden}
.score-fill {height:100%;border-radius:20px;background:linear-gradient(90deg,#7960e8,#b595ff)}
.score-fill.real {background:linear-gradient(90deg,#278b88,#55d6b0)}
.score-foot {color:#888ca7;font-size:11px;line-height:1.65;margin:15px 0 5px}
.footer {text-align:center;color:#71758d;font-size:11px;border-top:1px solid #26283c;padding:26px 0 4px;margin-top:42px;line-height:1.9}
[data-testid="stMetric"] {background:#151727;border:1px solid #303149;border-radius:15px;padding:16px 18px}
[data-testid="stMetricLabel"] {color:#a5a6bc!important}
[data-testid="stMetricValue"] {color:#f5f3ff!important}
@media(max-width:700px){.block-container{padding:1rem 1rem 2rem}.hero{padding:27px 23px}.hero h1{font-size:36px}.stats{gap:8px}.stat{padding:14px 12px}.stat-num{font-size:21px}.stat-label{font-size:10px}.hero:after{opacity:.4}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✦</div><div class="brand-name">FAKTRION<span>033</span></div></div><div class="brand-caption">Visual Authenticity Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
    media_type = st.radio("Detection mode", ["Photo Detection", "Video Detection"], label_visibility="collapsed")
    st.markdown('<div class="sidebar-note"><strong>Built for digital trust.</strong><br>Explore whether visual media is synthetic or authentic using independent deep learning models.<br><br>✦ 5 photo models &nbsp; · &nbsp; 5 video models</div>', unsafe_allow_html=True)
    st.caption("Research prototype · FAKTRION033")

st.markdown('<div class="topline"><span class="eyebrow">✦ AI FORENSICS / MEDIA INTELLIGENCE</span><span class="live-status"><span class="live-dot"></span>Detection workspace</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">NEXT-GENERATION CONTENT ANALYSIS</div><h1>See beyond the image.<br><em>Detect what’s real.</em></h1><p>Analyze photos and videos with deep learning. Choose a specialized model, upload your media, and explore its AI-versus-real classification scores.</p><div class="hero-pills"><span class="hero-pill">✦ Deep learning powered</span><span class="hero-pill">◈ Photo & video analysis</span><span class="hero-pill">◎ Model-level insights</span></div></div>', unsafe_allow_html=True)
st.markdown('<div class="stats"><div class="stat"><div class="stat-num">10</div><div class="stat-label">Research models</div></div><div class="stat"><div class="stat-num">02</div><div class="stat-label">Media formats</div></div><div class="stat"><div class="stat-num">AI / REAL</div><div class="stat-label">Classification output</div></div></div>', unsafe_allow_html=True)

is_photo = media_type == "Photo Detection"
models = PHOTO_MODELS if is_photo else VIDEO_MODELS
file_types = ["jpg", "jpeg", "png", "webp"] if is_photo else ["mp4", "mov", "avi", "mkv"]
mode_title = "Photo authenticity analysis" if is_photo else "Video authenticity analysis"
mode_desc = "Inspect an image with a selected visual encoder." if is_photo else "Analyze sampled video frames with a temporal detection model."
st.markdown(f'<div class="section-head"><div class="section-kicker">01 / ANALYSIS WORKSPACE</div><div class="section-title">{mode_title}</div><div class="section-desc">{mode_desc}</div></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.05], gap="large")
with left:
    selected_model = st.selectbox("Choose detection model", models, index=0)
    st.markdown(f'<div class="model-card"><small>ACTIVE DETECTION ENGINE</small><strong>✦ {html.escape(selected_model)}</strong></div>', unsafe_allow_html=True)
with right:
    uploaded_file = st.file_uploader("Upload photo" if is_photo else "Upload video", type=file_types, key="photo_upload" if is_photo else "video_upload")

if uploaded_file is not None:
    st.markdown('<div class="preview-head">◈ &nbsp; Media preview</div>', unsafe_allow_html=True)
    if is_photo:
        st.image(uploaded_file, caption="Uploaded image", width="stretch")
    else:
        st.video(uploaded_file)
    st.markdown(f'<div class="file-meta">FILE &nbsp; {html.escape(uploaded_file.name)} &nbsp; · &nbsp; {uploaded_file.size / (1024*1024):.2f} MB</div>', unsafe_allow_html=True)

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



st.markdown('<div class="section-kicker" style="margin-top:20px">02 / DETECTION</div>', unsafe_allow_html=True)
detect = st.button("✦  Analyze media authenticity", type="primary", use_container_width=True, disabled=uploaded_file is None)

if detect and uploaded_file is not None:
    model_id = MODEL_IDS[selected_model]
    try:
        with st.spinner("Loading detection engine and analyzing your media…"):
            result = detect_media(model_id, uploaded_file)
        prediction = result["prediction"]
        confidence = float(result["confidence"])
        ai_score = max(0.0, min(100.0, float(result["ai_probability"])))
        real_score = max(0.0, min(100.0, float(result["real_probability"])))
        is_ai = prediction == "AI"
        label = "AI-generated content" if is_ai else "Likely authentic content"
        css_class = "ai" if is_ai else "real"
        st.markdown(f'<div class="result-panel {css_class}"><div class="result-eyebrow">✦ &nbsp; Analysis complete</div><div class="result-title">{label}</div><div class="result-desc">Selected model: {html.escape(selected_model)} &nbsp; · &nbsp; Predicted class confidence: {confidence:.2f}%</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("AI-generated score", f"{ai_score:.2f}%")
        with c2:
            st.metric("Real-content score", f"{real_score:.2f}%")
        st.markdown(f'<div class="score-row"><span>AI-generated</span><span>{ai_score:.2f}%</span></div><div class="score-track"><div class="score-fill" style="width:{ai_score:.2f}%"></div></div><div class="score-row"><span>Real content</span><span>{real_score:.2f}%</span></div><div class="score-track"><div class="score-fill real" style="width:{real_score:.2f}%"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="score-foot">Research-use classification only. Scores are model softmax outputs, not calibrated probabilities of correctness or proof of authenticity. Results can vary by model and media quality.</div>', unsafe_allow_html=True)
    except Exception as error:
        st.error(f"Detection failed: {error}")

st.markdown('<div class="footer">✦ FAKTRION033 &nbsp; · &nbsp; Visual Authenticity Lab<br>Deep learning research prototype for synthetic and real media classification</div>', unsafe_allow_html=True)
