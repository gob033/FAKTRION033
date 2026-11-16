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
