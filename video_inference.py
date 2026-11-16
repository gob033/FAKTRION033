
import os
import cv2
import tempfile
import time
import logging
import numpy as np
from pathlib import Path
from PIL import Image


def extract_video_frames(video_path, num_frames=8, size=224):
    """
    Extract uniformly sampled RGB frames from a video.

    Returns:
        frames: list of PIL RGB images
        indices: selected original frame indices
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            raise ValueError(
                "Video frame count is unavailable."
            )

        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames
        ).astype(int)

        frames = []
        last_valid_frame = None

        for index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            success, frame = cap.read()

            if not success or frame is None:
                if last_valid_frame is None:
                    raise ValueError(
                        f"Could not decode frame {index}"
                    )
                frame = last_valid_frame.copy()

            frame = cv2.resize(
                frame,
                (size, size),
                interpolation=cv2.INTER_LINEAR
            )

            last_valid_frame = frame.copy()

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                Image.fromarray(rgb)
            )

        return frames, indices.tolist()

    finally:
        cap.release()


def extract_uploaded_video(uploaded_file):
    """
    Accept a Streamlit UploadedFile object.
    Extract frames and clean up temporary video.
    """

    suffix = Path(
        getattr(uploaded_file, "name", "video.mp4")
    ).suffix.lower()

    if suffix not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        raise ValueError("Unsupported video format.")

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp:

        temp_path = temp.name
        temp.write(uploaded_file.getvalue())

    try:
        return extract_video_frames(temp_path)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# VIDEO CLIP ViT-B/32 TEMPORAL MEAN POOLING
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoCLIPClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


def load_video_clip(models_dir, device="cpu"):

    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    encoder = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    encoder = encoder.to(device).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    classifier = VideoCLIPClassifier().to(device)

    checkpoint_path = (
        Path(models_dir)
        / "video_clip"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    classifier.load_state_dict(
        state,
        strict=True
    )

    classifier.eval()

    return encoder, processor, classifier


@torch.inference_mode()
def predict_video_clip(
    frames,
    encoder,
    processor,
    classifier,
    device="cpu"
):

    if len(frames) != 8:
        raise ValueError(
            "Video CLIP requires exactly 8 frames."
        )

    inputs = processor(
        images=frames,
        return_tensors="pt"
    )

    pixels = inputs["pixel_values"].to(device)

    features = encoder.get_image_features(
        pixel_values=pixels
    )

    # Support tensor and newer model output formats
    if not isinstance(features, torch.Tensor):
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "image_embeds"):
            features = features.image_embeds
        else:
            raise TypeError(
                "Unsupported CLIP feature output."
            )

    features = F.normalize(
        features.float(),
        dim=-1
    )

    video_feature = features.mean(
        dim=0,
        keepdim=True
    )

    video_feature = F.normalize(
        video_feature,
        dim=-1
    )

    logits = classifier(video_feature)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0].cpu().tolist()

    predicted = int(
        probabilities[1] > probabilities[0]
    )

    return {
        "prediction": "Real" if predicted else "AI",
        "confidence": round(
            max(probabilities) * 100, 2
        ),
        "ai_probability": round(
            probabilities[0] * 100, 2
        ),
        "real_probability": round(
            probabilities[1] * 100, 2
        )
    }


# =====================================================
# DINOV2 ViT-S/14 + LEARNABLE TEMPORAL ATTENTION
# =====================================================

class DINOv2TemporalAttention(nn.Module):

    def __init__(self, input_dim=384):
        super().__init__()

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU()
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        # Input: [batch, 8, 384]
        x = self.projection(x)

        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)

        pooled = (weights * x).sum(dim=1)

        return self.classifier(pooled)


def load_video_dinov2(models_dir, device="cpu"):

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    encoder = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True
    )

    encoder = encoder.to(device).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    # Original video training preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    classifier = DINOv2TemporalAttention().to(device)

    checkpoint_path = (
        Path(models_dir)
        / "video_dinov2"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    classifier.load_state_dict(
        state,
        strict=True
    )

    classifier.eval()

    return encoder, preprocess, classifier


@torch.inference_mode()
def predict_video_dinov2(
    frames,
    encoder,
    preprocess,
    classifier,
    device="cpu"
):

    if len(frames) != 8:
        raise ValueError(
            "DINOv2 video model requires exactly 8 frames."
        )

    pixels = torch.stack([
        preprocess(frame.convert("RGB"))
        for frame in frames
    ]).to(device)

    # Match original training's CUDA autocast behavior
    if str(device).startswith("cuda"):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            features = encoder(pixels)
    else:
        features = encoder(pixels)

    features = features.float()

    if features.shape != (8, 384):
        raise ValueError(
            f"Unexpected DINOv2 features: {features.shape}"
        )

    # Temporal model expects [batch, frames, features]
    features = features.unsqueeze(0)

    logits = classifier(features)

    probabilities = torch.softmax(
        logits, dim=1
    )[0].cpu().tolist()

    predicted = int(
        probabilities[1] > probabilities[0]
    )

    return {
        "prediction": "Real" if predicted else "AI",
        "confidence": round(
            max(probabilities) * 100, 2
        ),
        "ai_probability": round(
            probabilities[0] * 100, 2
        ),
        "real_probability": round(
            probabilities[1] * 100, 2
        )
    }


# =====================================================
# SWIN3D-T VIDEO INFERENCE
# =====================================================

class Swin3DClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


def load_video_swin(models_dir, device="cpu"):

    from torchvision.models.video import (
        swin3d_t,
        Swin3D_T_Weights
    )

    weights = Swin3D_T_Weights.KINETICS400_V1

    encoder = swin3d_t(weights=weights)

    encoder.head = nn.Identity()

    encoder = encoder.to(device).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    classifier = Swin3DClassifier().to(device)

    checkpoint_path = (
        Path(models_dir)
        / "video_swin"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    classifier.load_state_dict(
        state,
        strict=True
    )

    classifier.eval()

    return encoder, classifier


@torch.inference_mode()
def predict_video_swin(
    frames,
    encoder,
    classifier,
    device="cpu"
):

    if len(frames) != 8:
        raise ValueError(
            "Swin3D-T requires 8 sampled frames."
        )

    # Expand 8 frames to 16 frames
    indices = np.linspace(
        0, 7, 16
    ).round().astype(int)

    selected = [
        np.asarray(frames[i].convert("RGB"))
        for i in indices
    ]

    # [T, H, W, C]
    video = np.stack(selected).astype(np.float32) / 255.0

    # Kinetics-400 normalization
    mean = np.array(
        [0.43216, 0.394666, 0.37645],
        dtype=np.float32
    )

    std = np.array(
        [0.22803, 0.22145, 0.216989],
        dtype=np.float32
    )

    video = (video - mean) / std

    # [T, H, W, C] -> [1, C, T, H, W]
    tensor = torch.from_numpy(
        video
    ).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    features = encoder(tensor).float()

    if features.shape != (1, 768):
        raise ValueError(
            f"Unexpected Swin3D features: {features.shape}"
        )

    logits = classifier(features)

    probabilities = torch.softmax(
        logits, dim=1
    )[0].cpu().tolist()

    predicted = int(
        probabilities[1] > probabilities[0]
    )

    return {
        "prediction": "Real" if predicted else "AI",
        "confidence": round(
            max(probabilities) * 100, 2
        ),
        "ai_probability": round(
            probabilities[0] * 100, 2
        ),
        "real_probability": round(
            probabilities[1] * 100, 2
        )
    }


# =====================================================
# UNITE-INSPIRED SIGLIP TEMPORAL TRANSFORMER
# =====================================================

class UNITEInspiredTransformer(nn.Module):

    def __init__(self, input_dim=1152, num_frames=8):
        super().__init__()

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU()
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 256)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.attention_heads = nn.Linear(256, 4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        # x: [batch, 8, 1152]
        x = self.projection(x)

        x = x + self.pos_embedding

        x = self.transformer(x)

        # Four-head temporal attention
        scores = self.attention_heads(x)

        weights = torch.softmax(scores, dim=1)

        # [batch, 4, 256]
        pooled_heads = torch.einsum(
            "bth,btd->bhd",
            weights,
            x
        )

        # Average the four attention heads
        pooled = pooled_heads.mean(dim=1)

        return self.classifier(pooled)


def load_video_unite(models_dir, device="cpu"):

    from transformers import AutoProcessor, AutoModel

    _unite_start = time.perf_counter()

    model_id = "google/siglip-so400m-patch14-384"

    processor = AutoProcessor.from_pretrained(
        model_id
    )

    _unite_processor_done = time.perf_counter()

    encoder = AutoModel.from_pretrained(
        model_id,
        torch_dtype=(
            torch.float16
            if str(device).startswith("cuda")
            else torch.float32
        )
    )

    _unite_backbone_done = time.perf_counter()

    encoder = encoder.to(device).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    classifier = UNITEInspiredTransformer().to(device)

    checkpoint_path = (
        Path(models_dir)
        / "video_unite"
        / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    classifier.load_state_dict(
        state,
        strict=True
    )

    classifier.eval()

    _unite_end = time.perf_counter()
    logging.warning(
        "[FAKTRION UNITE TIMING] | processor_load=%.2fs | backbone_load=%.2fs | "
        "classifier_load=%.2fs | total_load=%.2fs",
        _unite_processor_done - _unite_start,
        _unite_backbone_done - _unite_processor_done,
        _unite_end - _unite_backbone_done,
        _unite_end - _unite_start,
    )

    return encoder, processor, classifier


@torch.inference_mode()
def predict_video_unite(
    frames,
    encoder,
    processor,
    classifier,
    device="cpu"
):

    if len(frames) != 8:
        raise ValueError(
            "UNITE-inspired model requires 8 frames."
        )

    _unite_start = time.perf_counter()

    inputs = processor(
        images=frames,
        return_tensors="pt"
    )

    pixels = inputs["pixel_values"].to(device)

    if str(device).startswith("cuda"):
        pixels = pixels.half()

    _unite_preprocess_done = time.perf_counter()

    # Step 161: CPU micro-batch experiment. All 8 original frames and
    # pretrained/fine-tuned weights are preserved. Set UNITE_CPU_BATCH_SIZE=8
    # to restore the original single-batch execution without another edit.
    batch_size = 8 if str(device).startswith("cuda") else int(
        os.environ.get("UNITE_CPU_BATCH_SIZE", "2")
    )
    if batch_size not in (1, 2, 4, 8):
        raise ValueError("UNITE_CPU_BATCH_SIZE must be 1, 2, 4, or 8")

    pooled_parts = []
    for start in range(0, pixels.shape[0], batch_size):
        batch_start = time.perf_counter()
        outputs = encoder.vision_model(
            pixel_values=pixels[start:start + batch_size]
        )
        pooled_parts.append(outputs.pooler_output.float())
        logging.warning(
            "[FAKTRION UNITE BATCH] | frames=%d-%d | seconds=%.2f",
            start + 1,
            min(start + batch_size, pixels.shape[0]),
            time.perf_counter() - batch_start,
        )

    _unite_vision_done = time.perf_counter()

    features = torch.cat(pooled_parts, dim=0)

    if features.shape != (8, 1152):
        raise ValueError(
            f"Unexpected SigLIP features: {features.shape}"
        )

    features = features.unsqueeze(0)

    logits = classifier(features)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0].cpu().tolist()

    predicted = int(
        probabilities[1] > probabilities[0]
    )

    _unite_end = time.perf_counter()
    logging.warning(
        "[FAKTRION UNITE TIMING] | preprocess=%.2fs | vision_inference=%.2fs | "
        "temporal_head=%.2fs | total_predict=%.2fs",
        _unite_preprocess_done - _unite_start,
        _unite_vision_done - _unite_preprocess_done,
        _unite_end - _unite_vision_done,
        _unite_end - _unite_start,
    )

    return {
        "prediction": "Real" if predicted else "AI",
        "confidence": round(
            max(probabilities) * 100, 2
        ),
        "ai_probability": round(
            probabilities[0] * 100, 2
        ),
        "real_probability": round(
            probabilities[1] * 100, 2
        )
    }



# =====================================================
# VIDEOMAE-BASE PARTIALLY FINE-TUNED INFERENCE
# =====================================================

class FineTuneVideoMAE(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, pixel_values):

        outputs = self.backbone(
            pixel_values=pixel_values
        )

        features = outputs.last_hidden_state.mean(dim=1)

        return self.classifier(features)


def load_video_videomae(models_dir, device="cpu"):

    from transformers import (
        VideoMAEModel,
        VideoMAEConfig,
        VideoMAEImageProcessor
    )

    model_id = "MCG-NJU/videomae-base"

    processor = VideoMAEImageProcessor.from_pretrained(
        model_id
    )

    # The fine-tuned checkpoint contains the complete backbone weights.
    # Avoid downloading an additional ~330 MB pretrained backbone.
    config = VideoMAEConfig.from_pretrained(model_id)
    backbone = VideoMAEModel(config)

    model = FineTuneVideoMAE(backbone)

    checkpoint_candidates = [
        Path(models_dir) / "video_videomae" / "best_model.pth",
        Path(models_dir).parent.parent
        / "video_runs"
        / "videomae_base_finetuned"
        / "best_model.pth"
    ]

    checkpoint_path = next(
        (p for p in checkpoint_candidates if p.is_file()),
        None
    )

    if checkpoint_path is None:
        # Hugging Face Hub caches this large checkpoint across app reruns
        # while the server's cache remains available.
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = Path(hf_hub_download(
                repo_id="gobeshonagar/FAKTRION033-VideoMAE",
                filename="best_model.pth",
                repo_type="model",
            ))
        except Exception as exc:
            raise RuntimeError(
                "VideoMAE checkpoint download failed. "
                "Check Hugging Face connectivity and available disk space."
            ) from exc

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    # Strict loading ensures all fine-tuned weights match.
    model.load_state_dict(state, strict=True)

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, processor


@torch.inference_mode()
def predict_video_videomae(
    frames,
    model,
    processor,
    device="cpu"
):

    if len(frames) != 8:
        raise ValueError(
            "VideoMAE requires exactly 8 sampled frames."
        )

    # Original training: resample 8 frames to 16.
    indices = np.linspace(
        0, 7, 16
    ).round().astype(int)

    video_frames = [
        frames[i].convert("RGB")
        for i in indices
    ]

    inputs = processor(
        images=video_frames,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    if pixel_values.shape[1] != 16:
        raise ValueError(
            f"Unexpected VideoMAE input: {pixel_values.shape}"
        )

    logits = model(pixel_values)

    probabilities = torch.softmax(
        logits, dim=1
    )[0].cpu().tolist()

    predicted = int(
        probabilities[1] > probabilities[0]
    )

    return {
        "prediction": "Real" if predicted else "AI",
        "confidence": round(
            max(probabilities) * 100, 2
        ),
        "ai_probability": round(
            probabilities[0] * 100, 2
        ),
        "real_probability": round(
            probabilities[1] * 100, 2
        )
    }
