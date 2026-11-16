
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights
)

MODEL_CONFIG = {
    "photo_clip": 512,
    "photo_dinov2": 384,
    "photo_efficientnet": 1280,
    "photo_convnext": 768,
    "photo_swin": 768,
}

class PhotoClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


def load_photo_model(model_name, models_dir, device="cpu"):
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown photo model: {model_name}")

    feature_dim = MODEL_CONFIG[model_name]

    if model_name == "photo_clip":
        import open_clip

        encoder, _, preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32-quickgelu",
                pretrained="openai"
            )
        )

        extract = lambda x: encoder.encode_image(x)

    elif model_name == "photo_dinov2":
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True
        )

        preprocess = transforms.Compose([
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        extract = lambda x: encoder(x)

    elif model_name == "photo_efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT
        encoder = efficientnet_b0(weights=weights)
        encoder.classifier = nn.Identity()
        preprocess = weights.transforms()
        extract = lambda x: encoder(x)

    elif model_name == "photo_convnext":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        encoder = convnext_tiny(weights=weights)
        encoder.classifier = nn.Flatten(1)
        preprocess = weights.transforms()
        extract = lambda x: encoder(x)

    else:
        weights = Swin_T_Weights.DEFAULT
        encoder = swin_t(weights=weights)
        encoder.head = nn.Identity()
        preprocess = weights.transforms()
        extract = lambda x: encoder(x)

    encoder = encoder.to(device).eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    classifier = PhotoClassifier(feature_dim).to(device)

    checkpoint_path = (
        models_dir / model_name / "best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    classifier.eval()

    return encoder, classifier, preprocess, extract


@torch.inference_mode()
def predict_photo(
    image,
    encoder,
    classifier,
    preprocess,
    extract,
    device="cpu"
):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected PIL Image")

    image = ImageOps.exif_transpose(image).convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(device)

    features = extract(tensor).float()
    features = F.normalize(features, dim=-1)

    logits = classifier(features)

    probabilities = torch.softmax(
        logits, dim=1
    )[0].cpu().tolist()

    predicted_class = int(
        probabilities[1] > probabilities[0]
    )

    return {
        "prediction": "Real" if predicted_class else "AI",
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
