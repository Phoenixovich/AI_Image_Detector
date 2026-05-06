from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import CNN
from utils import DEVICE, IMAGE_EXTENSIONS, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, OUTPUT_DIR, make_eval_transform


DEFAULT_MODEL_PATH = OUTPUT_DIR / "models" / "model_final.pth"
DEFAULT_THRESHOLD_PATH = OUTPUT_DIR / "models" / "model_final_threshold.json"
DEFAULT_EXPLANATIONS_DIR = OUTPUT_DIR / "explanations" / "predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether one image is real or AI-generated.")
    parser.add_argument("image_path", type=Path, help="Path to the image to classify.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained final model.")
    parser.add_argument("--threshold-path", type=Path, default=DEFAULT_THRESHOLD_PATH, help="Path to threshold JSON saved by training.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold (if provided, ignores threshold file).")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (flip + scale views).")
    parser.add_argument("--explain-regions", action="store_true", help="Save a Grad-CAM heatmap showing the region that influenced the prediction.")
    parser.add_argument("--explanations-dir", type=Path, default=DEFAULT_EXPLANATIONS_DIR, help="Directory for Grad-CAM explanation image.")
    parser.add_argument(
        "--heatmap-color",
        choices=["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "white"],
        default="red",
        help="Color used for Grad-CAM heatmap overlay.",
    )
    return parser.parse_args()


def load_threshold(threshold_path: Path) -> float:
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return float(data.get("threshold", 0.5))
    return 0.5


def load_model(model_path: Path) -> tuple[nn.Module, float]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CNN().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    else:
        model.load_state_dict(checkpoint)
        checkpoint_threshold = 0.5

    model.eval()
    return model, checkpoint_threshold


def tta_logits(model: nn.Module, image: torch.Tensor, use_tta: bool) -> torch.Tensor:
    logits: list[torch.Tensor] = [model(image)]
    if not use_tta:
        return logits[0]

    logits.append(model(torch.flip(image, dims=[3])))

    upscaled = F.interpolate(image, scale_factor=1.10, mode="bilinear", align_corners=False)
    upscaled = F.interpolate(upscaled, size=image.shape[-2:], mode="bilinear", align_corners=False)
    logits.append(model(upscaled))

    return torch.stack(logits, dim=0).mean(dim=0)


def area_name(x: int, y: int, width: int, height: int) -> str:
    horizontal = "left" if x < width / 3 else "right" if x >= 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y >= 2 * height / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "middle":
        return horizontal
    return f"{vertical}-{horizontal}"


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    mean = NORMALIZE_MEAN.to(image.device)
    std = NORMALIZE_STD.to(image.device)
    denormalized = (image.detach() * std + mean).clamp(0.0, 1.0)
    array = (denormalized.cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def heatmap_rgb(color_name: str) -> tuple[int, int, int]:
    colors = {
        "red": (255, 0, 0),
        "orange": (255, 120, 0),
        "yellow": (255, 220, 0),
        "green": (0, 220, 80),
        "cyan": (0, 210, 255),
        "blue": (50, 120, 255),
        "magenta": (255, 0, 220),
        "white": (255, 255, 255),
    }
    return colors[color_name]


def make_heatmap_overlay(image: torch.Tensor, heatmap: torch.Tensor, color_name: str) -> Image.Image:
    base = tensor_to_pil_image(image).convert("RGBA")
    heatmap = heatmap.detach().cpu().clamp(0.0, 1.0)
    red_value, green_value, blue_value = heatmap_rgb(color_name)
    alpha = (heatmap * 150.0).byte().numpy()
    red = (heatmap * red_value).byte().numpy()
    green = (heatmap * green_value).byte().numpy()
    blue = (heatmap * blue_value).byte().numpy()
    overlay = Image.merge(
        "RGBA",
        (
            Image.fromarray(red, mode="L"),
            Image.fromarray(green, mode="L"),
            Image.fromarray(blue, mode="L"),
            Image.fromarray(alpha, mode="L"),
        ),
    )
    return Image.alpha_composite(base, overlay).convert("RGB")


def gradcam_explanation(
    model: CNN,
    image: torch.Tensor,
    prediction: int,
    image_path: Path,
    explanations_dir: Path,
    heatmap_color: str,
) -> dict[str, str]:
    target_layer = model.rgb_branch.features[-3]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: nn.Module, _grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        input_image = image.unsqueeze(0).to(next(model.parameters()).device)
        logits = model(input_image)
        target = logits.view(-1)[0] if prediction == 1 else -logits.view(-1)[0]
        target.backward()

        if not activations or not gradients:
            return {"decision_region": "", "region_x": "", "region_y": "", "heatmap_path": ""}

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        flat_index = int(torch.argmax(heatmap).item())
        y, x = divmod(flat_index, heatmap.shape[1])
        region = area_name(x, y, heatmap.shape[1], heatmap.shape[0])

        explanations_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
        heatmap_path = explanations_dir / f"{safe_stem}_gradcam.jpg"
        make_heatmap_overlay(image, heatmap, heatmap_color).save(heatmap_path, quality=95)

        return {
            "decision_region": region,
            "region_x": str(x),
            "region_y": str(y),
            "heatmap_path": str(heatmap_path),
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()


def predict_image(model: nn.Module, image_path: Path, threshold: float, use_tta: bool) -> tuple[float, int, str, torch.Tensor]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    transform = make_eval_transform()
    with Image.open(image_path) as image:
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = tta_logits(model, tensor, use_tta=use_tta)
        probability = float(torch.sigmoid(logits).view(-1).item())

    predicted_label = 1 if probability >= threshold else 0
    predicted_outcome = "ai" if predicted_label == 1 else "real"
    return probability, predicted_label, predicted_outcome, tensor.squeeze(0).detach().cpu()


def main() -> None:
    args = parse_args()

    model, checkpoint_threshold = load_model(args.model_path)
    threshold = args.threshold if args.threshold is not None else load_threshold(args.threshold_path)
    if args.threshold is None and checkpoint_threshold != 0.5 and not args.threshold_path.exists():
        threshold = checkpoint_threshold

    probability, predicted_label, predicted_outcome, image_tensor = predict_image(
        model,
        args.image_path,
        threshold=threshold,
        use_tta=args.tta,
    )

    explanation = {"decision_region": "", "region_x": "", "region_y": "", "heatmap_path": ""}
    if args.explain_regions:
        explanation = gradcam_explanation(
            model,
            image_tensor,
            predicted_label,
            args.image_path,
            args.explanations_dir,
            args.heatmap_color,
        )

    print(f"Image: {args.image_path}")
    print(f"Device: {DEVICE}")
    print(f"TTA enabled: {args.tta}")
    print(f"Threshold: {threshold:.2f}")
    print(f"AI probability: {probability:.4f}")
    print(f"Predicted label: {predicted_label}")
    print(f"Prediction: {predicted_outcome}")
    if args.explain_regions:
        print(f"Decision region: {explanation['decision_region']}")
        print(f"Strongest region point: x={explanation['region_x']} y={explanation['region_y']}")
        print(f"Grad-CAM heatmap: {explanation['heatmap_path']}")


if __name__ == "__main__":
    main()
