"""ROCm inference and FreshPick response adapter."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


def _pad(image):
    content = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))
    canvas.paste(content, ((224 - content.width) // 2, (224 - content.height) // 2))
    return canvas


def image_transform():
    return transforms.Compose([transforms.Lambda(_pad), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def response(names, probabilities, version):
    stages = {"fresh": "Fresh", "semi_fresh": "Semi-fresh", "rotten": "Rotten"}
    totals = dict.fromkeys(stages, 0.0)
    parsed = []
    for name, probability in zip(names, probabilities):
        stage = next((s for s in stages if name.startswith(s + "_")), None)
        if stage is None:
            raise ValueError(f"Unsupported class: {name}")
        produce = name[len(stage) + 1:].replace(" ", "_")
        parsed.append((stage, produce))
        totals[stage] += float(probability)
    total = sum(totals.values())
    totals = {stage: value / total for stage, value in totals.items()}
    index = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
    stage, produce = parsed[index]
    return {"status": "ok", "produce": produce, "produce_label": ("Bitter melon" if produce == "bittermelon" else produce.replace("_", " ").title()),
        "freshness": stage, "freshness_label": stages[stage],
        "class_confidence": float(probabilities[index]), "freshness_confidence": totals[stage],
        "freshness_probabilities": totals,
        "class_probabilities": {n: float(p) for n, p in zip(names, probabilities)},
        "model_version": version}


class Predictor:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Model checkpoint unavailable; train the model first")
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("ROCm-enabled PyTorch with an available AMD GPU is required")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.names = saved["class_names"]
        self.version = "mobilenetv3-small-" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, len(self.names))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to("cuda").eval()
        self.transform = image_transform()

    def predict(self, image):
        with torch.inference_mode():
            probabilities = self.model(self.transform(image).unsqueeze(0).to("cuda")).softmax(1)[0].cpu().tolist()
        return response(self.names, probabilities, self.version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mobilenetv3-finetuned/best.pt"))
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint)
    with Image.open(args.image) as image:
        print(json.dumps(predictor.predict(image)))


if __name__ == "__main__":
    main()
