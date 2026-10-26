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


def _class_parts(names):
    stages = {"fresh": "Fresh", "semi_fresh": "Semi-fresh", "rotten": "Rotten"}
    parsed = []
    for name in names:
        stage = next((s for s in stages if name.startswith(s + "_")), None)
        if stage is None:
            raise ValueError(f"Unsupported class: {name}")
        produce = name[len(stage) + 1:].replace(" ", "_")
        parsed.append((stage, produce))
    return stages, parsed


def response(names, probabilities, version, abstention=None):
    """Build a response from marginal produce/freshness probabilities.

    A joint-class winner can be wrong even when the model assigns the largest total
    probability to the correct produce across its three freshness stages. Marginal
    predictions use all of that evidence. An optional checkpoint-calibrated produce
    threshold turns low-confidence inputs into an uncertain result.
    """
    if len(names) != len(probabilities) or not names:
        raise ValueError("Class names and probabilities must be non-empty and equal length")
    stages, parsed = _class_parts(names)
    freshness_totals = dict.fromkeys(stages, 0.0)
    produce_totals = {}
    for (stage, produce), probability in zip(parsed, probabilities):
        probability = float(probability)
        freshness_totals[stage] += probability
        produce_totals[produce] = produce_totals.get(produce, 0.0) + probability
    total = sum(freshness_totals.values())
    if total <= 0:
        raise ValueError("Class probabilities must have a positive total")
    freshness_totals = {key: value / total for key, value in freshness_totals.items()}
    produce_totals = {key: value / total for key, value in produce_totals.items()}
    stage = max(freshness_totals, key=freshness_totals.get)
    produce = max(produce_totals, key=produce_totals.get)
    joint_index = max(range(len(probabilities)), key=lambda i: float(probabilities[i]))
    threshold = float((abstention or {}).get("produce_confidence_threshold", 0.0))
    if not 0 <= threshold <= 1:
        raise ValueError("Produce confidence threshold must be in [0, 1]")
    status = "uncertain" if produce_totals[produce] < threshold else "ok"
    return {"status": status, "produce": produce, "produce_label": ("Bitter melon" if produce == "bittermelon" else produce.replace("_", " ").title()),
        "freshness": stage, "freshness_label": stages[stage],
        "class_confidence": float(probabilities[joint_index]),
        "produce_confidence": produce_totals[produce],
        "freshness_confidence": freshness_totals[stage],
        "produce_probabilities": produce_totals,
        "freshness_probabilities": freshness_totals,
        "class_probabilities": {n: float(p) for n, p in zip(names, probabilities)},
        "model_version": version}


def combine_predictions(freshness, identity):
    """Keep independent identity output distinct from the 24-class diagnostics."""
    result = dict(freshness)
    result['freshness_model_produce'] = freshness['produce']
    result['freshness_available'] = (identity['status'] == 'ok' and freshness['status'] == 'ok'
                                     and identity['produce'] == freshness['produce'])
    result.update(identity)
    result['pipeline'] = 'separate-eight-produce-identifier-and-24-class-freshness'
    return result


class Predictor:
    def __init__(self, checkpoint, identifier_checkpoint=None):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Model checkpoint unavailable; train the model first")
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("ROCm-enabled PyTorch with an available AMD GPU is required")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.names = saved["class_names"]
        self.abstention = saved.get("abstention")
        self.version = "mobilenetv3-small-" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, len(self.names))
        self.model.load_state_dict(saved["state_dict"])
        self.model.to("cuda").eval()
        self.transform = image_transform()
        self.identifier = None
        if identifier_checkpoint:
            from .identifier import Identifier
            self.identifier = Identifier(identifier_checkpoint)

    def predict(self, image):
        with torch.inference_mode():
            probabilities = self.model(self.transform(image).unsqueeze(0).to("cuda")).softmax(1)[0].cpu().tolist()
        result = response(self.names, probabilities, self.version, self.abstention)
        result['freshness_available'] = result['status'] == 'ok'
        result['freshness_model_produce'] = result['produce']
        if self.identifier is not None:
            result = combine_predictions(result, self.identifier.predict(image))
        else:
            result['pipeline'] = '24-class-baseline'
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mobilenetv3-finetuned/best.pt"))
    parser.add_argument("--identifier-checkpoint", type=Path, default=Path("runs/identifier/best.pt"))
    parser.add_argument("--baseline", action="store_true", help="Evaluate only the original 24-class classifier")
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, None if args.baseline else args.identifier_checkpoint)
    with Image.open(args.image) as image:
        print(json.dumps(predictor.predict(image)))


if __name__ == "__main__":
    main()
