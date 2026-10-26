import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mobilenetv3-small/best.pt"))
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    names = saved["class_names"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(names))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: _pad(image)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with Image.open(args.image) as image, torch.no_grad():
        probabilities = model(transform(image.convert("RGB")).unsqueeze(0)).softmax(1)[0]
    index = int(probabilities.argmax())
    print(json.dumps({"class_name": names[index], "confidence": float(probabilities[index])}))


def _pad(image: Image.Image) -> Image.Image:
    from PIL import ImageOps

    content = ImageOps.contain(ImageOps.exif_transpose(image), (224, 224), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))
    canvas.paste(content, ((224 - content.width) // 2, (224 - content.height) // 2))
    return canvas


if __name__ == "__main__":
    main()
