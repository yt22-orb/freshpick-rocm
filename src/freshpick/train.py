import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from .data import load_manifest, make_splits, split_digest


class ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_dir: Path, training: bool):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = data_dir
        operations = []
        if training:
            operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        operations += [transforms.ToTensor(), transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )]
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.data_dir / row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row.class_id)


def run_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    labels, predictions = [], []
    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * len(targets)
        total_items += len(targets)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": total_loss / total_items,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }, labels, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mobilenetv3-small"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch size, and patience must be positive")

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm-enabled PyTorch with an available AMD GPU is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = load_manifest(args.data_dir)
    splits = make_splits(manifest, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.output_dir / "splits.csv", index=False)
    class_names = (manifest[["class_id", "class_name"]].drop_duplicates()
                   .sort_values("class_id").class_name.tolist())
    datasets = {
        name: ImageDataset(splits[splits.split == name], args.data_dir, name == "train")
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                         num_workers=args.workers, pin_memory=True)
        for name, dataset in datasets.items()
    }
    weights = None if args.no_pretrained else models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.to(device)
    train_counts = np.bincount(splits[splits.split == "train"].class_id,
                               minlength=len(class_names))
    class_weights = torch.tensor(len(train_counts) / train_counts, dtype=torch.float32,
                                 device=device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best_score, stale = -1.0, 0
    checkpoint = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, loaders["train"], train_criterion,
                                        optimizer, scaler, device, args.amp)
        val_metrics, _, _ = run_epoch(model, loaders["val"], eval_criterion,
                                      None, scaler, device, args.amp)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score, stale = val_metrics["macro_f1"], 0
            torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                        "epoch": epoch, "split_sha256": split_digest(splits)}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    test_metrics, actual, predicted = run_epoch(model, loaders["test"], eval_criterion,
                                                 None, scaler, device, args.amp)
    report = classification_report(actual, predicted, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(class_names))))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        args.output_dir / "confusion_matrix.csv")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": torch.__version__, "hip": torch.version.hip,
                   "gpu": torch.cuda.get_device_name(0), "seed": args.seed}
    result = {"test": test_metrics, "per_class": report, "best_epoch": saved["epoch"],
              "split_sha256": saved["split_sha256"], "environment": environment,
              "configuration": vars(args) | {"data_dir": str(args.data_dir),
                                          "output_dir": str(args.output_dir)}}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"test": test_metrics, "gpu": environment["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
