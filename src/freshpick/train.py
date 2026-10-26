import argparse
import hashlib
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
from .predict import _pad


class ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_dir: Path, training: bool):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = data_dir
        operations = [transforms.Lambda(_pad)]
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
            tensor = self.transform(image)
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
    parser.add_argument("--init-checkpoint", type=Path, help="Fine-tune a checkpoint on its original split")
    parser.add_argument("--cosine", action="store_true", help="Decay learning rate over this run")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if args.init_checkpoint and args.init_checkpoint.resolve() == (args.output_dir / "best.pt").resolve():
        parser.error("Use a separate output directory to preserve the initial checkpoint")
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
    weights = None if args.no_pretrained or args.init_checkpoint else models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    parent_sha256 = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial["class_names"] != class_names or initial["split_sha256"] != split_digest(splits):
            raise ValueError("Fine-tuning requires matching class order and original split")
        model.load_state_dict(initial["state_dict"])
        parent_sha256 = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    model.to(device)
    train_counts = np.bincount(splits[splits.split == "train"].class_id,
                               minlength=len(class_names))
    class_weights = torch.tensor(len(train_counts) / train_counts, dtype=torch.float32,
                                 device=device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1) if args.cosine else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best_score, stale = -1.0, 0
    checkpoint = args.output_dir / "best.pt"
    baseline_validation = None
    if args.init_checkpoint:
        baseline_validation, _, _ = run_epoch(model, loaders["val"], eval_criterion,
                                               None, scaler, device, args.amp)
        best_score = baseline_validation["macro_f1"]
        torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                    "epoch": 0, "split_sha256": split_digest(splits),
                    "validation": baseline_validation, "parent_sha256": parent_sha256}, checkpoint)
        print(json.dumps({"baseline_validation": baseline_validation}), flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, loaders["train"], train_criterion,
                                        optimizer, scaler, device, args.amp)
        val_metrics, _, _ = run_epoch(model, loaders["val"], eval_criterion,
                                      None, scaler, device, args.amp)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score, stale = val_metrics["macro_f1"], 0
            torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                        "epoch": epoch, "split_sha256": split_digest(splits),
                        "validation": val_metrics, "parent_sha256": parent_sha256}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
        if scheduler:
            scheduler.step()

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
              "validation": saved.get("validation"), "baseline_validation": baseline_validation,
              "parent_sha256": parent_sha256, "epochs_completed": len(history),
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"test": test_metrics, "gpu": environment["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
