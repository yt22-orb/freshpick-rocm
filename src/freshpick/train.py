import argparse
import hashlib
import itertools
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

from .data import load_manifest, make_splits, split_digest
from .predict import _class_parts, _pad


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def training_transform(training: bool):
    operations = [transforms.Lambda(_pad)]
    if training:
        operations += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12)
            ], p=0.6),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.2),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )]
    return transforms.Compose(operations)


class ImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_dir: Path, training: bool):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = training_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.data_dir / row.image_path) as image:
            tensor = self.transform(image)
        return tensor, int(row.class_id)


class UnlabeledImageDataset(Dataset):
    """Images of unsupported items used for outlier exposure and calibration."""

    def __init__(self, root: Path, training: bool):
        self.paths = sorted(path for path in root.rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No supported images found under OOD directory: {root}")
        self.transform = training_transform(training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def hierarchy_indices(class_names):
    stages, parsed = _class_parts(class_names)
    produce_names = sorted({produce for _, produce in parsed})
    produce_lookup = {name: index for index, name in enumerate(produce_names)}
    stage_lookup = {name: index for index, name in enumerate(stages)}
    return {
        "produce_names": produce_names,
        "stage_names": list(stages),
        "produce": [produce_lookup[produce] for _, produce in parsed],
        "freshness": [stage_lookup[stage] for stage, _ in parsed],
    }


class HierarchicalLoss(nn.Module):
    """Joint-class CE plus marginal produce and freshness supervision."""

    def __init__(self, class_names, class_weights=None, label_smoothing=0.05,
                 produce_weight=0.35, freshness_weight=0.15):
        super().__init__()
        hierarchy = hierarchy_indices(class_names)
        self.register_buffer("produce_by_class", torch.tensor(hierarchy["produce"]))
        self.register_buffer("freshness_by_class", torch.tensor(hierarchy["freshness"]))
        self.joint = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=label_smoothing)
        self.produce_weight = produce_weight
        self.freshness_weight = freshness_weight

    @staticmethod
    def _marginal_nll(logits, targets, mapping):
        log_probabilities = logits.log_softmax(1)
        groups = torch.stack([
            torch.logsumexp(log_probabilities[:, mapping == group], dim=1)
            for group in range(int(mapping.max().item()) + 1)
        ], dim=1)
        return F.nll_loss(groups, mapping[targets])

    def forward(self, logits, targets):
        loss = self.joint(logits, targets)
        loss += self.produce_weight * self._marginal_nll(
            logits, targets, self.produce_by_class)
        loss += self.freshness_weight * self._marginal_nll(
            logits, targets, self.freshness_by_class)
        return loss


def classification_metrics(labels, predictions, hierarchy, produce_predictions,
                           freshness_predictions):
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    for key, marginal_predictions in (
            ("produce", produce_predictions), ("freshness", freshness_predictions)):
        actual = [hierarchy[key][index] for index in labels]
        result[f"{key}_accuracy"] = accuracy_score(actual, marginal_predictions)
        result[f"{key}_macro_f1"] = f1_score(
            actual, marginal_predictions, average="macro", zero_division=0)
    result["selection_score"] = (0.5 * result["macro_f1"]
                                 + 0.3 * result["produce_macro_f1"]
                                 + 0.2 * result["freshness_macro_f1"])
    return result


def run_epoch(model, loader, criterion, optimizer, scaler, device, amp, hierarchy,
              ood_loader=None, ood_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    labels, predictions = [], []
    produce_predictions, freshness_predictions = [], []
    produce_mapping = torch.tensor(hierarchy["produce"], device=device)
    freshness_mapping = torch.tensor(hierarchy["freshness"], device=device)
    ood_batches = itertools.cycle(ood_loader) if training and ood_loader else None
    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
                if ood_batches is not None:
                    ood_inputs = next(ood_batches).to(device, non_blocking=True)
                    ood_logits = model(ood_inputs)
                    uniform_kl = -ood_logits.log_softmax(1).mean() - math.log(ood_logits.shape[1])
                    loss = loss + ood_weight * uniform_kl
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * len(targets)
        total_items += len(targets)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        probabilities = logits.softmax(1)
        for mapping, destination in ((produce_mapping, produce_predictions),
                                     (freshness_mapping, freshness_predictions)):
            totals = torch.stack([
                probabilities[:, mapping == group].sum(1)
                for group in range(int(mapping.max().item()) + 1)
            ], dim=1)
            destination.extend(totals.argmax(1).cpu().tolist())
    return {"loss": total_loss / total_items,
            **classification_metrics(labels, predictions, hierarchy,
                                     produce_predictions, freshness_predictions)}, labels, predictions


def produce_confidences(model, loader, device, produce_by_class, amp):
    model.eval()
    values = []
    mapping = torch.tensor(produce_by_class, device=device)
    with torch.inference_mode():
        for batch in loader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            probabilities = model(inputs.to(device, non_blocking=True)).softmax(1)
            totals = torch.stack([
                probabilities[:, mapping == group].sum(1)
                for group in range(int(mapping.max().item()) + 1)
            ], dim=1)
            values.extend(totals.max(1).values.cpu().tolist())
    return values


def calibrate_abstention(model, known_loader, ood_loader, device, hierarchy, amp):
    known = produce_confidences(model, known_loader, device, hierarchy["produce"], amp)
    unknown = produce_confidences(model, ood_loader, device, hierarchy["produce"], amp)
    candidates = sorted(set(known + unknown))
    best = None
    for threshold in candidates:
        known_acceptance = sum(value >= threshold for value in known) / len(known)
        unknown_rejection = sum(value < threshold for value in unknown) / len(unknown)
        score = (known_acceptance + unknown_rejection) / 2
        candidate = (score, known_acceptance, threshold, unknown_rejection)
        if best is None or candidate > best:
            best = candidate
    return {"produce_confidence_threshold": best[2],
            "validation_balanced_accuracy": best[0],
            "known_acceptance": best[1], "ood_rejection": best[3],
            "known_samples": len(known), "ood_samples": len(unknown)}


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
    parser.add_argument("--ood-dir", type=Path,
                        help="Unsupported-item images (for example apples) for outlier exposure")
    parser.add_argument("--ood-weight", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--produce-loss-weight", type=float, default=0.35)
    parser.add_argument("--freshness-loss-weight", type=float, default=0.15)
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
    if not 0 <= args.label_smoothing < 1:
        parser.error("label smoothing must be in [0, 1)")
    if min(args.ood_weight, args.produce_loss_weight, args.freshness_loss_weight) < 0:
        parser.error("loss weights cannot be negative")

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
    hierarchy = hierarchy_indices(class_names)
    datasets = {
        name: ImageDataset(splits[splits.split == name], args.data_dir, name == "train")
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train",
                         num_workers=args.workers, pin_memory=True)
        for name, dataset in datasets.items()
    }
    ood_train_loader = ood_val_loader = None
    if args.ood_dir:
        ood_train_data = UnlabeledImageDataset(args.ood_dir, training=True)
        ood_val_data = UnlabeledImageDataset(args.ood_dir, training=False)
        if len(ood_train_data) < 2:
            parser.error("OOD calibration requires at least two images")
        indices = list(range(len(ood_train_data)))
        random.Random(args.seed).shuffle(indices)
        val_count = max(1, round(len(indices) * 0.2))
        val_indices, train_indices = indices[:val_count], indices[val_count:]
        ood_train_loader = DataLoader(
            Subset(ood_train_data, train_indices), batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers, pin_memory=True)
        ood_val_loader = DataLoader(
            Subset(ood_val_data, val_indices), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
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
    train_criterion = HierarchicalLoss(
        class_names, class_weights, args.label_smoothing,
        args.produce_loss_weight, args.freshness_loss_weight).to(device)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1) if args.cosine else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best_score, stale = -1.0, 0
    checkpoint = args.output_dir / "best.pt"
    baseline_validation = None
    if args.init_checkpoint:
        baseline_validation, _, _ = run_epoch(
            model, loaders["val"], eval_criterion, None, scaler, device, args.amp,
            hierarchy)
        best_score = baseline_validation["selection_score"]
        torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                    "epoch": 0, "split_sha256": split_digest(splits),
                    "validation": baseline_validation, "parent_sha256": parent_sha256}, checkpoint)
        print(json.dumps({"baseline_validation": baseline_validation}), flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, loaders["train"], train_criterion, optimizer, scaler, device,
            args.amp, hierarchy, ood_train_loader, args.ood_weight)
        val_metrics, _, _ = run_epoch(
            model, loaders["val"], eval_criterion, None, scaler, device, args.amp,
            hierarchy)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_metrics["selection_score"] > best_score:
            best_score, stale = val_metrics["selection_score"], 0
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
    abstention = None
    if ood_val_loader:
        abstention = calibrate_abstention(
            model, loaders["val"], ood_val_loader, device, hierarchy, args.amp)
        saved["abstention"] = abstention
        torch.save(saved, checkpoint)
    test_metrics, actual, predicted = run_epoch(
        model, loaders["test"], eval_criterion, None, scaler, device, args.amp,
        hierarchy)
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
              "abstention": abstention,
              "parent_sha256": parent_sha256, "epochs_completed": len(history),
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"test": test_metrics, "gpu": environment["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
