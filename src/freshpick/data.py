import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED = {"sample_id", "image_path", "class_id", "class_name", "pixel_sha256"}


def load_manifest(data_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(data_dir / "manifest.csv")
    missing = REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.empty or manifest.sample_id.duplicated().any():
        raise ValueError("Manifest is empty or has duplicate sample IDs")
    if manifest.pixel_sha256.duplicated().any():
        raise ValueError("Exact-pixel duplicates remain in prepared data")
    if any(not (data_dir / path).is_file() for path in manifest.image_path):
        raise FileNotFoundError("One or more prepared images are missing")
    class_ids = sorted(manifest.class_id.unique().tolist())
    if class_ids != list(range(len(class_ids))):
        raise ValueError("Class IDs must be contiguous, starting at zero")
    if manifest.groupby("class_id").class_name.nunique().max() != 1:
        raise ValueError("A class ID maps to multiple names")
    return manifest


def make_splits(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    train_val, test = train_test_split(
        manifest, test_size=0.15, random_state=seed, stratify=manifest.class_id
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=seed, stratify=train_val.class_id
    )
    parts = []
    for name, frame in (("train", train), ("val", val), ("test", test)):
        part = frame[["sample_id", "image_path", "class_id", "class_name", "pixel_sha256"]].copy()
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def split_digest(split: pd.DataFrame) -> str:
    payload = split.sort_values("sample_id").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()
