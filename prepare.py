"""Reproduce the FreshPick preparation from the audited notebook."""


# Uncomment if these packages are missing.
# %pip install numpy pandas pillow

from pathlib import Path
from collections import Counter
from io import BytesIO
import hashlib
import json
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "source" / "Processed Data.zip"
OUT = BASE / "prepared_dataset"
SOURCE.parent.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://data.mendeley.com/public-files/datasets/42m5tb7yv9/files/7b216853-8f3b-47fd-b1ae-df88a367a1d1/file_downloaded"
EXPECTED_SHA256 = "6607a48bf2817f9dfc8d4a1c32ce663c1d8308b4b1251e9bf65f074fcac53f11"
if not SOURCE.exists():
    partial = SOURCE.with_suffix(".zip.partial")
    urllib.request.urlretrieve(URL, partial)
    partial.replace(SOURCE)
def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
archive_hash = file_hash(SOURCE)
assert archive_hash == EXPECTED_SHA256, "Source archive checksum mismatch; download again."
print("Source checksum verified:", archive_hash)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
archive = zipfile.ZipFile(SOURCE)
assert archive.testzip() is None, "Archive CRC check failed."
entries = sorted(n for n in archive.namelist()
                 if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTENSIONS)
folders = sorted({Path(n).parent.name for n in entries})
def parse_label(folder):
    match = re.fullmatch(r"(.+?)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)", folder)
    if not match:
        raise ValueError(f"Unrecognized folder: {folder}")
    text = re.sub(r"[_\s]+", " ", match[1].strip()).lower()
    for prefix, freshness in [("semi fresh ", "semi_fresh"), ("fresh ", "fresh"), ("rotten ", "rotten")]:
        if text.startswith(prefix):
            produce = text[len(prefix):].strip()
            return {"class_name": f"{freshness}_{produce}", "produce": produce,
                    "freshness": freshness, "publisher_day_min": int(match[2]),
                    "publisher_day_max": int(match[3])}
    raise ValueError(f"Unrecognized freshness: {folder}")
labels = {folder: parse_label(folder) for folder in folders}
class_names = sorted({v["class_name"] for v in labels.values()})
class_to_id = {name: i for i, name in enumerate(class_names)}
class_table = pd.DataFrame([{**v, "class_id": class_to_id[v["class_name"]],
                            "source_folder": k} for k, v in labels.items()])
class_table.to_csv(OUT / "class_map.csv", index=False)
print("Archive images:", len(entries), "Classes:", len(class_names))
print(pd.Series(Counter(Path(n).parent.name for n in entries)).to_string())


audit = []
for index, name in enumerate(entries):
    folder = Path(name).parent.name
    row = {"sample_id": f"img_{index:05d}", "source_path": name,
           **labels[folder], "class_id": class_to_id[labels[folder]["class_name"]]}
    row["is_publisher_augmentation"] = bool(re.match(r"^aug_\d+_", Path(name).name, re.I))
    if row["is_publisher_augmentation"]:
        row["status"] = "excluded_publisher_augmentation"
    else:
        try:
            with Image.open(BytesIO(archive.read(name))) as image:
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                row.update({"width": rgb.width, "height": rgb.height,
                            "source_mode": image.mode,
                            "pixel_sha256": hashlib.sha256(
                                f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest(),
                            "status": "candidate"})
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            row.update({"status": "excluded_unreadable", "error": str(error)})
    audit.append(row)
audit = pd.DataFrame(audit)
candidates = audit[audit.status.eq("candidate")]
for digest, group in candidates.groupby("pixel_sha256", sort=True):
    if group.class_name.nunique() > 1:
        audit.loc[group.index, "status"] = "excluded_conflicting_labels"
    else:
        audit.loc[group.index[0], "status"] = "kept"
        audit.loc[group.index[1:], "status"] = "excluded_exact_duplicate"
kept = audit[audit.status.eq("kept")].copy()
assert len(kept), "No usable images remained."
assert kept.pixel_sha256.is_unique
print(audit.status.value_counts().to_string())
print("Candidate dimension counts:")
print(candidates.groupby(["width", "height"]).size().to_string())


def extract_features(content):
    rgb = np.asarray(content, dtype=np.float32) / 255.0
    hsv = np.asarray(content.convert("HSV"), dtype=np.float32) / 255.0
    features = {}
    for i, channel in enumerate("rgb"):
        features[f"{channel}_mean"] = float(rgb[..., i].mean())
        features[f"{channel}_std"] = float(rgb[..., i].std())
        counts, _ = np.histogram(rgb[..., i], bins=16, range=(0, 1))
        for bin_id, value in enumerate(counts / counts.sum()):
            features[f"{channel}_hist_{bin_id:02d}"] = float(value)
    angle = hsv[..., 0] * 2 * np.pi
    features["hue_sin_mean"] = float(np.sin(angle).mean())
    features["hue_cos_mean"] = float(np.cos(angle).mean())
    for i, channel in [(1, "saturation"), (2, "value")]:
        features[f"{channel}_mean"] = float(hsv[..., i].mean())
        features[f"{channel}_std"] = float(hsv[..., i].std())
    gray = np.asarray(content.convert("L"), dtype=np.float32) / 255.0
    features["brightness_mean"] = float(gray.mean())
    features["brightness_std"] = float(gray.std())
    p = np.bincount(np.asarray(content.convert("L")).ravel(), minlength=256).astype(float)
    p = p[p > 0] / p.sum()
    features["gray_entropy_bits"] = float(-(p * np.log2(p)).sum())
    if min(gray.shape) >= 3:
        center = gray[1:-1, 1:-1]
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * center
        features["laplacian_variance"] = float(lap.var())
        features["edge_strength_mean"] = float((np.abs(gray[1:, :-1] - gray[:-1, :-1]) + np.abs(gray[:-1, 1:] - gray[:-1, :-1])).mean())
    else:
        features["laplacian_variance"] = 0.0
        features["edge_strength_mean"] = 0.0
    return features

manifest_rows, feature_rows = [], []
for count, (idx, row) in enumerate(kept.iterrows(), 1):
    with Image.open(BytesIO(archive.read(row.source_path))) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        content = ImageOps.contain(rgb, (224, 224), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (224, 224), (0, 0, 0))
        canvas.paste(content, ((224 - content.width) // 2, (224 - content.height) // 2))
        relative_path = f"images/{row.class_name}/{row.sample_id}.png"
        destination = OUT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        metadata = row.to_dict()
        metadata.update({"image_path": relative_path,
                         "content_width": content.width, "content_height": content.height})
        manifest_rows.append(metadata)
        feature_rows.append({"sample_id": row.sample_id, **extract_features(content)})
        audit.loc[idx, "image_path"] = relative_path
    if count % 1000 == 0:
        print("Prepared", count, "images")
manifest = pd.DataFrame(manifest_rows)
features = pd.DataFrame(feature_rows)
manifest.to_csv(OUT / "manifest.csv", index=False)
features.to_csv(OUT / "features.csv", index=False)
audit.to_csv(OUT / "cleaning_audit.csv", index=False)
class_counts = manifest.groupby(["class_id", "class_name", "produce", "freshness"]).size().reset_index(name="image_count")
class_counts.to_csv(OUT / "class_counts.csv", index=False)
print("Prepared images:", len(manifest), "Numerical features:", features.shape[1] - 1)
print(class_counts.to_string(index=False))


assert len(audit) == len(entries)
assert len(manifest) == len(features)
assert manifest.sample_id.is_unique and features.sample_id.is_unique
assert manifest.sample_id.tolist() == features.sample_id.tolist()
assert manifest.pixel_sha256.is_unique
assert not manifest.is_publisher_augmentation.any()
assert np.isfinite(features.drop(columns="sample_id").to_numpy()).all()
assert manifest[["class_id", "class_name", "produce", "freshness", "image_path"]].notna().all().all()
for path in manifest.image_path:
    with Image.open(OUT / path) as image:
        image.load()
        assert image.size == (224, 224) and image.mode == "RGB"
print("All prepared images and feature rows passed validation.")
summary = {"dataset": "AgriFreshNET version 1", "doi": "10.17632/42m5tb7yv9.1",
           "source_url": "https://data.mendeley.com/datasets/42m5tb7yv9/1",
           "source_archive": "Processed Data.zip", "source_sha256": archive_hash,
           "source_is_raw": False, "raw_archive_published": False,
           "source_image_count": len(entries), "retained_image_count": len(manifest),
           "source_class_count": len(class_names), "retained_class_count": manifest.class_name.nunique(),
           "numerical_feature_count": features.shape[1] - 1,
           "status_counts": {str(k): int(v) for k, v in audit.status.value_counts().items()},
           "prepared_image_size": [224, 224], "validation_passed": True,
           "limitations": ["Source is already processed; strict raw-data requirement is unmet.",
                           "Folder labels are inherited, not visually validated.",
                           "Day intervals are publisher metadata, not measured remaining shelf life.",
                           "No verified subject IDs; near duplicates may remain.",
                           "Features include background; no produce segmentation."]}
(OUT / "preparation_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
