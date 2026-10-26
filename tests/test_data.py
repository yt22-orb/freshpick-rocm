from pathlib import Path

import pandas as pd
import pytest

from freshpick.data import load_manifest, make_splits


def test_stratified_splits_are_disjoint_and_complete():
    rows = [
        {"sample_id": f"{class_id}-{i}", "image_path": f"{class_id}/{i}.png",
         "class_id": class_id, "class_name": f"class_{class_id}",
         "pixel_sha256": f"{class_id}-{i}"}
        for class_id in range(3) for i in range(30)
    ]
    manifest = pd.DataFrame(rows)
    split = make_splits(manifest, 42)
    assert len(split) == len(manifest)
    assert split.sample_id.is_unique
    assert set(split.split) == {"train", "val", "test"}
    assert split.groupby(["class_id", "split"]).size().min() > 0
    assert split.equals(make_splits(manifest, 42))


def test_duplicate_pixels_are_rejected(tmp_path: Path):
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()
    pd.DataFrame([
        {"sample_id": "a", "image_path": "a.png", "class_id": 0,
         "class_name": "fresh_banana", "pixel_sha256": "same"},
        {"sample_id": "b", "image_path": "b.png", "class_id": 0,
         "class_name": "fresh_banana", "pixel_sha256": "same"},
    ]).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="Exact-pixel duplicates"):
        load_manifest(tmp_path)
