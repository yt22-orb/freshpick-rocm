# FreshPick ROCm training

Code to prepare AgriFreshNET images and train a 24-class produce freshness classifier on an AMD GPU with ROCm. **No model has been trained for this repository yet.** Training results must come from a run on the gaming PC.

## Dataset

- **Source:** [AgriFreshNET Freshness and Shelf-Life Image Dataset, version 1](https://data.mendeley.com/datasets/42m5tb7yv9/1), DOI [10.17632/42m5tb7yv9.1](https://doi.org/10.17632/42m5tb7yv9.1), by Fouzia Risdin, Hafsha Salauddin, Tasniya aziza tufa, and Md Jakir Hossain. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **FreshPick preparation:** `prepare.py` reproduces the included [preparation notebook](notebooks/FreshPick_Data_Preparation.ipynb). The audited run retained 5,174 images in 24 classes and excluded marked publisher augmentations and exact-pixel duplicates. The resulting images and archive are **not in this repository**.
- The available source archive is already processed. Folder freshness labels are inherited, not independently verified. Publisher day ranges are not measured remaining shelf life. Related views of one produce item may still cross a random split because item and capture IDs are unavailable. Report test scores with that limitation.

## On the gaming PC

Use a Linux version and ROCm/PyTorch combination supported for your exact AMD GPU. Check the [AMD Radeon ROCm compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html) and [AMD's PyTorch installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html). Install a ROCm-enabled `torch` and matching `torchvision` using AMD's instructions first. A CPU-only PyTorch build cannot run this training script.

```bash
git clone https://github.com/yt22-orb/freshpick-rocm.git
cd freshpick-rocm
python3 -m venv .venv
source .venv/bin/activate
# Install ROCm-enabled torch and torchvision as directed by AMD for this PC.
python -m pip install -e .
python -c "import torch; print('HIP:', torch.version.hip, 'GPU available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Download the `Processed Data.zip` file from the [dataset page](https://data.mendeley.com/datasets/42m5tb7yv9/1) and place it at `source/Processed Data.zip`. Alternatively, `prepare.py` attempts to download the exact version 1 archive when that file is absent. It checks the archive SHA-256 before processing. Install the preparation dependencies and run:

```bash
python -m pip install -r requirements.txt
python prepare.py
```

Preparation writes `prepared_dataset/manifest.csv`, the image folders, class map, feature table, and audit. The 65 hand-crafted features are preserved from the earlier preparation project; this image classifier trains from the PNG images.

Train and evaluate:

```bash
python -m freshpick.train --data-dir prepared_dataset --output-dir runs/mobilenetv3-small --epochs 20 --batch-size 32 --amp
```

The model uses MobileNetV3-Small with ImageNet pretrained weights by default; the first run may download those weights. Pass `--no-pretrained` to start from random weights. `--amp` enables mixed precision. Reduce the batch size if GPU memory is insufficient. The script requires ROCm, uses validation macro F1 for checkpoint selection and early stopping, then evaluates the held-out test split once. It saves `splits.csv`, `history.csv`, `best.pt`, `metrics.json`, and `confusion_matrix.csv` under `runs/`. These outputs remain local and are ignored by Git. Record the exact command and environment with any reported scores.

Run inference after training:

```bash
python -m freshpick.predict path/to/photo.jpg --checkpoint runs/mobilenetv3-small/best.pt
```

This predicts one of the 24 inherited produce-and-freshness labels. The reported confidence is a model score, not a calibrated food-safety probability. Do not use it to decide whether food is safe to eat.

## Checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The repository contains code and documentation only. The source dataset remains at Mendeley Data; no dataset images or ZIP files are committed here.
