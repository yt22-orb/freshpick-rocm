# FreshPick ROCm training

Code to prepare AgriFreshNET images and train a 24-class produce freshness classifier on an AMD GPU with ROCm. **Training completed on the local RX 7800 XT using ROCm.** The currently published checkpoint's test accuracy is 96.65% and macro F1 is 0.9626. See [fine-tuning results](reports/FINETUNING_RESULTS.md) for that checkpoint's original-versus-fine-tuned comparison. The newer hierarchical/outlier-exposure pipeline is documented in [the optimization report](reports/MODEL_OPTIMIZATION.md) and must be retrained before claiming replacement metrics.

For the supplied FreshPick UI reference, use the [UI alignment prompt](UI_ALIGNMENT_PROMPT.md). It defines the supported produce labels, inference response shape, and UI behavior expected from the model integration.

The responsive browser UI is in [`ui/`](ui/README.md) and is served together with the ROCm inference API. Run `.venv/bin/python -m uvicorn freshpick.api:app --host 127.0.0.1 --port 8000` and open http://localhost:8000. The UI uses the trained local checkpoint automatically.

The supplied UI kit now provides the branding, artwork, icons and styling. See
[UI kit integration](reports/UI_KIT_REFRESH.md) for the updated Home, Scan and
Insights views and verification results.

The [shelf-life reference dataset](datasets/shelf_life/README.md) now covers both
room and refrigerated storage for all eight produce types, with 32 source-reviewed
whole/cut records and downloadable CSV/JSON. Reported purchase age still applies
when no spoilage signs are observed. See [dataset integration](reports/SHELF_LIFE_DATASET.md).

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

For production-style fine-tuning, place a diverse set of unsupported-item images
(including apples) under `ood_dataset/`, keep that directory out of Git, and run:

```bash
python -m freshpick.train --data-dir prepared_dataset \
  --output-dir runs/mobilenetv3-hierarchical \
  --init-checkpoint runs/mobilenetv3-finetuned/best.pt \
  --ood-dir ood_dataset --epochs 20 --batch-size 32 \
  --learning-rate 0.00003 --patience 6 --cosine --amp
```

This adds camera-style augmentation, label smoothing, hierarchical produce and
freshness losses, and outlier exposure. Twenty percent of the unsupported images
are held out to calibrate a produce-confidence abstention threshold. Use enough
diverse, independently sourced images to make that held-out check meaningful; do
not evaluate on the same individual images used for training.

The model uses MobileNetV3-Small with ImageNet pretrained weights by default; the first run may download those weights. Pass `--no-pretrained` to start from random weights. `--amp` enables mixed precision. Reduce the batch size if GPU memory is insufficient. The script requires ROCm, uses validation macro F1 for checkpoint selection and early stopping, then evaluates the held-out test split once. It saves `splits.csv`, `history.csv`, `best.pt`, `metrics.json`, and `confusion_matrix.csv` under `runs/`. These outputs remain local and are ignored by Git. Record the exact command and environment with any reported scores.

Run inference after training:

```bash
python -m freshpick.predict path/to/photo.jpg --checkpoint runs/mobilenetv3-finetuned/best.pt --identifier-checkpoint runs/identifier/best.pt
```

This predicts one of the 24 inherited produce-and-freshness labels. The reported confidence is a model score, not a calibrated food-safety probability. Do not use it to decide whether food is safe to eat.

## Checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The repository contains code and documentation only. The source dataset remains at Mendeley Data; no dataset images or ZIP files are committed here.

## FreshPick inference API

Install the API dependencies after installing the ROCm runtime:

```bash
python -m pip install -e '.[api,identifier,test]'
uvicorn freshpick.api:app --host 127.0.0.1 --port 8000
curl -F 'file=@path/to/photo.jpg' http://127.0.0.1:8000/predict
```

Set `FRESHPICK_CHECKPOINT` to override `runs/mobilenetv3-finetuned/best.pt`.
Interactive endpoint documentation is at `/docs`. `POST /predict` accepts a
multipart `image` (or the CLI-adapter-compatible `file` field) (JPEG, PNG, WebP, or BMP, maximum 20 MiB). It returns the contract
in `UI_ALIGNMENT_PROMPT.md`, including all 24 joint probabilities, three summed
freshness probabilities, the joint winner, and a checkpoint SHA-256 based model
version. Inference requires an available AMD GPU and ROCm PyTorch. Images are
processed in memory and are not saved as scan history by this adapter.

Errors: 400 for unreadable images, 413 for oversized uploads, 415 for unsupported
image formats, 422 for missing input, and 503 for unavailable weights/runtime or
failed inference. Missing weights never produce a successful prediction.
Preprocessing handles EXIF orientation, converts to RGB, preserves aspect ratio
with black padding to 224×224, and applies ImageNet normalization. Training adds
horizontal flips and rotations; evaluation and inference share preprocessing.

Display `freshness_confidence` as “Freshness confidence” and use
`freshness_probabilities` for the three bars. Scores are uncalibrated. The v0.2.0 baseline has no validated
abstention threshold. The new separate identifier has a small-pilot-calibrated
supported-versus-unsupported similarity check; it is not general unknown recognition. A
new checkpoint trained with `--ood-dir` stores its calibrated threshold and can
return `status: "uncertain"`; this is rejection, not apple recognition. Shelf life,
food safety, and explanations of specific image features are not model outputs.

The UI and model API share one origin; no endpoint configuration or CORS setup is
needed for local use. The information button permits an optional custom endpoint.
See [UI verification](reports/UI_VERIFICATION.md) for tested flows and remaining limits.


## Download the trained model

The default UI/API checkpoint is distributed with [release v0.2.0](https://github.com/yt22-orb/freshpick-rocm/releases/tag/v0.2.0), rather than committed to Git.

```bash
mkdir -p runs/mobilenetv3-finetuned
gh release download v0.2.0 --repo yt22-orb/freshpick-rocm --pattern best.pt --pattern SHA256SUMS --dir runs/mobilenetv3-finetuned
(cd runs/mobilenetv3-finetuned && sha256sum --ignore-missing -c SHA256SUMS)
.venv/bin/python -m uvicorn freshpick.api:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. To fine-tune your own checkpoint, use the command in
[the fine-tuning report](reports/FINETUNING_RESULTS.md), adjusting `--init-checkpoint`
and keeping a separate output directory. The original class order and split digest
must match. A new optimizer is used; this is weight initialization, not exact optimizer-state resumption.


## Generalization audit

The [model evaluation report](reports/MODEL_EVALUATION.md) compares both checkpoints,
all 24 classes, confidence calibration, image degradation, and internet probes.
The fine-tuned FP32 audit scores 96.78% joint accuracy on the reused dataset test
split, but only 10/24 correct produce identities on a small, manually reviewed
internet convenience sample. Web freshness accuracy is unknown because those
images lack verified stage labels. Similar train/test views, cultivar/background
variation, blur sensitivity, and confident errors limit general-use claims.
Prioritize an independent item/session-grouped camera benchmark and better
training coverage before more tuning or wider deployment. The [PDF report](reports/MODEL_EVALUATION.pdf), [standalone HTML report](reports/MODEL_EVALUATION.html),
and [next implementation plan](reports/IMPROVEMENT_PLAN.md) are included. This
historical audit measures `be31339`; later optimization and product work are
covered by [implemented improvements](reports/IMPLEMENTED_IMPROVEMENTS.md).


## Current local pilot: identification, confirmation and storage guidance

The running pipeline uses a separate eight-class produce identifier and retains
MobileNetV3-Small for the 24-class freshness diagnostics. On the new 20-image
source-grouped web pilot, produce identification improved from 6/20 to 17/20;
the original dataset produce accuracy is 99.74%. These small measurements do not
establish general-use accuracy. The first text-only candidate and an augmented
CNN were rejected; details and per-produce precision/recall/F1 are in
[the implementation report](reports/IMPLEMENTED_IMPROVEMENTS.md).

The UI shows produce and freshness scores separately. An uncertain identity,
model disagreement or user correction withholds incompatible freshness output.
Source-linked typical storage guidance asks the user to confirm the actual item
and its storage conditions. It is general guidance, not predicted remaining days.
See [storage guidance](reports/STORAGE_GUIDANCE.md).

For a fresh checkout, download the two small pipeline checkpoints from the
[v0.3.0 pilot release](https://github.com/yt22-orb/freshpick-rocm/releases/tag/v0.3.0-pilot):

```bash
python -m pip install -e '.[api,identifier]'
mkdir -p runs/release-v0.3 runs/mobilenetv3-finetuned runs/identifier
gh release download v0.3.0-pilot --repo yt22-orb/freshpick-rocm --pattern best.pt --pattern identifier.pt --pattern SHA256SUMS --dir runs/release-v0.3
(cd runs/release-v0.3 && sha256sum -c SHA256SUMS)
cp runs/release-v0.3/best.pt runs/mobilenetv3-finetuned/best.pt
cp runs/release-v0.3/identifier.pt runs/identifier/best.pt
python -m uvicorn freshpick.api:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. The first identifier request downloads the pinned
OpenCLIP encoder (about 600 MB) from Hugging Face, then processes images locally
on ROCm. No uploaded photo is sent there. Encoder revision:
`1a25a446712ba5ee05982a381eed697ef9b435cf`.
`FRESHPICK_IDENTIFIER` overrides the identity-head checkpoint; set it to an empty
string to deliberately run only the baseline. CLI `--baseline` evaluates the
original 24-class pipeline without the separate identifier.

Reproduce the identity experiment after preparing AgriFreshNET:

```bash
python -m pip install -e '.[identifier,evaluation]'
python scripts/collect_identity_pilot.py
python scripts/fetch_web_evaluation.py
python -m freshpick.train --data-dir prepared_dataset --output-dir runs/mobilenetv3-optimized --init-checkpoint runs/mobilenetv3-finetuned/best.pt --epochs 15 --batch-size 32 --learning-rate 0.00005 --patience 5 --cosine --amp
python scripts/train_identifier.py
python scripts/compare_identity_pilot.py
python scripts/report_product_update.py
```

The downloader uses the frozen manifests and SHA-256 records by default;
`--discover` collects new candidates that require manual review and a new freeze.
The archived audit commands reproduce historical joint-winner measurements;
new comparisons must name the decision policy as well as the checkpoint.

Browser verification requires `pip install -e '.[browser]'` and
`python -m playwright install chromium`, then run
`python scripts/verify_product_flows.py` with the server running. Physical-camera
hardware and a future independent phone benchmark remain unverified.


## Photo-assisted storage form

The form now asks “Is this [produce]?” instead of requesting the identification
again. No reveals the correction selector. Whole/cut/peeled and ripe-looking
appearance are suggested from the photo, with an optional details editor.
Only storage location, unusual odor/leakage/major softening and days stored remain
as regular questions. Unknown observations stay unknown and may prevent a numeric
range. Source guidance is conditional on any model-suggested details being correct.
See [photo-observation results and limitations](reports/PHOTO_OBSERVATIONS.md).

Freshness confidence remains unavailable when the identity and freshness models
disagree; the card now explains which produce each model identified. Confirming
an identity or seeing a ripe-looking suggestion does not validate freshness.


## Storage-window status

Enter days already stored to see a reference-window estimate, near-limit notice,
or a conservative discard recommendation at/past the cited maximum. This is
rule-based arithmetic, not a model prediction of biological shelf life or a safety
guarantee. Spoilage and unsafe handling override positive estimates. Missing age
or unsupported conditions show no invented countdown. See
[storage timing](reports/STORAGE_TIMING.md) for the rules, examples and verification.
