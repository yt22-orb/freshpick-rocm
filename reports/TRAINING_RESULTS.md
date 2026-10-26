# FreshPick measured training results

Completed on the local AMD GPU using `2.11.0+rocm7.2`,
HIP `7.2.26015`, GPU `AMD Radeon RX 7800 XT`.

Command:

```bash
.venv/bin/python -m freshpick.train --data-dir prepared_dataset --output-dir runs/mobilenetv3-small --epochs 20 --batch-size 32 --amp
```

Prepared 5174 images, 24 classes; source SHA-256
`6607a48bf2817f9dfc8d4a1c32ce663c1d8308b4b1251e9bf65f074fcac53f11`. Exclusions:
`{'excluded_publisher_augmentation': 8943, 'kept': 5174, 'excluded_conflicting_labels': 34, 'excluded_exact_duplicate': 9}`.

Seed 42, stratified random 70/15/15 split: {'train': 3621, 'test': 777, 'val': 776}. Pretrained ImageNet
MobileNetV3-Small, 224×224 black-padded RGB images, ImageNet normalization,
horizontal flips and 10-degree rotations during training, AdamW learning rate
0.0003, batch size 32, AMP, maximum 20 epochs, early-stopping patience 5.
Early stopping ended training after epoch 16.
Best validation macro-F1 checkpoint: epoch 11.

Held-out test accuracy: **95.8816%**; macro F1:
**0.957233**; cross-entropy loss: 0.119079.
Split SHA-256: `97a36346183f44b03205bd884896f7fd072635727d67eaac3c45fed3e81e30d8`.

Local artifacts in `runs/mobilenetv3-small/`: `best.pt`, `history.csv`,
`training.log`, `splits.csv`, `metrics.json` (including per-class scores),
`confusion_matrix.csv`, `environment.json`, `runtime-requirements.txt`, and
`api-verification.json`. Model and dataset files stay ignored by Git.

Verification: six automated tests passed. The API was also called with an actual
held-out image and the trained ROCm checkpoint: HTTP 200, all 24 joint probabilities,
three normalized freshness sums, and consistent displayed freshness confidence.
Invalid/missing/oversized/unsupported-format images, missing weights, and inference
failure responses were checked. No unsupported-item rejection threshold is validated.

These are measurements on inherited AgriFreshNET folder labels, not verified
shelf-life or food-safety performance. Related views can cross random splits because
item/capture IDs are unavailable. Exact duplicate removal does not prevent this
leakage. No separate natural-use camera evaluation was performed.

The UI was subsequently added upstream and integrated with the trained model. See
[UI verification](UI_VERIFICATION.md) for current end-to-end checks and limitations.
