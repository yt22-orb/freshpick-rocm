# FreshPick fine-tuning results

Fine-tuned the original MobileNetV3-Small checkpoint on the RX 7800 XT with
PyTorch 2.11.0+rocm7.2 / HIP 7.2.26015.
The split stayed fixed: 3,621 training, 776 validation, 777 test images.
Split SHA-256: `97a36346183f44b03205bd884896f7fd072635727d67eaac3c45fed3e81e30d8`.
Parent checkpoint SHA-256: `7e1ee2803754f3ca324371b4be8911c2aa52e3b8ad2658006e6ec02229537746`.

```bash
.venv/bin/python -m freshpick.train --data-dir prepared_dataset \
  --output-dir runs/mobilenetv3-finetuned \
  --init-checkpoint runs/mobilenetv3-small/best.pt \
  --epochs 15 --batch-size 32 --learning-rate 0.00003 --patience 5 --cosine --amp
```

All layers were fine-tuned using AdamW and the original class-weighted loss and
training augmentations. Cosine decay approaches 10% of the starting rate.
The initial checkpoint was validated before training and retained as a fallback.
Validation macro F1 alone selected the checkpoint; test scores were evaluated after
selection, with no test-based threshold tuning. This reuses the previously evaluated
test set, so the comparison is not an independent fresh test.

| Metric | Original | Fine-tuned |
| --- | ---: | ---: |
| Validation macro F1 | 0.954665 | 0.971147 |
| Validation accuracy | 96.1340% | 97.4227% |
| Test accuracy | 95.8816% | 96.6538% |
| Test macro F1 | 0.957233 | 0.962623 |
| Test loss | 0.119079 | 0.098786 |

Stopped after 6 epochs; selected fine-tuning epoch 1.
Fine-tuned checkpoint SHA-256: `a9e3a06213d9c2442700a88bf3f970ebd3ded9941e7b3ecb19355e2e99a5cf24`.
The app and CLI now default to `runs/mobilenetv3-finetuned/best.pt`.
Detailed metrics, environment/configuration, per-class results, history, and confusion
matrix are in [finetuning/](finetuning/). The full checkpoint and run evidence are
published as release assets; dataset images and archives are excluded.

Limitations: inherited folder labels, no item/capture IDs, and possible related views
across random splits. These measurements do not establish general camera accuracy,
unknown-produce rejection, shelf life, or food safety. No new images were added.
