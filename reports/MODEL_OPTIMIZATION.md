# Model optimization update

## Problem found

The deployed model has 24 joint labels formed from three freshness states and only
eight produce types. Apple is absent. The v0.2.0 checkpoint therefore cannot call
an apple an apple and has no calibrated unknown-item path, so a tomato result is a
forced closed-set prediction rather than evidence that tomato training accuracy is
low. Re-aggregating the published confusion matrix confirms 100% produce accuracy
and 1.0 produce macro-F1 on its eight-produce test split; its errors are between
freshness stages of the same produce. This closed-set result says nothing about
apples.

## Changes implemented

- Produce and freshness predictions now use marginal probabilities across the 24
  joint classes instead of copying the single joint-class winner.
- Training adds explicit marginal produce and freshness losses alongside joint
  cross-entropy, with label smoothing.
- Camera-domain robustness is improved with modest affine, brightness, contrast,
  saturation, scale, translation, and perspective augmentation. These are kept
  conservative so visible spoilage cues are not deliberately erased.
- `--ood-dir` enables outlier exposure. Unsupported images are trained toward a
  uniform class distribution rather than a false supported label.
- A deterministic 80/20 split of the unsupported-image pool reserves calibration
  images. The selected produce-confidence threshold balances known-item acceptance
  and unsupported-item rejection and is stored in the checkpoint.
- Evaluation now records produce accuracy/macro-F1 and freshness
  accuracy/macro-F1 in addition to the original 24-class metrics. Checkpoint
  selection combines joint, produce, and freshness macro-F1.
- The API and UI support `status: "uncertain"` and expose produce marginal scores.

## Verification and measured status

The Windows checkout can run CPU unit/API tests, but it has no WSL, ROCm PyTorch,
prepared dataset, or model checkpoint. Therefore no new held-out model score is
claimed here. The published v0.2.0 benchmark remains 96.6538% joint test accuracy
and 0.962623 joint macro-F1 until the optimized pipeline is run on the AMD Linux
training computer. The same confusion matrix yields 96.6538% freshness accuracy
and 0.966472 freshness macro-F1. Automated verification of the changed code passes
9 tests; JavaScript syntax checking also passes.

After that run, compare `test.accuracy`, `test.macro_f1`,
`test.produce_accuracy`, `test.produce_macro_f1`, `test.freshness_accuracy`, and
`test.freshness_macro_f1` in the old and new `metrics.json` files. Also report the
new `abstention.known_acceptance`, `abstention.ood_rejection`, and held-out OOD
balanced accuracy. An independent natural-camera set, especially apples and other
unsupported items not used in outlier training, is still required before claiming
real-world unknown-item performance.
