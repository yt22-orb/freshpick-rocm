# FreshPick model and UI alignment prompt

Use this prompt with a coding agent that can access both this repository and the FreshPick UI source. The attached screenshots are the visual and interaction reference. Treat UI copy that says “sample” or “illustrative” as placeholder content, never as evidence of model results.

---

## Prompt

Build the FreshPick image-classification pipeline and connect it to the FreshPick UI shown in the supplied screenshots. Inspect the actual UI source before changing it. Keep its warm cream background, orange accent, rounded cards, and five-screen flow: Home, Scan Fruit, Evaluating, Inspector, and Report. Preserve the intended camera and gallery paths, show progress while inference runs, and let the user scan another image.

### Model and supported labels

Train and run the model using the prepared AgriFreshNET version 1 data described in this repository. It contains 24 joint classes: eight supported produce types crossed with Fresh, Semi-fresh, and Rotten. The supported produce types are banana, bitter melon, cucumber, eggplant, orange, papaya, pineapple, and tomato. The dataset contains no apple images. Do not return “Red Apple” or any other unsupported produce label. Replace screenshot sample data in the live UI with clearly marked placeholder content until a real inference response is available. Checkpoints trained with unsupported-item outlier exposure may use their held-out calibrated threshold to return `status: "uncertain"`; describe this as rejection rather than apple recognition. A user-selected produce type may be used only if the UI clearly asks for it.

Keep the 24-class model output. For each inference, return:

```json
{
  "status": "ok",
  "produce": "banana",
  "produce_label": "Banana",
  "freshness": "semi_fresh",
  "freshness_label": "Semi-fresh",
  "class_confidence": 0.61,
  "produce_confidence": 0.68,
  "freshness_confidence": 0.78,
  "freshness_probabilities": {
    "fresh": 0.12,
    "semi_fresh": 0.78,
    "rotten": 0.10
  },
  "produce_probabilities": {
    "banana": 0.68,
    "...remaining 7 produce probabilities...": "..."
  },
  "class_probabilities": {
    "fresh_banana": 0.04,
    "semi_fresh_banana": 0.61,
    "rotten_banana": 0.03,
    "...remaining 21 class probabilities...": "..."
  },
  "model_version": "..."
}
```

The example numbers and labels above are schema examples only; the ellipsis is explanatory and is not a literal JSON key. Return all 24 actual entries in `class_probabilities`. Derive predicted produce by summing its three freshness-class probabilities and taking the largest produce marginal. Derive freshness the same way across all eight produce types. `class_confidence` remains the largest joint-class probability; `produce_confidence` and `freshness_confidence` are the winning marginal probabilities. Use `freshness_confidence` for the prominent UI percentage and label it “Freshness confidence.” The Inspector’s freshness bars must use the three marginal values and display percentages that total 100% (within rounding). Never substitute invented or constant confidence values.

### UI behavior and result wording

- Home: “Scan fruit” opens the scanner; recent scans must come from real session history or be labeled as examples.
- Scan: support camera capture and gallery upload. Validate that an image was selected, show camera permission and inference errors in plain language, and send the original image to inference.
- Evaluating: show “Image received,” preprocessing, and model evaluation states only when those steps are actually happening. Provide a working cancel/back action where practical.
- Report: show the predicted produce and freshness stage, the freshness confidence, and a visible-condition disclaimer. Do not estimate shelf life: this dataset does not provide validated remaining-shelf-life targets. Guidance must be cautious and must not claim that food is safe to eat.
- Inspector: show the input image, the three freshness probabilities, model/version information, and supported diagnostics only. The model is a whole-image classifier; it does not currently explain specific color, texture, or shape cues. Do not fabricate feature explanations. Either implement and validate separate image diagnostics or mark those sections unavailable/remove them.
- Store scans only as required for the displayed history and report flow. Make retention behavior clear and do not upload images to an unrelated service.

### API and integration

Provide a documented inference endpoint or adapter that accepts one image and returns the JSON contract above. Keep model preprocessing identical between training and inference: EXIF orientation, RGB conversion, aspect-preserving 224×224 resize with black padding, and the same normalization. Return useful HTTP errors for invalid files, unsupported inputs, missing model weights, and inference failures. Do not return a fake successful response while the model is untrained or unavailable.

The training and inference code must use ROCm on the configured gaming PC. Verify that the installed PyTorch build detects the AMD GPU before training. Do not call training complete until the training process has run on that PC and saved its actual logs, checkpoint, environment details, and held-out test metrics. Keep dataset images and archives out of the GitHub repository; link to AgriFreshNET and document how to prepare/download the data.

### Verification and handoff

Run the UI flows with real inference responses and with loading, invalid-image, unsupported-item, low-confidence, and server-error states. Check that the mobile layouts in the reference remain usable. Confirm the displayed freshness probabilities come from the response, and that neither the screenshots’ sample values nor placeholder data appear as real predictions. Update the README with setup, run, API, and training instructions; report which UI screens and end-to-end flows were actually verified. Do not claim UI integration or ROCm training that was not completed.


## Implemented separate-identifier pilot contract (v0.3.0)

The pilot now keeps the 24-class MobileNet outputs as freshness diagnostics and
adds an independent eight-class identifier. Its `produce_probabilities` (all eight
normalized entries), `produce_confidence` and `identifier_version` describe the
identity head. These need not equal marginal sums of `class_probabilities` from
the separate 24-class model. `freshness_probabilities` continue to be the three
summed 24-class outputs. `class_confidence` refers to that model's top joint class.

`freshness_model_produce` preserves the original CNN produce candidate;
`freshness_available` is false on identity rejection or disagreement. The UI must
also suppress incompatible freshness after an explicit user correction. Model
scores must never be relabeled as scores for user-confirmed produce. User identity
and storage inputs remain separate in history and exports.

`POST /storage-guidance` supplies source-backed general storage guidance from
explicit user answers. Numeric ranges are typical household reference windows,
never predictions of the scanned item's remaining shelf life. Missing confirmation,
uncertain spoilage, and unsupported conditions must not produce invented ranges.


## Photo-based form suggestions

`photo_observations.by_produce` holds explicit whole/cut and ripe-looking appearance
proposals, their raw model scores and a method/version/scope. These are not
freshness labels or verified maturity. The usual form asks Yes/No for the produce,
then storage location, odor/leakage/softening and elapsed days. Model proposals
pre-fill an optional details editor. Unknown condition stays unknown and blocks
numeric guidance. Corrected produce selects its own photo proposals. Do not copy
a previous produce's ripeness or convert Fresh into ripe. Storage requests include
`observation_source` to distinguish model suggestions from user edits.


## Reference-window timing extension

`storage-guidance.timing` compares reported elapsed age with a matching source-backed
reference range. Preserve both endpoints, clamp negatives to zero, and distinguish
within_window, near_limit, at_limit, past_limit and unavailable/age-needed states.
At/past the upper endpoint the app recommends discard as an explicit conservative
policy, not a proven expiry. “You still have” must say within the typical window,
never safe time guaranteed. Spoilage and unsafe handling override positive time.
Persist/export the entered-age snapshot, its caveat and calculated_at. Editing inputs
invalidates stale timing; rescanning must not reset elapsed age.
