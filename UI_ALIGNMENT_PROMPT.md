# FreshPick model and UI alignment prompt

Use this prompt with a coding agent that can access both this repository and the FreshPick UI source. The attached screenshots are the visual and interaction reference. Treat UI copy that says “sample” or “illustrative” as placeholder content, never as evidence of model results.

---

## Prompt

Build the FreshPick image-classification pipeline and connect it to the FreshPick UI shown in the supplied screenshots. Inspect the actual UI source before changing it. Keep its warm cream background, orange accent, rounded cards, and five-screen flow: Home, Scan Fruit, Evaluating, Inspector, and Report. Preserve the intended camera and gallery paths, show progress while inference runs, and let the user scan another image.

### Model and supported labels

Train and run the model using the prepared AgriFreshNET version 1 data described in this repository. It contains 24 joint classes: eight supported produce types crossed with Fresh, Semi-fresh, and Rotten. The supported produce types are banana, bitter melon, cucumber, eggplant, orange, papaya, pineapple, and tomato. The dataset contains no apple images. Do not return “Red Apple” or any other unsupported produce label. Replace screenshot sample data in the live UI with clearly marked placeholder content until a real inference response is available. This classifier has no “other produce” class, so its softmax scores cannot reliably detect apples or other unsupported items. If the UI needs an unknown-item path, use a validation-supported abstention threshold and describe it only as an uncertainty check, not verified out-of-domain recognition. A user-selected produce type may be used only if the UI clearly asks for it.

Keep the 24-class model output. For each inference, return:

```json
{
  "status": "ok",
  "produce": "banana",
  "produce_label": "Banana",
  "freshness": "semi_fresh",
  "freshness_label": "Semi-fresh",
  "class_confidence": 0.61,
  "freshness_confidence": 0.78,
  "freshness_probabilities": {
    "fresh": 0.12,
    "semi_fresh": 0.78,
    "rotten": 0.10
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

The example numbers and labels above are schema examples only; the ellipsis is explanatory and is not a literal JSON key. Return all 24 actual entries in `class_probabilities`. Derive the predicted produce and freshness from the highest-probability joint class. Derive the three `freshness_probabilities` by summing the 24 joint-class probabilities for each freshness stage, then renormalize for rounding. `class_confidence` is the winning joint-class probability; `freshness_confidence` is the summed probability for the predicted freshness stage. Use `freshness_confidence` for the prominent UI percentage and label it “Freshness confidence.” The Inspector’s freshness bars must use these three summed values and display percentages that total 100% (within rounding). Never substitute invented or constant confidence values.

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
