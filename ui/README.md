# FreshPick UI

Responsive browser UI based on the five supplied FreshPick screen references. The original screenshots are kept in [`references/`](references/) for comparison; the running app does not load or display those screenshots.

## Run locally

```bash
.venv/bin/python -m uvicorn freshpick.api:app --host 127.0.0.1 --port 8000
```

Run from the repository root after installing `.[api]` and the ROCm runtime. Open `http://localhost:8000`. The built-in `/predict` endpoint is connected by default. Camera access works on localhost or a secure HTTPS origin. The user can select the inference API URL with the information button; the setting is stored in this browser only.

## Connect model inference

The UI sends `POST` to the configured endpoint as `multipart/form-data`, with the image in the `image` field. The response must follow the contract in [`../UI_ALIGNMENT_PROMPT.md`](../UI_ALIGNMENT_PROMPT.md), including the 24 joint-class probabilities and the three freshness probabilities. The built-in service shares the UI origin. A custom cross-origin inference service must allow requests from the UI origin through CORS.

The included API loads `runs/mobilenetv3-finetuned/best.pt`, or the path in `FRESHPICK_CHECKPOINT`. The trained checkpoint exists locally and stays ignored by Git. Missing weights produce a visible service error; no successful prediction is fabricated. Scan history stores the produce label, freshness label, score, and timestamp in local browser storage; it does not retain the scanned image. The current image remains in memory for the Report and Inspector. Home shows the latest real scan. Clearing browser site data removes saved history.

The model supports banana, bitter melon, cucumber, eggplant, orange, papaya, pineapple, and tomato. It has no apple or general “other produce” class. Feature explanations and shelf-life estimates are unavailable. The report includes visible-condition and food-safety limits.


The current pilot also requires the separate identity checkpoint and `.[identifier]`.
Use the setup in the root README. Produce and freshness scores are distinct;
identity disagreement, rejection or a user correction withholds freshness.
The storage form requires explicit user confirmation and observed storage details.
Typical ranges link to their reviewed sources, are invalidated on edits, and are
included in exports/history separately from model predictions. No item-specific
remaining-day estimate is made. See `reports/STORAGE_GUIDANCE.md`.
