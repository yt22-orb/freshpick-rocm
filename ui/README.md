# FreshPick UI

Responsive browser UI based on the five supplied FreshPick screen references. The original screenshots are kept in [`references/`](references/) for comparison; the running app does not load or display those screenshots.

## Run locally

```bash
python3 -m http.server 8080 --directory ui
```

Open `http://localhost:8080`. Camera access works on localhost or a secure HTTPS origin. The user can select the inference API URL with the information button; the setting is stored in this browser only.

## Connect model inference

The UI sends `POST` to the configured endpoint as `multipart/form-data`, with the image in the `image` field. The response must follow the contract in [`../UI_ALIGNMENT_PROMPT.md`](../UI_ALIGNMENT_PROMPT.md), including the 24 joint-class probabilities and the three freshness probabilities. The inference service must allow requests from the UI origin through CORS.

No inference API is included yet, and no trained checkpoint is in this repository. Until a compatible service is connected, the UI shows that a model endpoint is missing and does not fabricate a result. Scan history stores the produce label, freshness label, score, and timestamp in local browser storage; it does not retain the scanned image.

The model supports banana, bitter melon, cucumber, eggplant, orange, papaya, pineapple, and tomato. It has no apple or general “other produce” class. Feature explanations and shelf-life estimates are unavailable. The report includes visible-condition and food-safety limits.
