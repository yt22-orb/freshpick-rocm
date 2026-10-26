# FreshPick UI integration verification

Updated to upstream commit `b7e477f` and retained the locally trained ROCm model
and inference adapter. UI and API run together at http://localhost:8000.

```bash
.venv/bin/python -m uvicorn freshpick.api:app --host 127.0.0.1 --port 8000
```

`POST /predict` accepts the UI's `image` upload field and the original `file`
field. The UI defaults to the same-origin endpoint. Model weights remain local
at `runs/mobilenetv3-finetuned/best.pt`. No retraining was required.

Verified in headless Chromium with the actual trained AMD/ROCm inference service:

- Mobile Home → Scan and simulated browser camera preview
- Gallery upload → Evaluating → real ROCm response → Report; confidence matches API
- Inspector freshness totals, 24 joint classes, model version, unavailable diagnostics
- Report export and persistent real scan history on Home and History
- Simulated camera capture → real ROCm inference → Report
- Actual invalid-image API error shown in Scan
- Simulated server error shown without fabricated result
- Simulated uncertain response presentation; no validated threshold claimed
- Cancel pending inference returns to Scan
- Home, Scan, Report, Inspector: no horizontal overflow at 320/390/768/1440 px
- Unavailable camera provides gallery fallback
- Simulated camera permission denial provides gallery fallback

Six Python tests pass, JavaScript syntax checks pass, and the browser reported no
uncaught JavaScript errors. Visual inspection of the mobile Report confirmed the
input image, actual confidence, cautious guidance, and unobstructed navigation.
A late camera-startup result can no longer overwrite an inference error or leave
a stale camera stream running after navigation. Cancelled responses cannot replace
a newer scan. Home displays real history instead of a permanent empty state.

Evidence stays local under `runs/ui-verification/`: five screen captures,
`live-response.json`, `checks.json`, `exported-report.txt`, and `browser_check.py`.

Limits: the camera capture used Chromium's simulated camera, not physical camera
hardware. Permission-denial and uncertain/server-error responses were simulated;
an actual unavailable-camera fallback and invalid-image API response were checked.
No unsupported-item detector or validated confidence threshold exists. The app
can classify unsupported produce into one of its known classes. No camera-domain
accuracy, shelf-life, or food-safety performance is established by this UI check.
Scan summaries persist in browser local storage (up to 12); image previews live
only in memory. The inference adapter does not save uploaded images.

All browser checks were rerun successfully after fine-tuning. The live API model
version matched the fine-tuned checkpoint SHA-256. See [fine-tuning results](FINETUNING_RESULTS.md).
