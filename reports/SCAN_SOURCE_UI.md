# Camera and Gallery scan views

Verified locally on 2026-10-26 using Chromium and the running FastAPI inference
service at `http://localhost:8000`.

## Behavior

- Camera requests video permission on entry, with no audio request. A ready video
  preview enables capture and switching between front/rear camera preferences.
- Start/Retry and explanatory messages handle denied permission, missing devices,
  busy cameras and unavailable camera APIs. Camera access requires localhost or
  HTTPS and browser/device permission.
- Opening Gallery stops active video tracks. Late permission results are also
  stopped if the user has already switched away. Leaving Scan or hiding the page
  releases the camera; use Start camera when returning to a stopped preview.
- Gallery has a dashed upload area, drag highlight and a Choose image button.
  Selecting its tab opens the panel without opening the operating system picker.
  The shutter, flash and camera-switch controls are hidden.
- One JPG, PNG, WebP or BMP image up to 20 MB can be selected or dropped. Type,
  empty-file, size and file-count validation precede inference. Server decoding
  still rejects unreadable image bytes. Upload errors return to Gallery.
- Arrow keys/Home/End switch source tabs. The upload area supports Enter/Space,
  and the Choose image button provides the standard file-picker interaction.

## Verification

With the API running, install the browser extras and Chromium as described in the
root README, then run:

```bash
.venv/bin/python scripts/verify_scan_sources.py
.venv/bin/python scripts/verify_product_flows.py
```

These scripts use the existing evaluation photo at
`runs/evaluation-web/images/banana_direct_19231740.jpg` and installed model assets.
Results and screenshots are written under `runs/scan-source-verification/` and
`runs/product-verification/` (ignored by Git).

The scan-source check passed nine groups covering file selection and native
DataTransfer drops against the real inference service, hidden camera controls,
track shutdown, delayed camera access, upload failures, keyboard interactions,
permission/device/busy errors with retry, and widths of 320/390/768/1440 pixels.
It reported no page JavaScript errors. The existing product-flow regression
also passed, including capture, reports, storage guidance, history and exports.

Camera frames came from Chromium's simulated media device. No physical camera
device was exposed to this runtime. Physical webcam permission, device switching
and capture still need a manual check in the user's browser. The browser code
uses `getUserMedia` for an attached physical camera during normal use.
