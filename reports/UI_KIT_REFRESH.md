# Supplied UI kit integration

Implemented on 2026-10-26 from the user-provided `freshpick_ui_kit` package.

## Design and behavior

- Imported the supplied brand, hero, placeholder, shutter and individual SVG
  icons into `ui/assets/kit/`, alongside the original CSS and JSON design tokens.
- Applied orange `#FF7417`, cream `#F4EFEA`, card `#F9F5F1`, charcoal `#24262A`,
  token-based typography and spacing to the running interface. The font stack
  falls back to the system font; no remote fonts are required.
- Home uses the apple mark and hero, orange-emphasized title, scan icon CTA and
  a compact recent-scan card. That card opens Insights and is empty until a real
  scan is saved. User corrections retain the model's original candidate; the
  Home card describes uncalibrated confidence as high/moderate/low rather than
  presenting a rounded 100% as verified accuracy.
- Labeled Home / Scan / Insights navigation replaces the old unlabeled glyphs.
  Insights contains the existing local scan history and counts saved scans and
  user-confirmed labels. It uses up to 12 saved records and makes no new
  accuracy or freshness claims. `aria-current` follows the selected view.
- Camera and Gallery tabs have matching icons. Camera uses the new artwork,
  shutter and labeled controls, with the live preview and permission/retry
  behavior intact. Unsupported flash remains disabled. Camera status sits below
  the frame so permission errors are visible before capture controls.
- Gallery keeps its drag-and-drop area and explicit Choose image button, with
  camera controls hidden. The scan screens use a full view without bottom
  navigation, as in the supplied prototype.
- Reports, storage guidance, history and Inspector inherit the palette and font
  stack while retaining their existing data and behavior.

## Verification

Run with the FastAPI service and model assets installed:

```bash
.venv/bin/python scripts/verify_ui_refresh.py
.venv/bin/python scripts/verify_scan_sources.py
.venv/bin/python scripts/verify_product_flows.py
.venv/bin/python -m pytest -q
node --check ui/app.js
git diff --check
```

The UI refresh check uses isolated browser history fixtures for confirmed,
corrected and uncertain states; these are never seeded into normal app use.
It checks asset loads, JavaScript errors, exact brand orange, font fallback,
recent-card navigation, saved-record counts, correction wording, a simulated
missing camera, hidden Gallery camera controls and 320/390/768/1440px layouts.
Home's note fits above navigation at 390x844; navigation targets are at least 44px.

Screenshots and results are saved under `runs/kit-ui-verification/` and ignored
by Git. Fixture screenshots are named with `fixture` to distinguish them from
real inference results. Existing product and scan-source checks also exercise
the actual model service using the original evaluation banana image and a
simulated browser camera. Physical camera hardware remains a manual check.

All three browser scripts passed: six UI-refresh groups, nine scan-source groups
and fourteen product-flow groups. No page JavaScript errors were reported. All
54 automated tests passed, along with JavaScript syntax and whitespace checks.

## Alignment follow-up

The camera frame previously reached its 405px height limit and shrank to about
352px wide inside the 390px content column. Its left edge stayed aligned with the
tabs, which moved its center about 19px left. An explicit full width now keeps
both edges aligned with the tabs. Equal control columns also center the shutter
independently of the widths of the Flash and Switch labels.

Removed the viewport-based minimum height from individual screens and reserve
bottom navigation space only on Home and Insights. The shell retains its normal
viewport minimum height; longer content still scrolls. This removes extra page
overflow on tall desktop viewports without moving page headers vertically.

Browser measurements cover camera/shutter centering at widths of
320/390/522/768/1440px, and confirm short views fit a 1383px-tall viewport without
extra page scrolling. The full product and scan-source checks passed after the
screen sizing change.
