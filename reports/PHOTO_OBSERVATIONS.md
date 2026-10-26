# Photo-based storage details and simpler confirmation

The normal form now asks only whether the model identification is correct, where
it is stored, unusual odor/leakage/major softening, and optional days stored.
The produce selector is hidden unless the user chooses No. Whole/cut/peeled and
ripe-looking details are shown as photo suggestions, not additional required
questions. An optional details editor lets users correct them.

## Inference and evidence

The existing pinned OpenCLIP image embedding is reused for text-based appearance
proposals. No new image encoder or identity/freshness weights were trained for
this change. Whole versus cut/peeled means whether exposed flesh or cut pieces
are visible. Ripeness appearance is assessed only for banana, papaya and tomato;
other produce shows not assessed. Green color is not automatically called spoilage.
No proposal derives ripeness from the inherited Fresh/Semi-fresh/Rotten labels.

Single-reviewer visual labels for the frozen 80-photo identity pilot are recorded
in photo-observations/visual-labels.json. They are appearance labels, not verified
physical maturity, edibility or freshness. One plate-of-bitter-melon condition
annotation was corrected to whole after inspecting the full source image.

Confidence cutoffs use the existing validation split only: maximize the number
of non-unknown proposals while requiring at least five proposals and at least
90% empirical visual-label agreement. This rule does not guarantee population
precision. Thresholds are midpoints between observed scores for numerical stability.
The prompt and label hashes, validation counts, test records and errors are in
photo-observations/evaluation.json. The calibration JSON is included in the Python
package; missing or incompatible calibration yields unknown instead of a default.

Test results: whole/cut suggestions on 16/20 test photos, with 15/16 correct
(93.75% among suggestions). Ripe-looking/unripe-looking suggestions on 6/8 relevant
test photos, with 4/6 matching the reviewed visual labels (66.67%). The ripeness
sample is particularly weak, includes errors on mixed or distant produce, and must
not be presented as reliable maturity recognition. These photos were also used in
previous model experiments, so this is a small reused pilot, not an untouched new
benchmark. Suggestions are visibly provisional and editable.

For the real white-background banana regression image, live inference returned
whole (model score about 96.9%) and ripe-looking (about 98.2%). These scores are
not maturity or food-safety probabilities. No freshness confidence is manufactured:
the identifier calls it Banana but the retained CNN calls it Cucumber. The report
now explains that specific disagreement beside the unavailable score.

## Storage and state handling

Photo-derived suggestions use observation_source=photo_model. Manually edited
values use user_edited_photo. The source-backed storage endpoint records those
origins and explicitly makes any general range conditional on the suggested
details being correct. Unknown condition never silently becomes whole; it blocks
numeric guidance. A ripe-looking image does not create verified freshness.

Changing the identity selects suggestions for the corrected produce, rather than
reusing banana ripeness for another item. Changing details invalidates existing
advice and aborts pending requests. Model observations and edited details are stored
separately in limited session history and exports. New scans reset confirmation,
correction and overrides. No uploaded images are retained by the server.

## Verification

37 Python tests pass. Updated live browser checks cover yes-only auto-fill,
No/correction, optional edits, conditional source-linked guidance, unavailable
freshness explanation, stale-advice invalidation, export/history, uncertainty,
invalid/server-error states, simulated camera capture/permission denial,
cancellation and 320/390/768/1440px layouts. Physical camera hardware and general
maturity accuracy remain unverified.

Reproduce suggestion calibration with:

```bash
python scripts/collect_identity_pilot.py
python scripts/train_identifier.py
python scripts/calibrate_photo_observations.py
```

The calibration script reuses the pinned-encoder feature caches for the frozen
image manifests. The reviewed source photos remain outside Git; source URLs,
licenses and checksums are in reports/identity-pilot/manifest.json.
