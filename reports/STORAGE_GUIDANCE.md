# Storage guidance implementation

The report asks the user to identify the item and explicitly confirm it, then
provide whole/cut condition, observed ripeness, counter/refrigerator location,
visible spoilage signs and optional elapsed days. The model's produce guess,
user-confirmed identity and storage-rule inputs are separate fields.

`POST /storage-guidance` accepts JSON, for example:

```json
{"produce":"banana","confirmed":true,"condition":"whole","ripeness":"ripe","location":"refrigerator","spoilage":"no","days_stored":1}
```

A response includes `status`, `window_label`, optional `typical_window_days`,
`suggestions`, `assumptions`, `sources` with links/review dates, the original
`inputs` and a visible disclaimer. The quoted window is a general reference;
it is not an estimate of days remaining and does not restart at the scan date.
The endpoint is deterministic and does not require inference weights.

Rules and their reviewed primary sources are in `src/freshpick/storage.py`.
[The implementation report](IMPLEMENTED_IMPROVEMENTS.md) lists the numeric rules
and source URLs. Guidance review date: 2026-10-26. Required conditions differ
by produce; no numeric rule is invented for unsupported combinations. Whole ripe
tomatoes have qualitative guidance, because the chosen source says a few days.

Confirmed produce is required. Observed or uncertain spoilage blocks numerical
windows and prompts inspection/disposal guidance. Cut produce on a counter prompts
refrigeration/disposal handling rather than a day count. Unripe or uncertain whole
banana/papaya/tomato ripeness does not produce a fixed day range. Elapsed days are
not subtracted into an inferred expiry; reaching the reference maximum adds an
explicit reminder to reassess rather than treating the interval as extra time.

Editing any input cancels a pending guidance request and invalidates previous
advice. A late response cannot attach to a different scan. Corrected produce
suppresses the incompatible model freshness assessment and labels the confirmed
identity separately. Exports include assumptions, sources and model-versus-user
identity. Browser history retains up to 12 summaries and guidance; it stores no
image bytes. Clearing site data removes those records.

This is household storage information, not a food-safety determination. Shelf-life
prediction for an individual item remains outside the trained model's targets.


## Streamlined photo-assisted form

The UI now asks “Is this [model candidate]?” with Yes / No. No reveals a correction
selector. Condition and ripeness appearance are pre-filled from explicit photo
observation outputs and displayed as suggestions, with an optional editor.
They are never inferred by relabeling a Fresh classification as ripe.
The remaining regular inputs are storage location, odor/leakage/softening and
optional days stored. observation_source records whether details came from the
photo model or a user edit. Unknown condition blocks a numerical window.
[Photo observations](PHOTO_OBSERVATIONS.md) documents the pilot accuracy and limits.
