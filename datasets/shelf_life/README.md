# FreshPick storage reference dataset

Version `2026-10-26.1`, reviewed 26 October 2026. This dataset contains household
quality/storage guidance, not measured expiry labels for individual photographs.

## Files

- [`produce_storage.csv`](produce_storage.csv): 32 reviewed rows: eight produce ×
  whole/cut × counter/refrigerator. There are 24 numerical windows and eight
  handling-only rows for cut produce on a counter. Blank day values mean no
  multi-day window, never zero days or unlimited storage.
- [`../../src/freshpick/shelf_life_reference.json`](../../src/freshpick/shelf_life_reference.json):
  the packaged runtime reference, with source registry, version and assumptions.
- [`foodkeeper_mirror_snapshot.json`](foodkeeper_mirror_snapshot.json): 661
  FoodKeeper-derived product records in a third-party transformed display format.
  This is a pinned research archive, not the runtime reference or a verified
  current export from USDA.
- [`foodkeeper_mirror_manifest.json`](foodkeeper_mirror_manifest.json): retrieval
  URL, immutable commit, SHA-256, size, record count and provenance limitation.

The official [USDA FSIS FoodKeeper catalog](https://catalog.data.gov/dataset/fsis-foodkeeper-data)
lists its underlying dataset under CC0. Both official JSON endpoints returned
HTTP 403 here. The archived copy came from a pinned public mirror; its exact match
to current upstream content has **not** been verified. The runtime instead uses
the separately reviewed source rows below. It never downloads data during a scan.

## Whole-produce reference windows

These are conditional quality references. Purchase age is subtracted from the
matching range even when no spoilage signs are observed. References can differ
between sources; do not merge maxima into a longer window. A ripe-fruit-only
range requires the corresponding maturity detail. Full purchase age is used as
a conservative comparison rather than restarting a clock at ripening/chilling.

| Produce | Cool room / counter | Refrigerator | Source for room; source for fridge |
| --- | --- | --- | --- |
| Banana | 3–7 days | 3 days, ripe only | [Netherlands Nutrition Centre](https://www.voedingscentrum.nl/nl/thema/kopen-koken-bewaren/eten-bewaren/hoe-lang-kan-ik-bewaren/banaan.aspx); [Virginia Cooperative Extension](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Bitter melon | 3–4 days | 3–5 days | [Guyana New GMC](https://newgmc.com/gmc_docs/brochures/Bitter_Melon.pdf); [Canadian Produce Marketing Association](https://www.halfyourplate.ca/produce/veggies/bitter-melon/) |
| Cucumber | 7 days, whole with skin | 4–6 days | [Netherlands Nutrition Centre](https://www.voedingscentrum.nl/nl/thema/kopen-koken-bewaren/eten-bewaren/hoe-lang-kan-ik-bewaren/komkommer-met-schil.aspx); [Virginia Cooperative Extension](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Eggplant | 1 day | 4–7 days | [Virginia Cooperative Extension](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Orange | 10 days | 10–21 days | [Virginia Cooperative Extension, citrus row](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Papaya | 3–5 days | 4–7 days, ripe only | [Penn State Extension](https://extension.psu.edu/farmers-market-storage-postcard); [University of Florida IFAS](https://ask.ifas.ufl.edu/publication/MG054) |
| Pineapple | 2–3 days | 5–7 days | [Oregon State University Extension](https://extension.oregonstate.edu/newsletter/food-hero-monthly/pineapple-pineapple); [Virginia Cooperative Extension](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Tomato | 2–3 days, ripe only | 2–3 days, ripe only | [University of California ANR](https://ucanr.edu/sites/default/files/2018-09/290542.pdf); [Michigan State University Extension](https://www.canr.msu.edu/mi_fresh/uploads/files/2023%20Tomatoes_AA.pdf) |

Room references assume suitable cool indoor conditions, generally around
20–21°C; a hot Philippine kitchen is not necessarily covered by those conditions.
The bitter-melon ambient range describes the reported onset of postharvest
quality deterioration; the source does not specify its exact ambient temperature.
Refrigerator guidance assumes a working household refrigerator at or below 4°C.
Cucumbers, eggplant and bitter melon can suffer chilling injury, so refrigeration
does not universally lengthen a reference or improve quality. Commercial storage
at 10°C is not the same as a household refrigerator.

Cut/peeled produce uses the existing [Maine Extension handling guidance](https://extension.umaine.edu/food-health/2025/05/29/storing-and-washing-fresh-fruits-and-vegetables/)
of 3–4 refrigerated days; cut tomato retains the stricter three-day
[Health Canada reference](https://www.canada.ca/en/health-canada/services/food-safety-fruits-vegetables/tomatoes.html).
Counter storage of cut produce receives a handling action, not a multi-day
expiry range. Timing starts at cutting, and earlier unsafe exposure is not
cancelled by refrigeration.

## Reproduce the export

```bash
.venv/bin/python scripts/export_shelf_life_dataset.py
.venv/bin/python -m pytest -q
```

The export verifies the archived mirror's SHA-256 and regenerates CSV from the
packaged JSON. Live reference JSON is also available at `GET /shelf-life-reference`.
The app's upper-window discard cutoff is its conservative policy, not a source
claim that every item becomes unsafe at that exact age. Reported spoilage or
unsafe handling takes priority over any positive reference time.
