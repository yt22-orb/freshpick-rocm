# Location-specific storage reference dataset

The app now reads a versioned reference dataset for all eight supported produce
types, for both counter and refrigerated storage. The dataset contains 32 rows,
including separate cut/peeled rules, source links, condition requirements, source
temperature descriptions and age basis. See the
[dataset, source table and provenance](../datasets/shelf_life/README.md).

## Changed behavior

“No signs observed” continues to use purchase/harvest age and the selected storage
location. It cannot bypass the reference cutoff. Examples for a whole banana,
assuming the stated conditions and ripe fruit when refrigerated:

| Storage / reported age | Reference | Result |
| --- | --- | --- |
| Counter, 2 days | 3–7 days | About 1–5 days within the reference window |
| Counter, 7 days | 3–7 days | Reference maximum reached; discard conservatively, even without visible signs |
| Refrigerator, 2 days | 3 days | Near the reference limit; up to 1 reference day left |
| Refrigerator, 3 days | 3 days | Reference maximum reached; discard conservatively |

Missing age asks for elapsed days and never substitutes zero. Editing storage or
age invalidates the old result. A storage-location change does not reset purchase
age. Model photo suggestions remain editable assumptions, not verified maturity.
Reported spoilage and cut-produce handling limits still override the clock.

Counter coverage now includes all eight whole-produce references. Whole ripe
tomato refrigerator guidance is numeric (2–3 days) rather than “a few days”.
Ripe papaya's refrigerator reference changes from the earlier Clemson 1–2-day
choice to the reviewed UF/IFAS 4–7-day guidance. Different sources have different
quality windows; the dataset records this selected source rather than implying
all sources agree. Refrigerator references and cool room assumptions are shown
in the report. API results, exports and saved history retain the dataset version
and record ID for traceability.

## Data access and reproducibility

- Runtime: `src/freshpick/shelf_life_reference.json`, included in Python packages.
- CSV: `datasets/shelf_life/produce_storage.csv`.
- Public read-only API: `GET /shelf-life-reference`.
- Export script: `scripts/export_shelf_life_dataset.py`.
- Research archive: pinned 661-product FoodKeeper-derived mirror snapshot and
  SHA-256 manifest. Official downloads returned HTTP 403; this transformed mirror
  has not been verified against the current USDA upstream and does not drive
  runtime guidance.

This is a storage-rule dataset, not training data for a remaining-life regressor.
No image-model weights were retrained. Accurate item-specific shelf-life training
would require longitudinal records with initial maturity, purchase/harvest time,
measured temperature/humidity, storage changes and a defined quality endpoint.

## Validation

Automated checks cover every whole-produce/location cutoff with no observed
spoilage, location-dependent remaining intervals, record provenance, dataset
completeness, snapshot integrity, existing uncertainty/spoilage priority and API
access without loading inference weights. Browser verification submits the real
evaluation banana image, switches room/refrigerator conditions, checks no-signs
cutoffs, and exercises the existing report, storage, export and history flows.

All 74 automated tests and 15 product-flow browser check groups passed. The Python
wheel builds successfully and includes the runtime reference JSON. JavaScript
syntax and whitespace checks passed. Browser capture used a simulated camera;
the banana upload and storage requests used the actual running API.
