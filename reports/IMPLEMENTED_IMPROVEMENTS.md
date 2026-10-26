# FreshPick product and model update

Implemented on 26 October 2026 on the RX 7800 XT. These are pilot measurements,
not a claim of production reliability or food-safety accuracy.

## Product changes

- Separate produce confidence and freshness confidence. Independent eight-class
  identity output no longer gets overwritten by the 24-class freshness model.
- If the identifier rejects an image, the two models disagree, or the user corrects
  the produce, the prominent freshness result becomes unavailable. Raw classifier
  probabilities remain available in Inspector as diagnostics.
- An explicit confirmation form asks the user for actual produce, whole/cut state,
  observed ripeness, storage location, spoilage signs and optional time already stored.
  Correcting the produce does not relabel the model's freshness prediction.
- Source-backed typical storage guidance for all eight produce types. Each response
  includes assumptions, source links and review date. No inferred remaining days,
  shelf-life countdown, or image-based food-safety claim is returned. Spoilage,
  unconfirmed items and unsupported combinations block numerical windows.
- Exports and the last 12 browser history summaries retain model versus user identity
  separately. Editing inputs cancels and invalidates previous guidance. Image previews
  remain in memory; the server does not save uploaded images.
- Gallery, camera, loading/cancellation, Report, Inspector, history and responsive
  layouts are preserved. The report photo now uses contain rather than cropping it.

## Data and benchmark

80 newly reviewed Commons photos: 40 train, 20 validation, 20 test. All eight
produce types appear in each split. No photographer or dHash≤5 connected component
crosses those splits. A single reviewer verified identities before model inference;
no web freshness labels were assigned. Metadata, source URLs, licenses, file hashes,
manual choices and the frozen manifest are included in identity-pilot/.

Manifest SHA-256: `66f80deb0a8667cbc24c544966796db36d015a853f5e54aba06617e5fc2ab5e7`. A separate unsupported pool has seven
calibration and eight test images (apples plus clearly nonproduce examples).
No new web test photos train the classifier. Source-photographer groups are a
conservative proxy; physical item/session IDs and overlap with large-scale CLIP
pretraining remain unknown. This is substantially smaller than the proposed
30–50-photo-per-produce benchmark and must remain a pilot.

## Model experiments and selection

The upstream augmented/hierarchical MobileNet pipeline was actually fine-tuned:
8 completed epochs, selected epoch 3, test joint accuracy
96.1390%, versus the published 96.6538%.
It was not deployed because it did not improve that benchmark and identified only
6/20 new test photos.

A frozen [OpenCLIP ViT-B/32 encoder](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
feeds an eight-class supervised identity head initialized from text prototypes.
The encoder revision is pinned to `1a25a446712ba5ee05982a381eed697ef9b435cf`. Only the small
head is fitted on 3,621 original training images and 40 new web training photos,
with equal total weight for the two domains. Two regularization settings and a
text-only head were compared using validation only. A 98% dataset validation
macro-F1 guard rejects a large dataset regression; the remaining validation score
weights web macro-F1 75% and dataset macro-F1 25%. Temperature is fitted on web
validation. No test labels fit the head, temperature, or threshold.

Transparency: the initial unconstrained text-only candidate's test results were
inspected before the validation guard was added. It scored 19/20 on the web but
only 69.63% on the original dataset and was rejected. The small pilot has therefore
informed an architecture decision; it must not be represented as a final untouched
independent benchmark. Its measurements need confirmation on a future collection.
The record is in rejected-text-only.json.

## Measured identity results

| Measurement | Published CNN (current marginal policy) | Separate identifier |
| --- | ---: | ---: |
| New web test identity | 6/20 (30%) | 17/20 (85%) |
| Original 24 web regression probes | 11/24 | 21/24 |
| Original dataset produce accuracy | 100.00% | 99.7426% |
| New web test macro precision | — | 86.4583% |
| New web test macro recall | — | 85.4167% |
| New web test macro F1 | 0.2687 | 0.8446 |

The historic joint-winner v0.2.0 audit reported 10/24; the current marginal baseline
reports 11/24. These are different decision policies, not another weight improvement.

| produce | n | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| banana | 2 | 66.67% | 100.00% | 80.00% |
| bittermelon | 3 | 100.00% | 66.67% | 80.00% |
| cucumber | 2 | 50.00% | 50.00% | 50.00% |
| eggplant | 3 | 100.00% | 66.67% | 80.00% |
| orange | 2 | 100.00% | 100.00% | 100.00% |
| papaya | 3 | 75.00% | 100.00% | 85.71% |
| pineapple | 2 | 100.00% | 100.00% | 100.00% |
| tomato | 3 | 100.00% | 100.00% | 100.00% |

Per-type test supports are just 2–3. The overall identity accuracy 95% Wilson interval
is 64.0%–94.8%.
The original dataset is reused and can contain related train/test views.

## Rejection, calibration and coverage

The margin between supported and unsupported text prototypes is calibrated on
20 web validation photos, 776 original validation photos, and seven unsupported
validation photos. Known domains have equal weight, so the larger original pool
cannot overwhelm web coverage. Thresholds are midpoints between observed scores
for floating-point stability. Validation domain acceptance: {'web': 1.0, 'agrifresh': 0.9355670103092784}.

On the new web test, 19/20 pass the check
(95% coverage), with
84.21% accuracy among accepted
identity predictions. Unsupported test rejection is 7/8
(87.5%). This does not establish general unknown
recognition. Incorrect accepted classifications still occur.

On the original dataset, identity acceptance is
92.15%; prominent freshness is available
for 714/777 (91.89%).
Joint correctness among those available assessments is
96.36%. Coverage and abstentions
must be reported alongside conditional accuracy.

## Freshness experiment

A separate 520→128→3 head was trained on frozen image features plus produce identity,
using only original training freshness labels. Validation selected epoch
130; dataset test accuracy was
88.93%, macro F1
0.8868. This is below the retained
freshness baseline, so it is not deployed. Web freshness accuracy remains unknown.
The deployed freshness scores still come from the existing 24-class MobileNet;
they are withheld when incompatible with the identifier or user correction.

## Storage rules and sources

General household ranges are conditional on user answers and initially sound produce.
Sources can disagree; each rule names the particular source used. The ranges do
not restart at scanning and are never subtracted into predicted remaining days.

| Item / condition | Reference window | Source |
| --- | --- | --- |
| Ripe whole banana, refrigerated | 3 days | [Virginia Cooperative Extension](https://www.pubs.ext.vt.edu/348/348-960/348-960.html) |
| Whole cucumber, refrigerated | 4–6 days | Virginia Cooperative Extension |
| Whole eggplant, refrigerated | 4–7 days | Virginia Cooperative Extension |
| Whole orange, refrigerated | 10–21 days | Virginia Cooperative Extension |
| Whole pineapple, refrigerated | 5–7 days | Virginia Cooperative Extension |
| Ripe whole papaya, refrigerated | 1–2 days | [Clemson Extension](https://hgic.clemson.edu/factsheet/selecting-storing-fruits-vegetables/) |
| Whole bitter melon, refrigerated | 3–5 days | [Canadian Produce Marketing Association](https://www.halfyourplate.ca/produce/veggies/bitter-melon/) |
| Cut produce, promptly refrigerated | 3–4 days | [University of Maine Extension](https://extension.umaine.edu/food-health/2025/05/29/storing-and-washing-fresh-fruits-and-vegetables/) |
| Cut tomato, refrigerated | 3 days | [Health Canada](https://www.canada.ca/en/health-canada/services/food-safety-fruits-vegetables/tomatoes.html) |

[Penn State Extension](https://extension.psu.edu/proper-care-and-handling-of-fruits-and-vegetables)
supports prompt chilling and discarding cut produce left out beyond two hours.
[Maryland Extension](https://www.extension.umd.edu/resource/storing-garden-fruits-and-vegetables)
supports the observed-spoilage handling check. These rules are not model targets.

## Verification

32 Python checks pass, including calibration, split integrity, conditional guidance
and incompatible-freshness behavior. Live Chromium checks exercise gallery upload,
simulated camera capture/permission denial, correct and corrected identities,
uncertainty, source-linked advice, spoilage blocking, history/export, cancellation,
invalid/server-error states and 320/390/768/1440px layouts. Physical camera hardware
and natural-image freshness labels remain unverified. Evidence is in
identity-pilot/product-verification.json; local screenshots are in runs/product-verification/.

## Serving latency

The warmed two-model localhost API measured median 26.9 ms and p95 34.6 ms over 20 sequential test-image uploads after three warmups. This excludes cold model/encoder download, WAN latency, camera startup and concurrent load.

## Next validation needed

Collect a substantially larger future web/phone set with physical item/session IDs,
multiple reviewers and recorded freshness definitions. Add more cucumber and eggplant
varieties and handling conditions. Keep that future set untouched by model decisions.
The present local app and model release are explicitly experimental.
