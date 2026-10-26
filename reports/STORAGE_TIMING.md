# Storage-window estimates and discard reminders

Implemented on 26 October 2026. This is a source-backed rule/notification feature,
not newly trained model knowledge of shelf life. Image-model weights are unchanged.

## Behavior

The cited household window matches confirmed produce, whole/cut state, ripeness
where relevant and storage location. Reported elapsed age is compared with both
ends of that window. “You still have” always refers to the typical reference
window, not guaranteed safe remaining life. A photo-derived condition or maturity
suggestion remains an explicit, editable assumption.

For a ripe whole refrigerated banana with a cited three-day reference:

| Reported age | Display/action |
| --- | --- |
| 0 days | About 3 reference days left |
| 1 day | You still have about 2 days within the typical storage window |
| 2 days | Near the limit, up to 1 reference day left; use promptly if appropriate |
| 2.5 days | Near the limit, up to 0.5 reference days left |
| 3 days | Limit reached; discard conservatively |
| More than 3 days | Past limit; discard conservatively |
| Unknown age | Ask for age; no numerical countdown |

For a 3–5 day range and two days already stored, the interval is 1–3 days, not a
promise of three more days. At four days the earlier endpoint has passed and the
UI shows near-limit/up-to-one reference day. Negative time is never displayed.
Cut/peeled windows count from cutting; whole-produce entries count from purchase
or harvest under the reported conditions. Earlier unknown or unsafe handling is
not cancelled by refrigeration. A saved result is a snapshot of the entered age;
the user must update age when revisiting. Fractional days are accepted.

## Discard decision and priority

The user-requested automatic cutoff uses the reference window's upper endpoint.
At or past it, status is discard_recommended and timing.action is discard. This
is an explicitly conservative app policy, not a source assertion that every item
becomes biologically spoiled at that exact age.

- Reported unusual odor, leakage or significant softening prompts disposal before
  any positive reference countdown.
- Unknown spoilage or condition, unconfirmed identity and unsupported combinations
  cannot produce a positive remaining-window estimate.
- Cut produce on a counter gets no safe-time countdown. Reported exposure at or
  beyond two hours triggers conservative discard; otherwise prompt refrigeration
  and the shorter one-hour hot-condition rule are displayed.
- No extra time is created by a new scan or a storage-location change.

The feature uses [Virginia Cooperative Extension household storage guidance](https://www.pubs.ext.vt.edu/348/348-960/348-960.html),
which describes general recommendations under suitable conditions, not measured
item-specific expiry. [University of Maryland Extension](https://www.extension.umd.edu/resource/storing-garden-fruits-and-vegetables)
supports discarding produce with the listed spoilage signs.
[Penn State Extension](https://extension.psu.edu/proper-care-and-handling-of-fruits-and-vegetables)
advises against keeping cut produce left at room temperature beyond two hours.
[FDA](https://www.fda.gov/food/consumers/tips-reduce-food-waste) documents the
shorter one-hour limit above 32°C / 90°F. The maximum-window discard policy is the
app's conservative choice; it should not be misrepresented as a prediction from
these sources or from the image model.

## API and UI

POST /storage-guidance adds a timing object containing state, action, heading,
body, elapsed_days, reference_days_left, limit_days, basis, policy, calculated_at,
a caveat and whether photo-based assumptions apply. Before/near/at/past-limit
states are explicit, and unknown age/conditions never fabricate a timer.
Sources and existing typical_window_days remain available. Discard results
suppress conflicting suggestions to continue storage.

The result card highlights the action, explains the reference assumptions and
shows when the entered age needs updating. Changes invalidate old advice and
cancel in-flight requests. Exports include the timing/action and calculated
snapshot alongside source links and model-versus-user observations; browser
history stores the same snapshot without retaining the image.

## Verification

54 Python tests pass, including zero/fractional ages, exact and exceeded limits,
range endpoints, missing age/details, photo assumptions, spoilage overrides and
cut-counter exposure. Live browser checks exercise within/near/at/past limits,
missing-age and stale-result behavior, source links, export/history, image/camera
flows, uncertainty, errors, cancellation and responsive layouts.
