# Improvement plan — implementation status

The planned product and model work is now implemented on this branch, which
includes main-branch commit `0a01691519f1782a2c5ee9328f3ecdc0ab28b486`.
The historic audit below continues to describe the earlier `be31339` snapshot.

Completed:

- Separate identity/freshness displays and an eight-class identification head.
- Explicit produce confirmation and correction without relabeling model scores.
- Conditional, sourced household storage windows and handling suggestions.
- Persistent source/assumption records in exports and limited browser history.
- An 80-photo source-grouped web pilot and separate unsupported validation/test pool.
- ROCm runs of the augmented CNN and a separate identity/freshness-head experiment.
- Validation-based model/head selection and pilot rejection calibration.
- Browser tests of normal, corrected, uncertain, invalid, unavailable, cancellation,
  storage, export, history and responsive states.

See [implemented improvements](IMPLEMENTED_IMPROVEMENTS.md) for measured results,
rejected candidates, inference coverage and limitations. The model release is a
local research pilot. A larger future benchmark with true physical item/session
IDs and independently reviewed freshness labels remains the next validation step.
Individual-item shelf-life prediction and production reliability are not claimed.

---

# FreshPick improvement plan

## Branch scope and baseline

This branch preserves the measured evaluation of commit
`be313398bf4aa67322573d4a537070ae7120ace5` and the v0.2.0 checkpoints.
The subsequently published `main` optimization commit
`0a01691519f1782a2c5ee9328f3ecdc0ab28b486` is not included in this audit snapshot.
Its marginal prediction rules, augmentations, auxiliary losses, and optional
outlier exposure require their own measured run; this report does not evaluate them.

Completed here: dataset and internet-image evaluation, source/attribution manifests,
per-produce/per-class metrics, calibration and degradation measurements, split
similarity screening, live API checks, and reproducible scripts. The evaluation
uses the original joint-winner decision policy. No new weights or storage guidance
are implemented in this commit.

## Next implementation work

1. Display produce and freshness confidence separately. Allow users to confirm or
   correct the produce identity before showing produce-specific handling guidance.
   Preserve model output separately from user confirmation in exports/history.
2. Replace the use-period placeholder with sourced typical storage guidance.
   Ask for whole/cut condition, relevant ripeness, and storage location. Store
   source links, assumptions, and review dates alongside each rule. Present general
   ranges rather than claiming a prediction of this item's remaining shelf life.
   Source selection and numerical windows are not yet implemented or verified.
3. Freeze an independent internet-image benchmark before expanding training data.
   Retain source/license and item/session identifiers; remove duplicate and related
   captures across splits. The 24 audited web probes are regression cases, not a
   representative final benchmark.
4. Compare an eight-class identifier and a separate freshness head with the
   24-class baseline and the newer main-branch optimization pipeline. Keep splits
   fixed and publish per-class precision, recall, F1, calibration, and coverage.
5. Validate any uncertainty/rejection threshold on held-out known and unsupported
   inputs. Verify incorrect, uncertain, unavailable, confirmation, guidance, and
   export flows. Calibrated confidence alone does not establish unknown recognition.

## Acceptance criteria

- Measured results refer to a checkpoint hash, source revision, split digest,
  inference precision/batch size, and complete source provenance.
- No improvement or shelf-life claim without matching evidence.
- Guidance remains unavailable until required user inputs and sourced rules exist.
- Existing camera/gallery/report/Inspector/history/export flows continue to pass.
- Model replacement requires independent benchmark evidence, rather than a gain
  on the reused random split alone.
