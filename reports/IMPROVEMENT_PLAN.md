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
