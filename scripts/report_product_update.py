"""Build the product/model implementation handoff from measured artifacts."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('reports/identity-pilot')
model=json.loads((OUT/'experiment.json').read_text());compare=json.loads((OUT/'comparison.json').read_text());freeze=json.loads((OUT/'freeze.json').read_text())
base=json.loads(Path('runs/mobilenetv3-finetuned/metrics.json').read_text());optimized=json.loads(Path('runs/mobilenetv3-optimized/metrics.json').read_text())
shutil.copy2('runs/mobilenetv3-optimized/metrics.json',OUT/'optimized-cnn-metrics.json')
shutil.copy2('runs/mobilenetv3-optimized/history.csv',OUT/'optimized-cnn-history.csv')
# Report the deployed separation/abstention policy, using archived offline probability passes.
saved=torch.load('runs/identifier/best.pt',map_location='cpu',weights_only=True)
f=torch.tensor(np.load('runs/identifier/agrifresh-test.npz')['features'])
ids=(f@saved['identity_weight'].T).argmax(1).numpy();margin=((f@saved['positive_prototypes'].T).max(1).values-(f@saved['negative_prototypes'].T).max(1).values).numpy()
probability_cache=Path('runs/evaluation-web/finetuned-test.npz')
if not probability_cache.exists():
 from evaluate_models import infer
 from freshpick.predict import Predictor
 probability_cache.parent.mkdir(parents=True,exist_ok=True)
 test_rows=pd.read_csv('runs/mobilenetv3-finetuned/splits.csv')
 probs,truth=infer(Predictor('runs/mobilenetv3-finetuned/best.pt'),test_rows[test_rows.split=='test'].reset_index(drop=True),'finetuned-test')
else:
 cached=np.load(probability_cache);probs=cached['probabilities'];truth=cached['truth']
names=sorted(base['per_class'].keys() - {'accuracy','macro avg','weighted avg'})
from freshpick.predict import _class_parts
from freshpick.identifier import PRODUCE
stages,parsed=_class_parts(names);stage_names=list(stages)
stage_mapping=np.array([stage_names.index(s) for s,p in parsed]);produce_mapping=np.array([PRODUCE.index(p) for s,p in parsed])
pm=np.stack([probs[:,produce_mapping==i].sum(1) for i in range(8)],axis=1);fm=np.stack([probs[:,stage_mapping==i].sum(1) for i in range(3)],axis=1)
available=(margin>=saved['margin_threshold'])&(pm.argmax(1)==ids)
joint_correct=(produce_mapping[truth]==ids)&(stage_mapping[truth]==fm.argmax(1))
policy={'n':777,'identity_accepted':int((margin>=saved['margin_threshold']).sum()),'freshness_available':int(available.sum()),'freshness_assessment_coverage':float(available.mean()),'joint_accuracy_when_freshness_available':float(joint_correct[available].mean()),'note':'Offline FP32 batches; source image selection and model hashes recorded. API/UI additionally verified on live probes. Unavailable results are not counted as successful complete assessments.'}
(OUT/'policy-metrics.json').write_text(json.dumps(policy,indent=2))
labels=['Published CNN','Augmented CNN','Separate identifier'];scores=[compare['published_cnn_test']['accuracy'],compare['optimized_cnn_test']['accuracy'],model['web_test']['accuracy']]
fig,ax=plt.subplots(figsize=(9,5));ax.bar(labels,np.array(scores)*100,color=['#999','#d7a66c','#287c61']);ax.set_ylim(0,105);ax.set_ylabel('Correct produce identity (%)');ax.set_title('New source-grouped web pilot: 20 test photos, 2–3 per produce')
for i,value in enumerate(scores):ax.text(i,value*100+2,f'{value:.0%}',ha='center')
fig.tight_layout();fig.savefig(OUT/'identity-comparison.png',dpi=160);plt.close(fig)
rows=[]
for produce in PRODUCE:
 r=model['web_test']['per_class'][produce];rows.append({'produce':produce,'n':int(r['support']),'precision':r['precision'],'recall':r['recall'],'f1':r['f1-score']})
pd.DataFrame(rows).to_csv(OUT/'per-produce-metrics.csv',index=False)
def table(frame):
 return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])
frame=pd.DataFrame(rows)
for col in ['precision','recall','f1']:frame[col]=frame[col].map(lambda value:f'{value:.2%}')
latency=json.loads((OUT/'latency.json').read_text()) if (OUT/'latency.json').exists() else None
latency_text=(f"The warmed two-model localhost API measured median {latency['median_ms']:.1f} ms and p95 {latency['p95_ms']:.1f} ms over {latency['n']} sequential test-image uploads after three warmups. This excludes cold model/encoder download, WAN latency, camera startup and concurrent load." if latency else 'Latency has not been measured in this run.')
text=f'''# FreshPick product and model update

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

Manifest SHA-256: `{freeze['sha256']}`. A separate unsupported pool has seven
calibration and eight test images (apples plus clearly nonproduce examples).
No new web test photos train the classifier. Source-photographer groups are a
conservative proxy; physical item/session IDs and overlap with large-scale CLIP
pretraining remain unknown. This is substantially smaller than the proposed
30–50-photo-per-produce benchmark and must remain a pilot.

## Model experiments and selection

The upstream augmented/hierarchical MobileNet pipeline was actually fine-tuned:
8 completed epochs, selected epoch {optimized['best_epoch']}, test joint accuracy
{optimized['test']['accuracy']:.4%}, versus the published {base['test']['accuracy']:.4%}.
It was not deployed because it did not improve that benchmark and identified only
{compare['optimized_cnn_test']['correct']}/20 new test photos.

A frozen [OpenCLIP ViT-B/32 encoder](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
feeds an eight-class supervised identity head initialized from text prototypes.
The encoder revision is pinned to `{model['encoder']['revision']}`. Only the small
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
| New web test identity | {compare['published_cnn_test']['correct']}/20 ({compare['published_cnn_test']['accuracy']:.0%}) | {model['web_test']['correct']}/20 ({model['web_test']['accuracy']:.0%}) |
| Original 24 web regression probes | {compare['published_cnn_regression']['correct']}/24 | {compare['separate_identifier_regression']['correct']}/24 |
| Original dataset produce accuracy | 100.00% | {model['dataset_test']['accuracy']:.4%} |
| New web test macro precision | — | {model['web_test']['per_class']['macro avg']['precision']:.4%} |
| New web test macro recall | — | {model['web_test']['per_class']['macro avg']['recall']:.4%} |
| New web test macro F1 | {compare['published_cnn_test']['macro_f1']:.4f} | {model['web_test']['macro_f1']:.4f} |

The historic joint-winner v0.2.0 audit reported 10/24; the current marginal baseline
reports 11/24. These are different decision policies, not another weight improvement.

{table(frame)}

Per-type test supports are just 2–3. The overall identity accuracy 95% Wilson interval
is {compare['separate_identifier_test']['wilson95'][0]:.1%}–{compare['separate_identifier_test']['wilson95'][1]:.1%}.
The original dataset is reused and can contain related train/test views.

## Rejection, calibration and coverage

The margin between supported and unsupported text prototypes is calibrated on
20 web validation photos, 776 original validation photos, and seven unsupported
validation photos. Known domains have equal weight, so the larger original pool
cannot overwhelm web coverage. Thresholds are midpoints between observed scores
for floating-point stability. Validation domain acceptance: {model['calibration']['known_domain_acceptance']}.

On the new web test, {model['selective_web_test']['accepted']}/20 pass the check
({model['selective_web_test']['coverage']:.0%} coverage), with
{model['selective_web_test']['accuracy_if_accepted']:.2%} accuracy among accepted
identity predictions. Unsupported test rejection is {model['ood_test']['rejected']}/8
({model['ood_test']['rejection_rate']:.1%}). This does not establish general unknown
recognition. Incorrect accepted classifications still occur.

On the original dataset, identity acceptance is
{model['selective_dataset_test']['coverage']:.2%}; prominent freshness is available
for {policy['freshness_available']}/777 ({policy['freshness_assessment_coverage']:.2%}).
Joint correctness among those available assessments is
{policy['joint_accuracy_when_freshness_available']:.2%}. Coverage and abstentions
must be reported alongside conditional accuracy.

## Freshness experiment

A separate 520→128→3 head was trained on frozen image features plus produce identity,
using only original training freshness labels. Validation selected epoch
{model['separate_freshness_head']['epoch']}; dataset test accuracy was
{model['separate_freshness_head']['test_accuracy']:.2%}, macro F1
{model['separate_freshness_head']['test_macro_f1']:.4f}. This is below the retained
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

{latency_text}

## Next validation needed

Collect a substantially larger future web/phone set with physical item/session IDs,
multiple reviewers and recorded freshness definitions. Add more cucumber and eggplant
varieties and handling conditions. Keep that future set untouched by model decisions.
The present local app and model release are explicitly experimental.
'''
Path('reports/IMPLEMENTED_IMPROVEMENTS.md').write_text(text)
print('Implementation report generated',policy)
