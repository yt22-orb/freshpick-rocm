"""Build measured Markdown/HTML reports and standalone research figures."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path('reports/evaluation')
summary=json.loads((ROOT/'summary.json').read_text());fine=summary['models']['finetuned'];original=summary['models']['original']
per=pd.read_csv(ROOT/'finetuned-per-produce.csv');web=pd.read_csv(ROOT/'web-per-produce.csv')
webpred=pd.read_csv(ROOT/'finetuned-web-predictions.csv')
lat=json.loads((ROOT/'latency.json').read_text());sim=json.loads((ROOT/'split-similarity.json').read_text())
paired=json.loads((ROOT/'paired-comparison.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})


def pct(value):return f'{float(value)*100:.2f}%'
def label(value):return 'Bitter melon' if value=='bittermelon' else str(value).replace('_',' ').capitalize()
def as_table(frame):
    columns=[str(c).replace('|','/') for c in frame.columns]
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(str(value).replace('|','/') for value in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)

def matrix_plot(path,title,dest):
 frame=pd.read_csv(path,index_col=0);matrix=frame.to_numpy();fig,ax=plt.subplots(figsize=(9,7))
 im=ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(len(frame.columns)),[label(x) for x in frame.columns],rotation=40,ha='right')
 ax.set_yticks(range(len(frame)),[label(x) for x in frame.index]);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(title)
 for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black')
 fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout();fig.savefig(ROOT/dest,dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(11,6));x=np.arange(len(per));bar=per.joint_accuracy.to_numpy()*100
ax.bar(x-.18,bar,.36,label='Dataset: produce + freshness',color='#237b65')
ax.bar(x+.18,web.identity_accuracy.to_numpy()*100,.36,label='Web: produce identity only (n=3 each)',color='#e78937')
ax.set_xticks(x,[label(p) for p in per.produce],rotation=20);ax.set_ylim(0,112);ax.set_ylabel('Correct (%)')
ax.set_title('Strong dataset results do not carry over to these web probes')
ax.legend(loc='upper center',fontsize=9)
for i,(a,b) in enumerate(zip(bar,web.identity_accuracy*100)):
 ax.text(i-.18,a+1,f'{a:.1f}',ha='center',fontsize=8);ax.text(i+.18,b+1,f'{int(round(b/100*3))}/3',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'dataset-vs-web.png',dpi=170);plt.close(fig)
matrix_plot(ROOT/'finetuned-freshness-confusion.csv','Freshness stage: held-out dataset, fine-tuned FP32','freshness-confusion.png')
matrix_plot(ROOT/'finetuned-web-confusion.csv','Web produce identification: 24 fixed convenience images','web-confusion.png')
rob=pd.read_csv(ROOT/'robustness.csv');fig,ax=plt.subplots(figsize=(10,5))
ax.barh([label(v) for v in rob.variant],rob.joint_accuracy*100,color=['#237b65']+['#e78937']*4)
ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Joint produce + freshness accuracy (%)');ax.set_title('Synthetic degradation of the same 777 test images')
for y,a in enumerate(rob.joint_accuracy):ax.text(a*100+1,y,pct(a),va='center')
fig.tight_layout();fig.savefig(ROOT/'robustness.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,5));bins=fine['joint_calibration']['bins']
ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
ax.plot([r['mean_confidence'] for r in bins],[r['accuracy'] for r in bins],'-o',color='#237b65',label='Dataset bins')
for r in bins:ax.annotate(str(r['count']),(r['mean_confidence'],r['accuracy']),fontsize=8)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean joint-class confidence',ylabel='Observed joint accuracy',title='Dataset reliability; labels show bin sample counts');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'calibration.png',dpi=170);plt.close(fig)

# Local contact sheets include attribution in the source appendix; originals stay outside Git.
from PIL import Image
for part in range(2):
 fig,axes=plt.subplots(4,3,figsize=(12,12))
 for ax,(_,row) in zip(axes.ravel(),webpred.iloc[part*12:(part+1)*12].iterrows()):
  with Image.open(row.path) as image:ax.imshow(image)
  ax.axis('off');ax.set_title(f'{label(row.produce)} → {label(row.predicted_produce)}\n{label(row.predicted_freshness)} | joint {row.class_confidence:.1%}',fontsize=9,color='#237b65' if row.produce_correct else '#a43227')
 fig.suptitle('Internet probes: identity labels reviewed; freshness is NOT ground truth',fontsize=12)
 fig.tight_layout();fig.savefig(ROOT/f'web-examples-{part+1}.png',dpi=120);plt.close(fig)

joined=per.merge(web,on='produce',suffixes=('','_web'))
table=pd.DataFrame({'Produce':joined.produce.map(label),'Dataset n':joined.n,'Joint correct':joined.joint_correct.astype(str)+'/'+joined.n.astype(str),
 'Joint accuracy':joined.joint_accuracy.map(pct),'Macro precision':joined.stage_macro_precision.map(pct),'Macro recall':joined.stage_macro_recall.map(pct),'Macro F1':joined.stage_macro_f1.map(pct),
 'Web identity':joined.identity_correct.astype(str)+'/3'})
table.to_csv(ROOT/'report-per-produce.csv',index=False)
overall=pd.DataFrame([
 {'Metric':'24-class accuracy','Original':pct(original['joint_accuracy']),'Fine-tuned':pct(fine['joint_accuracy'])},
 {'Metric':'Macro precision','Original':pct(original['macro']['precision']),'Fine-tuned':pct(fine['macro']['precision'])},
 {'Metric':'Macro recall / balanced accuracy','Original':pct(original['macro']['recall']),'Fine-tuned':pct(fine['macro']['recall'])},
 {'Metric':'Macro F1','Original':f"{original['macro']['f1-score']:.4f}",'Fine-tuned':f"{fine['macro']['f1-score']:.4f}"},
 {'Metric':'Weighted F1','Original':f"{original['weighted']['f1-score']:.4f}",'Fine-tuned':f"{fine['weighted']['f1-score']:.4f}"},
 {'Metric':'Produce identity accuracy (dataset)','Original':pct(original['produce_accuracy_joint_winner']),'Fine-tuned':pct(fine['produce_accuracy_joint_winner'])},
 {'Metric':'Freshness accuracy (dataset)','Original':pct(original['freshness_accuracy_joint_winner']),'Fine-tuned':pct(fine['freshness_accuracy_joint_winner'])},
 {'Metric':'Joint NLL (lower better)','Original':f"{original['joint_calibration']['nll']:.4f}",'Fine-tuned':f"{fine['joint_calibration']['nll']:.4f}"},
 {'Metric':'Joint Brier sum (lower better)','Original':f"{original['joint_calibration']['multiclass_brier_sum']:.4f}",'Fine-tuned':f"{fine['joint_calibration']['multiclass_brier_sum']:.4f}"},
 {'Metric':'ECE (10 equal-width bins)','Original':pct(original['joint_calibration']['ece_10_equal_width_bins']),'Fine-tuned':pct(fine['joint_calibration']['ece_10_equal_width_bins'])},
 {'Metric':'Web identity correct','Original':'10/24 (41.67%)','Fine-tuned':'10/24 (41.67%)'},
])
class_table=pd.read_csv(ROOT/'finetuned-per-class.csv',index_col=0).loc[[n for n in pd.read_csv(ROOT/'finetuned-joint-confusion.csv',index_col=0).index]]
class_table=class_table.reset_index().rename(columns={'index':'Class','precision':'Precision','recall':'Recall','f1-score':'F1','support':'n'})
for col in ['Precision','Recall','F1']:class_table[col]=class_table[col].map(pct)
class_table['n']=class_table.n.astype(int)
fresh=pd.read_csv(ROOT/'finetuned-freshness.csv',index_col=0).loc[['fresh','semi_fresh','rotten']].reset_index().rename(columns={'index':'Stage'})
for col in ['precision','recall','f1-score']:fresh[col]=fresh[col].map(pct)
fresh['support']=fresh.support.astype(int)
ci_table=pd.DataFrame({'Produce':per.produce.map(label),'Dataset accuracy':per.joint_accuracy.map(pct),'95% Wilson interval':per.apply(lambda r:f'{pct(r.joint_accuracy_ci95_low)}–{pct(r.joint_accuracy_ci95_high)}',axis=1),
 'Web identity':web.identity_correct.astype(str)+'/3','Web 95% Wilson interval':web.apply(lambda r:f'{pct(r.wilson95_low)}–{pct(r.wilson95_high)}',axis=1)})
rob_table=rob[['variant','n','joint_accuracy','macro_f1']].copy();rob_table.joint_accuracy=rob_table.joint_accuracy.map(pct);rob_table.macro_f1=rob_table.macro_f1.map(lambda x:f'{x:.4f}')
probes=pd.read_csv(ROOT/'nonproduce-probes.csv');probe_table=probes[['id','predicted_produce','predicted_freshness','class_confidence','freshness_confidence','status']].copy()
for col in ['class_confidence','freshness_confidence']:probe_table[col]=probe_table[col].map(pct)
web_report=pd.read_csv(ROOT/'finetuned-web-classification-report.csv',index_col=0).loc[per.produce]
web_report=web_report.reset_index().rename(columns={'index':'Produce'})
for col in ['precision','recall','f1-score']:web_report[col]=web_report[col].map(pct)
web_report.support=web_report.support.astype(int)

sections=[]
def add(title,text='',frame=None,image=None):sections.append({'title':title,'text':text,'frame':frame,'image':image})
add('FreshPick model evaluation — 26 October 2026',f'''The fine-tuned model performs well on the prepared AgriFreshNET test split, but these experiments do not support calling it reliable on general internet or camera images. It correctly predicted both produce and freshness on {fine['joint_correct']}/777 dataset images ({pct(fine['joint_accuracy'])}); it identified the correct produce in only 10/24 selected internet images (41.67%). Both original and fine-tuned checkpoints achieved the same web identity count. Do not describe the web result as measured freshness accuracy: no verified freshness labels exist for those images.

This audit uses FP32 offline batches of 64 with the deployed preprocessing on the RX 7800 XT. The earlier training report evaluated with AMP and obtained 96.65% accuracy / 0.9626 macro F1. FP32 obtains 96.78% / 0.9657; this is a numerical precision/batching difference with one additional correct prediction net, not another training improvement. The audit batch size differs from single-image serving; the full dataset numbers are batched offline measurements. All 24 web identity predictions were separately verified through the live single-image API. No weights, deployment rules, or thresholds were changed during this audit.''')
add('1. Scope and reproducibility',f'''Compared the original and fine-tuned MobileNetV3-Small checkpoints on the same 777 test images (3,621 training / 776 validation). Evaluated 24 joint classes (eight produce types × three inherited freshness stages), collapsed produce identity, collapsed freshness, confidence calibration, four synthetic corruptions, and the running HTTP API. Same EXIF/RGB/aspect-preserving 224×224 black-padding/ImageNet normalization as the app. No retraining or test-driven model selection was performed.

Checkpoint: {fine['checkpoint_sha256']}. Split digest: {summary['split_sha256']}. Fine-tuned model version: {fine['version']}.

Internet probes: three images per produce, 24 in total. Search candidates were visually reviewed and the selection manifest frozen before inference. Excluded dishes, illustrations, maps, the wrong species (e.g. tamarillo), and ambiguous mixed-produce scenes. Kept cultivar variation, multiple same-type produce, on-plant views, and four cut/open-fruit images as clearly disclosed challenges. Images and identity labels were checked against source metadata and visible content; there was one reviewer, no independent labeling panel, and no freshness ground truth. Internet-image overlap with ImageNet pretraining cannot be excluded. This is a convenience stress probe, not a representative unbiased benchmark.''')
add('2. Overall measured results',frame=overall)
add('3. Per-produce dataset performance and internet probes','Joint accuracy requires BOTH the right produce and freshness. Macro precision/recall/F1 average the three freshness-class metrics for that produce with equal weight; predictions into that produce from elsewhere count as false positives. All 777 produce identities were correct for the fine-tuned dataset pass, so remaining dataset errors are freshness confusions. Do not substitute the 100% produce identity result for freshness accuracy.',table,image='dataset-vs-web.png')
add('4. Uncertainty from small samples','Intervals below are 95% Wilson intervals assuming independent trials. Related views violate that assumption, so even these intervals may be too optimistic. Three web images per produce cannot establish production accuracy. A 3/3 result has a lower bound of about 43.85%; 0/3 does not prove the model always fails.',ci_table)
add('5. Which freshness stages are missed?',f'''Freshness is derived from the winning joint class, as in the UI. There are {fine['rotten_as_fresh']} inherited Rotten test labels predicted Fresh and {fine['rotten_as_semi_fresh']} predicted Semi-fresh (243 Rotten examples total). These errors matter even when overall accuracy looks high. Eggplant is weakest overall: 85/96 joint predictions correct. Semi-fresh eggplant recall is 31/38 = 81.58%. Semi-fresh papaya and rotten cucumber each have only seven test examples; their class recall is unstable. In the offline audit, one test image has a winning-joint freshness stage different from the summed-marginal argmax; reported results follow the actual UI contract, not a substituted decision rule.''',fresh,image='freshness-confusion.png')
add('6. All 24 joint classes',frame=class_table)
add('7. Internet experiment: identity only','Fine-tuned macro precision = 48.96%, macro recall = 41.67%, macro F1 = 0.3798. Pineapple was predicted for 12/24 images, but only three were actual pineapples: its web precision is 25%. Cucumber and papaya had 0/3 correct identities; pineapple had 3/3, which is too small to establish reliability. The internet freshness outputs are predictions to inspect, not correctness labels. Natural ripening, yellowing, dark spots, and open ripe fruit must not automatically be labeled Rotten.',web_report,image='web-confusion.png')
add('8. Inspect the internet predictions: first 12',image='web-examples-1.png')
add('9. Inspect the internet predictions: remaining 12',image='web-examples-2.png')
add('10. Confidence does not transfer to the web',f'''An on-tree papaya was classified as fresh pineapple with 99.92% winning-class confidence and 99.93% summed freshness confidence. Two cucumber photos were classified as pineapple with 89.34% and 88.32% joint confidence. A visually identifiable yellow banana on white background was classified as cucumber. These failures are consistent with domain/background/cultivar sensitivity; they do not prove which feature the network used. No validated feature explanation or causal background attribution was performed.

Dataset joint ECE is {pct(fine['joint_calibration']['ece_10_equal_width_bins'])}, NLL {fine['joint_calibration']['nll']:.4f}, and multiclass Brier sum {fine['joint_calibration']['multiclass_brier_sum']:.4f}. These values only characterize this test distribution. Brier here sums squared errors over all 24 classes. Calibration bins are small outside the high-confidence group, and ECE alone can conceal subgroup failures. No temperature scaling or validated abstention threshold was fitted in this audit. A 90% confidence rule would still accept the papaya failure.''',image='calibration.png')
add('11. Synthetic image robustness','The same 777 prepared 224×224 images were degraded without changing labels: brightness multiplied by 0.5 or 1.5 (PIL clipping applies), Gaussian blur radius 2 pixels, and JPEG quality 30 before normal preprocessing. These are fixed stress settings, not estimates of the frequency or severity of real camera conditions. They demonstrate sensitivity and are not additional independent test samples.',rob_table,image='robustness.png')
add('12. Train/test similarity audit',f'''No exact source-pixel hash overlaps between train and test were found. A 64-bit grayscale dHash nearest-neighbor screen found {sim['test_with_nearest_train_dhash_le_3']}/777 test images within Hamming distance ≤3 of a training image and {sim['test_with_nearest_train_dhash_le_5']}/777 within ≤5. Visual inspection of the eight closest pairs showed very similar backgrounds, poses, and produce appearance, consistent with related captures or near-duplicate views. Item identity cannot be proven without capture IDs, and dHash matches are not a measured leakage rate.

This reinforces the existing split limitation: random image splitting is not the same as testing on new produce items, sessions, cameras, or locations. The local evidence montage is runs/evaluation-web/nearest-train-test.jpg; pair paths and distances are in test-train-nearest-dhash.csv. Do not call a subset filtered by dHash a verified independent test.''')
add('13. Fine-tuning comparison is modest, not a proven domain improvement',f'''On FP32 dataset inference the fine-tuned checkpoint gains seven correct predictions net: {paired['original_wrong_finetuned_right']} original errors become correct while {paired['original_right_finetuned_wrong']} previously correct cases become errors. The exploratory exact McNemar p-value is {paired['exact_mcnemar_p']:.3f}, which does not provide strong evidence of an improvement at 5% under independent paired-trial assumptions; related images and prior test reuse further limit that analysis. Web identity count stays at 10/24. More epochs on the same data are not the main next step.''')
add('14. Unsupported/nonproduce diagnostic','Four visually reviewed nonproduce images (shoes, an oil can, a street, and a clothing shop) were also sent to the running service. All four returned status ok and a supported produce label, as expected from a classifier with no unknown class. This is a diagnostic demonstration, not a measured general out-of-distribution rejection rate. The current service has no rejection gate.',probe_table)
add('15. Serving measurements',f'''All 24 internet-image API predictions matched the offline fine-tuned produce predictions, and the server model version matched the checkpoint hash. HTTP upload/decode/preprocessing/GPU inference/JSON timing after three warmups: median {lat['median_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms, range {lat['min_ms']:.1f}–{lat['max_ms']:.1f} ms over 24 sequential localhost requests. This excludes wide-area network latency, camera startup, concurrent load, and cold model loading. Speed is adequate for the prototype; data generalization is the pressing issue.''')
add('16. What to do next, in priority order','''1. Build an independent camera benchmark before further tuning. Collect all eight produce types across devices, backgrounds, lighting, cultivars, and storage stages; retain item_id, capture_session, camera, site, and date. Assign all views/timepoints of one item to the same split. Keep a future collection/site/device untouched as a final test. Target at least 50 independent items per produce×freshness cell (1,200 item-level observations), then size the final study using the desired interval width; 50 is a pilot target, not a precision guarantee.
2. Improve labels. Write visible-condition criteria per produce and have two reviewers label images independently with disagreement adjudication. Distinguish unripe/ripe/overripe appearance from decay; mark ambiguous/occluded/cut/mixed-item images rather than forcing them into a freshness stage. Add a separate measurement protocol if shelf life is ever a goal.
3. Expand the training distribution. Add real phone photos, different backgrounds and cultivars, off-center objects, clutter, shadows, blur and compression. Prioritize eggplant freshness errors, cucumber/papaya identity failures, yellow/green cultivar variation, and the pineapple overprediction seen here. Add expert-reviewed negative/unsupported images. Keep this audit's web probes out of training until a new untouched benchmark is reserved.
4. Compare an object-crop/segmentation pipeline and a two-head produce+freshness model against the current 24-class baseline on identical grouped splits. A crop could reduce background sensitivity but must be measured, not assumed to solve it. Try brightness/color/blur/JPEG augmentation using training data only; keep realistic severity and verify per-produce recall.
5. Fit uncertainty behavior using validation data. Calibrate scores, evaluate selective accuracy versus coverage and per-produce coverage, and measure rejection on a held-out unsupported-image set. Low softmax confidence is not a verified unknown-item detector. Prevent a definitive assessment on rejected images; show a retake message. Do not deploy an arbitrary threshold based on these 24 probes.
6. Report the right release gates. Track joint macro F1, per-class recall with confidence intervals, especially Rotten recall and Rotten→Fresh mistakes; separately report produce identity, freshness, calibration, domain/slice robustness, and latency. Set acceptable errors and coverage with the intended users before training. Publish results from the untouched grouped benchmark, not just this reused random split.
7. Keep the current app clearly experimental until those checks pass. Show visible-condition limits, no shelf-life/food-safety claims, and no invented feature explanations. Do not turn confident failures into confident advice. A higher-scoring backbone is a later experiment after the data and evaluation protocol are fixed.''')
add('17. Metric definitions','''Accuracy: fraction of predictions that match the label; joint accuracy requires both produce and freshness. Precision: TP/(TP+FP), answering how often a predicted class is correct. Recall: TP/(TP+FN), answering how many true examples were found. F1: harmonic mean of precision and recall. Support: number of true examples. Macro averages give each class equal weight; weighted averages weight by support. Multiclass balanced accuracy equals macro recall. Confusion matrices use actual classes as rows and predictions as columns. ECE compares confidence with correctness within bins; NLL penalizes assigning low probability to the true class. None of these is a food-safety probability. Web identity metrics do not measure freshness accuracy.''')

md=[];body=[]
for section in sections:
 title=section['title'];md.append('# '+title if not md else '## '+title);body.append('<h1>'+html.escape(title)+'</h1>' if len(body)==0 else '<h2>'+html.escape(title)+'</h2>')
 if section['text']:
  md.append(section['text'])
  body.extend('<p>'+html.escape(paragraph).replace('\n','<br>')+'</p>' for paragraph in section['text'].split('\n\n'))
 if section['frame'] is not None:
  md.append(as_table(section['frame']));body.append('<div class="table-wrap">'+section['frame'].to_html(index=False,escape=True,border=0)+'</div>')
 if section['image']:
  name=section['image'];md.append(f'![{title}](evaluation/{name})')
  data=base64.b64encode((ROOT/name).read_bytes()).decode();body.append(f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{data}"></figure>')

sources=json.loads((ROOT/'web-sources.json').read_text())
md.append('## Source attribution and individual experiment records')
body.append('<h2>Source attribution and individual experiment records</h2><p>Images were downloaded as Commons-provided display-sized versions. They are shown resized in contact sheets with labels added; original authors and source licenses apply. Source pages, download URLs, SHA-256 checksums, context, and identities are in web-sources.json/CSV. No claim of source-author endorsement is made.</p>')
body.append('<table><thead><tr><th>ID / source</th><th>Author</th><th>License</th></tr></thead><tbody>')
for row in sources:
 md.append(f"- [{row['id']}: {row['title']}]({row['source_page']}) — {row['artist']}; {row['license']}.")
 body.append('<tr><td><a href="'+html.escape(row['source_page'],quote=True)+'">'+html.escape(row['id'])+'</a></td><td>'+html.escape(row['artist'])+'</td><td><a href="'+html.escape(row.get('license_url') or row['source_page'],quote=True)+'">'+html.escape(row['license'])+'</a></td></tr>')
body.append('</tbody></table>')
md.append('## Reproduce and inspect\n\nInstall ROCm PyTorch and run `.venv/bin/python -m pip install -e ".[api,evaluation]"`. With the local API running, execute:\n\n```bash\n.venv/bin/python scripts/fetch_web_evaluation.py\n.venv/bin/python scripts/evaluate_models.py\n.venv/bin/python scripts/audit_split_similarity.py\n.venv/bin/python scripts/evaluate_live_service.py\n.venv/bin/python scripts/build_evaluation_report.py\n```\n\nThe downloader uses the frozen source manifest and verifies SHA-256; `--discover` searches for new candidates instead. Detailed CSV/JSON files and charts are in reports/evaluation/. Downloaded images and probability caches stay local in runs/evaluation-web/. The app and checkpoint were not changed by this evaluation.')
Path('reports/MODEL_EVALUATION.md').write_text('\n\n'.join(md)+'\n')
css='''body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#26312f;background:#fff}h1{font-size:32px;color:#174c40}h2{font-size:23px;margin-top:40px;color:#174c40}p{white-space:normal}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;border-bottom:1px solid #d8e4e0;text-align:left;vertical-align:top}th{background:#eaf3ef}a{color:#125b82;overflow-wrap:anywhere}.table-wrap{overflow:auto}img{max-width:100%;max-height:850px;object-fit:contain}figure{margin:20px 0} @page{size:A4;margin:16mm} @media print{body{font-size:10px;margin:0;padding:0}h1{font-size:23px}h2{font-size:17px;break-after:avoid}table{font-size:8px}th,td{padding:4px}figure{break-inside:avoid}img{max-height:235mm}tr{break-inside:avoid}a{color:#125b82}.table-wrap{overflow:visible}}'''
Path('reports/MODEL_EVALUATION.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>FreshPick measured model evaluation</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>')
print('Created Markdown and self-contained HTML reports.')
