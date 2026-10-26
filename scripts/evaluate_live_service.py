"""Verify frozen web probes through the running local API, without changing it."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix

root=Path('runs/evaluation-web');out=Path('reports/evaluation')
selected=json.loads((root/'selected.json').read_text())
more=json.loads((out/'web-sources.json').read_text())
negative_ids=['banana_more_01','banana_more_02','banana_more_03','banana_more_05']
negative=[{**r,'produce':'unsupported_nonproduce','selection':'Visually reviewed nonproduce diagnostic; no abstention implemented'} for r in more if r['id'] in negative_ids]
(root/'negative-selected.json').write_text(json.dumps(negative,indent=2))
expected=json.loads((out/'summary.json').read_text())['models']['finetuned']['version']
# Warm up the same serving process before recording request latencies.
for _ in range(3):
 with open(selected[0]['path'],'rb') as stream:
  response=requests.post('http://localhost:8000/predict',files={'image':stream},timeout=60)
  response.raise_for_status()
rows=[]
for source in selected+negative:
 start=time.perf_counter()
 with open(source['path'],'rb') as stream:
  response=requests.post('http://localhost:8000/predict',files={'image':stream},timeout=60)
 elapsed=(time.perf_counter()-start)*1000
 response.raise_for_status();payload=response.json();assert payload['model_version']==expected
 rows.append({'id':source['id'],'source_page':source['source_page'],'actual_produce':source['produce'],
   'http_status':response.status_code,'latency_ms':elapsed,'response':payload})
(out/'live-api-web-results.json').write_text(json.dumps(rows,indent=2))
times=[r['latency_ms'] for r in rows[:24]]
(out/'latency.json').write_text(json.dumps({'n':len(times),'warmup_requests':3,'scope':'Sequential localhost multipart HTTP, image decode, preprocessing, serialized GPU inference and JSON; no network WAN or camera startup',
  'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'min_ms':float(min(times)),'max_ms':float(max(times))},indent=2))
pd.DataFrame([{'id':r['id'],'actual':r['actual_produce'],'predicted_produce':r['response']['produce'],'predicted_freshness':r['response']['freshness'],
 'class_confidence':r['response']['class_confidence'],'freshness_confidence':r['response']['freshness_confidence'],'status':r['response']['status'],'source_page':r['source_page']} for r in rows[24:]]).to_csv(out/'nonproduce-probes.csv',index=False)
labels=['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
for model in ['original','finetuned']:
 table=pd.read_csv(out/(model+'-web-predictions.csv'))
 actual=table.produce.tolist();predicted=table.predicted_produce.tolist()
 report=classification_report(actual,predicted,labels=labels,target_names=labels,output_dict=True,zero_division=0)
 pd.DataFrame(report).T.to_csv(out/(model+'-web-classification-report.csv'))
 pd.DataFrame(confusion_matrix(actual,predicted,labels=labels),index=labels,columns=labels).to_csv(out/(model+'-web-confusion.csv'))
 if model=='finetuned':
  assert all(a==b['response']['produce'] for a,b in zip(predicted,rows[:24]))
print('All 24 supported-image API predictions matched offline predictions; all 4 nonproduce probes returned ok.')
print(json.dumps(json.loads((out/'latency.json').read_text()),indent=2))

from evaluate_models import wilson
fine=pd.read_csv(out/'finetuned-web-predictions.csv')
per=[]
for produce, group in fine.groupby('produce',sort=False):
 correct=int(group.produce_correct.sum());lo,hi=wilson(correct,len(group))
 per.append({'produce':produce,'n':len(group),'identity_correct':correct,
  'identity_accuracy':correct/len(group),'wilson95_low':lo,'wilson95_high':hi})
pd.DataFrame(per).to_csv(out/'web-per-produce.csv',index=False)
