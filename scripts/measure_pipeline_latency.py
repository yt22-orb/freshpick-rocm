"""Warm sequential localhost timing of the actual two-model serving pipeline."""
import json,time
from pathlib import Path
import numpy as np
import requests

rows=[r for r in json.loads(Path('reports/identity-pilot/manifest.json').read_text()) if r['split']=='test']
for _ in range(3):
 with open(rows[0]['path'],'rb') as image:
  result=requests.post('http://localhost:8000/predict',files={'image':image},timeout=60);result.raise_for_status()
times=[]
for row in rows:
 start=time.perf_counter()
 with open(row['path'],'rb') as image:
  result=requests.post('http://localhost:8000/predict',files={'image':image},timeout=60);result.raise_for_status()
 assert result.json()['identifier_version']
 times.append((time.perf_counter()-start)*1000)
metrics={'n':len(times),'warmup_requests':3,'scope':'Sequential local multipart upload, decode, preprocessing, CNN and CLIP inference, JSON; excludes cold weights/download and WAN/camera/concurrent load',
 'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'min_ms':float(min(times)),'max_ms':float(max(times)),'identifier_version':result.json()['identifier_version']}
Path('reports/identity-pilot/latency.json').write_text(json.dumps(metrics,indent=2));print(metrics)
