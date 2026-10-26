"""Download provenance-recorded Commons candidates before running model inference."""
import concurrent.futures
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import requests
from PIL import Image

OUT = Path('runs/evaluation-web')
QUERIES = {
 'banana': 'intitle:banana fruit -bread -cake -flower -illustration -map',
 'bittermelon': 'intitle:"bitter melon" -flower -seed -illustration',
 'cucumber': 'intitle:cucumber -flower -salad -soup -illustration -sandwich',
 'eggplant': 'intitle:eggplant -flower -dish -cooked -illustration',
 'orange': 'intitle:"orange fruit" -juice -illustration',
 'papaya': 'intitle:papaya fruit -flower -seed -illustration',
 'pineapple': 'intitle:pineapple fruit -cake -juice -illustration',
 'tomato': 'intitle:tomato fruit -soup -sauce -illustration',
}
HEADERS = {'User-Agent':'FreshPickEvaluation/0.2 (local research; provenance recorded)'}

def clean(value):
 return ' '.join(html.unescape(re.sub('<[^>]+>', '', str(value))).split())

def fetch(item):
 produce, query = item
 response = requests.get('https://commons.wikimedia.org/w/api.php', params={
  'action':'query','format':'json','generator':'search','gsrsearch':query,
  'gsrnamespace':6,'gsrlimit':12,'prop':'imageinfo','iiprop':'url|extmetadata','iiurlwidth':800,
 }, headers=HEADERS, timeout=40)
 response.raise_for_status()
 pages=sorted(response.json().get('query',{}).get('pages',{}).values(),key=lambda p:p.get('index',999))
 rows=[]
 for page in pages:
  info=page.get('imageinfo',[{}])[0]
  if not info.get('url','').split('?')[0].lower().endswith(('.jpg','.jpeg','.png')): continue
  index=len(rows)+1
  target=OUT/'images'/f'{produce}_{index:02d}.jpg'
  url=info.get('thumburl',info['url'])
  try:
   if not target.exists():
    image=requests.get(url,headers=HEADERS,timeout=40);image.raise_for_status()
    target.write_bytes(image.content)
   with Image.open(target) as image: image.verify()
  except Exception as error:
   print(produce, page['title'], str(error), flush=True)
   target.unlink(missing_ok=True)
   continue
  meta=info.get('extmetadata',{})
  rows.append({'id':target.stem,'produce':produce,'path':str(target),'title':page['title'],
    'source_page':info['descriptionurl'],'download_url':url,
    'description':clean(meta.get('ImageDescription',{}).get('value','')),
    'artist':clean(meta.get('Artist',{}).get('value','')),
    'license':clean(meta.get('LicenseShortName',{}).get('value','')),
    'license_url':meta.get('LicenseUrl',{}).get('value',''),
    'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'freshness_ground_truth':None,'selection':'candidate; not yet visually reviewed'})
  if len(rows)==6: break
 print(produce, len(rows), flush=True)
 return rows

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
 manifest=Path('reports/evaluation/web-sources.json')
 if manifest.exists() and '--discover' not in sys.argv:
  sources=json.loads(manifest.read_text())
  for row in sources:
   target=Path(row['path']);target.parent.mkdir(exist_ok=True,parents=True)
   if not target.exists():
    result=requests.get(row['download_url'],headers=HEADERS,timeout=60);result.raise_for_status()
    target.write_bytes(result.content)
   if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
    raise ValueError(f"Downloaded image changed: {row['id']}")
  (OUT/'selected.json').write_text(json.dumps([r for r in sources if r['produce']!='unsupported_nonproduce'],indent=2))
  (OUT/'negative-selected.json').write_text(json.dumps([r for r in sources if r['produce']=='unsupported_nonproduce'],indent=2))
  print('Verified frozen experiment images:',len(sources))
  raise SystemExit(0)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  rows=[row for group in pool.map(fetch,QUERIES.items()) for row in group]
 (OUT/'candidates.json').write_text(json.dumps(rows,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for group in [list(QUERIES)[:4],list(QUERIES)[4:]]:
  fig,axes=plt.subplots(4,6,figsize=(18,12))
  for rowaxes,produce in zip(axes,group):
   candidates=[r for r in rows if r['produce']==produce]
   for ax,row in zip(rowaxes,candidates):
    with Image.open(row['path']) as image: ax.imshow(image)
    ax.set_title(row['id'],fontsize=10); ax.axis('off')
  fig.tight_layout();fig.savefig(OUT/f'candidates-{group[0]}.jpg',dpi=130);plt.close(fig)
