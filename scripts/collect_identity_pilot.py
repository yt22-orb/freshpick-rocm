"""Collect Commons photo candidates for manual review, BEFORE model inference."""
import concurrent.futures, hashlib, html, json, random, re, time, sys
from pathlib import Path
import requests
from PIL import Image
OUT=Path('runs/identity-pilot');(OUT/'images').mkdir(parents=True,exist_ok=True)
HEADERS={'User-Agent':'FreshPickResearch/0.3 (licensed produce-photo benchmark; manual review)'}
CATEGORIES={'banana':'Bananas','bittermelon':'Bitter melon','cucumber':'Cucumbers','eggplant':'Eggplant','orange':'Oranges','papaya':'Papaya','pineapple':'Pineapples','tomato':'Tomatoes'}
EXCLUDE=re.compile(r'\b(sea|art|illustration|drawing|painting|book|advert|label|map|cake|soup|salad|dish|curry|juice|bread|fried|sandwich|icon|logo|tamarillo|wild cucumber|sculpture|seedling|leaf|leaves|flower|herbarium)\b',re.I)
old=json.loads(Path('reports/evaluation/web-sources.json').read_text());excluded={r['title'] for r in old};excluded_hashes={r['sha256'] for r in old}
def get(url,**kwargs):
 for attempt in range(4):
  response=requests.get(url,headers=HEADERS,timeout=35,**kwargs)
  if response.status_code in [502,503]:time.sleep(1+attempt*2);continue
  response.raise_for_status();return response
 response.raise_for_status()
def fetch(item):
 produce,category=item;rows=[]
 data=get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'categorymembers','gcmtitle':'Category:'+category,'gcmtype':'file','gcmlimit':200,'prop':'imageinfo','iiprop':'url|extmetadata','iiurlwidth':960}).json()
 pages=list(data.get('query',{}).get('pages',{}).values());random.Random(73).shuffle(pages)
 for page in pages:
  info=page.get('imageinfo',[{}])[0];title=page['title']
  if title in excluded or EXCLUDE.search(title) or not info.get('url','').split('?')[0].lower().endswith(('.jpg','.jpeg')):continue
  meta=info.get('extmetadata',{});license=meta.get('LicenseShortName',{}).get('value','')
  if license not in ['CC0','Public domain','CC BY 2.0','CC BY 3.0','CC BY 4.0','CC BY-SA 2.0','CC BY-SA 3.0','CC BY-SA 4.0']:continue
  path=OUT/'images'/f'{produce}_{page["pageid"]}.jpg';url=info.get('thumburl',info['url']).split('?')[0]
  try:
   if not path.exists():path.write_bytes(get(url).content)
   with Image.open(path) as im:im.verify()
  except Exception as e:
   print(title,str(e),flush=True);path.unlink(missing_ok=True);continue
  sha=hashlib.sha256(path.read_bytes()).hexdigest()
  if sha in excluded_hashes:continue
  rows.append({'id':path.stem,'produce':produce,'path':str(path),'title':title,'source_page':info['descriptionurl'],'download_url':url,'sha256':sha,
    'artist':' '.join(html.unescape(re.sub('<[^>]+>','',meta.get('Artist',{}).get('value',''))).split()),'license':license,'license_url':meta.get('LicenseUrl',{}).get('value',''),
    'description':' '.join(html.unescape(re.sub('<[^>]+>','',meta.get('ImageDescription',{}).get('value',''))).split())[:300]})
  if len(rows)==20:break
 (OUT/(produce+'-candidates.json')).write_text(json.dumps(rows,indent=2))
 print(produce,len(rows),flush=True);return rows
if __name__=='__main__':
 manifest=Path('reports/identity-pilot/manifest.json')
 if manifest.exists() and '--discover' not in sys.argv:
  rows=json.loads(manifest.read_text())+json.loads(Path('reports/identity-pilot/ood-manifest.json').read_text())
  for row in rows:
   path=Path(row['path']);path.parent.mkdir(parents=True,exist_ok=True)
   if not path.exists():path.write_bytes(get(row['download_url']).content)
   if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source image changed: '+row['id'])
  print('Frozen pilot image hashes verified:',len(rows));raise SystemExit(0)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=[r for group in pool.map(fetch,CATEGORIES.items()) for r in group]
 (OUT/'candidates.json').write_text(json.dumps(rows,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for produce in CATEGORIES:
  selected=[r for r in rows if r['produce']==produce];fig,axes=plt.subplots(4,5,figsize=(15,12))
  for ax,row in zip(axes.ravel(),selected):
   with Image.open(row['path']) as im:ax.imshow(im)
   ax.set_title(row['id'],fontsize=8);ax.axis('off')
  fig.tight_layout();fig.savefig(OUT/(produce+'-review.jpg'),dpi=90);plt.close(fig)
