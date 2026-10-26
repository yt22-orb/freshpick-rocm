"""Freeze manually reviewed labels and photographer/visual groups before training."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from freshpick.identifier import PRODUCE
OUT=Path('reports/identity-pilot');OUT.mkdir(exist_ok=True)
CHOSEN={
'banana':[13562039,158273990,189867100,155277115,10428533,105143521,142420981,117674320,40700294,86185231],
'bittermelon':[18944917,5752291,57172358,91509135,87616305,58506842,63640233,16195164,17835656,64148872,42170619,18027163],
'cucumber':[46421038,136749,26755539,62163191,136208145,72875560,154104159,18129774],
'eggplant':[130444553,32168242,72463229,63680959,50862324,63679488,63536239,32454674,100794235,15538,86445479,19045976],
'orange':[18046857,109903508,141783830,30682443,163915111,57023272,64061227,164959],
'papaya':[161623975,125752280,167234399,167388919,162953501,114131683,83528632,150434941,140172657,147341090],
'pineapple':[11753559,103363162,30028144,29388890,30298502,87431522,197150780,9666872,149018390,11428648],
 'tomato':[120024611,57007180,71303763,52988395,27942302,57001507,115805187,104985353,144792122,63753287]}

def dhash(path):
 with Image.open(path) as im:a=np.asarray(im.convert('L').resize((9,8)))
 return (a[:,1:]>a[:,:-1]).ravel()
if __name__=='__main__':
 rows=[]
 for produce,ids in CHOSEN.items():
  candidates=json.loads(Path(f'runs/identity-pilot/{produce}-candidates.json').read_text())
  for ident in ids:
   row=next(r for r in candidates if r['id']==f'{produce}_{ident}').copy();row.pop('description',None)
   row['review']='Single reviewer visually verified supported produce identity before inference; freshness unlabeled.'
   row['freshness_ground_truth']=None;rows.append(row)
 assert len({r['sha256'] for r in rows})==len(rows)
 n=len(rows);parents=list(range(n))
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 def union(a,b):parents[find(a)]=find(b)
 hashes=np.stack([dhash(r['path']) for r in rows])
 for i in range(n):
  for j in range(i):
   same_artist=rows[i]['artist'].strip().casefold()==rows[j]['artist'].strip().casefold()
   if same_artist or (hashes[i]!=hashes[j]).sum()<=5:union(i,j)
 groups=np.array([find(i) for i in range(n)]);labels=np.array([PRODUCE.index(r['produce']) for r in rows])
 best=None
 for seed in range(1,501):
  splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed)
  folds=np.zeros(n,dtype=int)
  for fold,(_,idx) in enumerate(splitter.split(np.zeros(n),labels,groups)):folds[idx]=fold
  names=np.where(folds==0,'test',np.where(folds==1,'val','train'))
  counts=pd.crosstab(labels,names).reindex(index=range(8),columns=['train','val','test'],fill_value=0).fillna(0)
  if (counts[['train','val','test']]>=2).all().all():
   score=float(((counts.values-np.array([5,2.5,2.5]))**2).sum())
   if best is None or score<best[0]:best=(score,seed,names,counts)
 if best is None:raise RuntimeError('Not enough independent groups to cover every split')
 for i,row in enumerate(rows):row['group_id']=f'photographer-or-visual-{find(i)}';row['split']=str(best[2][i])
 (OUT/'manifest.json').write_text(json.dumps(rows,indent=2));pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False)
 digest=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
 (OUT/'freeze.json').write_text(json.dumps({'sha256':digest,'seed':best[1],'images':n,'groups':len(set(groups)),
  'selection':'Frozen before feature extraction, model fitting, calibration or evaluation.',
  'split_rule':'No photographer or dHash<=5 connected component crosses splits. Identity labels only; source photograph/pretraining contamination cannot be excluded.',
  'counts':best[3].to_dict()},indent=2))
 print(best[3]);print('Manifest',digest)
