"""Fit suggestion cutoffs on frozen web validation labels; report test separately.

Labels describe visible skin/flesh and ripe-looking appearance only, not physical
maturity or safety. Mixed and clearly decayed appearances receive unknown.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from freshpick.identifier import create_encoder
from freshpick.photo_observations import PhotoObserver,PROMPT_SHA256,CALIBRATION_PATH

OUT=Path('reports/photo-observations');OUT.mkdir(exist_ok=True)
CUT={'bittermelon_58506842','cucumber_26755539','cucumber_136208145',
     'orange_109903508','orange_30682443','orange_57023272','orange_164959',
     'papaya_167234399','papaya_167388919','papaya_114131683','pineapple_197150780'}
RIPENESS={
 'banana_13562039':'ripe','banana_158273990':'ripe','banana_189867100':'unripe',
 'banana_155277115':'ripe','banana_10428533':'ripe','banana_105143521':'unripe',
 'banana_142420981':'ripe','banana_117674320':'unripe','banana_40700294':'unknown','banana_86185231':'ripe',
 'papaya_161623975':'unknown','papaya_125752280':'ripe','papaya_167234399':'unknown',
 'papaya_167388919':'ripe','papaya_162953501':'unknown','papaya_114131683':'unknown',
 'papaya_83528632':'unknown','papaya_150434941':'unripe','papaya_140172657':'unripe','papaya_147341090':'unknown',
 'tomato_120024611':'ripe','tomato_57007180':'unknown','tomato_71303763':'ripe',
 'tomato_52988395':'ripe','tomato_27942302':'ripe','tomato_57001507':'ripe',
 'tomato_115805187':'ripe','tomato_104985353':'ripe','tomato_144792122':'ripe','tomato_63753287':'ripe',
}

def fit(rows,minimum=5):
 values=sorted(set(r['confidence'] for r in rows))
 confidences=[0.0]+[(low+high)/2 for low,high in zip(values,values[1:])];best=None
 for threshold in confidences:
  accepted=[r for r in rows if r['predicted']!='unknown' and r['confidence']>=threshold]
  if len(accepted)<minimum:continue
  accuracy=np.mean([r['predicted']==r['actual'] for r in accepted])
  if accuracy < .9:continue
  candidate=(len(accepted),accuracy,-threshold)
  if best is None or candidate>best[0]:best=(candidate,threshold,len(accepted),accuracy)
 return {'threshold':None if best is None else float(best[1]),'validation_n':len(rows),
         'validation_suggestions':0 if best is None else best[2],
         'validation_accuracy_if_suggested':None if best is None else float(best[3]),
         'criterion':'At least 5 validation suggestions with >=90% empirical visual-label precision; maximize coverage. No reliable-population precision guarantee.'}


def summarize(rows,threshold):
 accepted=[r for r in rows if threshold is not None and r['confidence']>=threshold and r['predicted']!='unknown']
 return {'n':len(rows),'suggestions':len(accepted),'coverage':len(accepted)/len(rows) if rows else None,
         'accuracy_if_suggested':float(np.mean([r['actual']==r['predicted'] for r in accepted])) if accepted else None,
         'errors':[r['id'] for r in accepted if r['actual']!=r['predicted']]}

if __name__=='__main__':
 manifest=json.loads(Path('reports/identity-pilot/manifest.json').read_text())
 labels=[{'id':r['id'],'split':r['split'],'condition':'cut' if r['id'] in CUT else 'whole',
          'ripe_appearance':RIPENESS.get(r['id']),
          'review':'Single-reviewer visual appearance, no maturity/freshness/safety ground truth.'} for r in manifest]
 (OUT/'visual-labels.json').write_text(json.dumps(labels,indent=2))
 model,_=create_encoder();observer=PhotoObserver(model,calibration_path=OUT/'not-yet-calibrated.json')
 # No retraining: cached pinned-encoder features for exactly these frozen image IDs.
 by_id={}
 for split in ['val','test']:
  group=[r for r in manifest if r['split']==split]
  features=torch.tensor(np.load(f'runs/identifier/web-{split}.npz')['features'],device='cuda')
  assert len(group)==len(features)
  with torch.inference_mode():
   for row,feature in zip(group,features):by_id[row['id']]=observer.raw(feature.unsqueeze(0),row['produce'])
 records={'condition':[],'ripeness':[]}
 for row,label in zip(manifest,labels):
  if row['split'] not in ['val','test']:continue
  raw=by_id[row['id']]
  for field,actual in [('condition',label['condition']),('ripeness',label['ripe_appearance'])]:
   if actual is None or field not in raw:continue
   predicted=max(raw[field],key=raw[field].get)
   records[field].append({'id':row['id'],'split':row['split'],'actual':actual,'predicted':predicted,'confidence':raw[field][predicted]})
 calibration={'prompt_sha256':PROMPT_SHA256,'scope':'Small source-grouped photo appearance pilot; suggestions, not verified maturity or freshness.',
              'label_sha256':hashlib.sha256((OUT/'visual-labels.json').read_bytes()).hexdigest()}
 for field in records:
  calibration[field]=fit([r for r in records[field] if r['split']=='val'])
 CALIBRATION_PATH.write_text(json.dumps(calibration,indent=2))
 report={'calibration':calibration,'test':{field:summarize([r for r in rows if r['split']=='test'],calibration[field]['threshold']) for field,rows in records.items()},'records':records}
 (OUT/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['calibration'],indent=2));print(report['test'])
