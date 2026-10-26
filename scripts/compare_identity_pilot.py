"""Report baseline/optimized CNN identity decisions and the validation-selected head.
No model, threshold, or temperature is selected from these test results.
"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score,f1_score,confusion_matrix
from freshpick.predict import Predictor
from freshpick.identifier import Identifier,PRODUCE
sys.path.insert(0,'scripts')
from evaluate_models import wilson
OUT=Path('reports/identity-pilot')
web=json.loads((OUT/'manifest.json').read_text());test=[r for r in web if r['split']=='test']
regression=[r for r in json.loads(Path('reports/evaluation/web-sources.json').read_text()) if r['produce'] in PRODUCE]
results={};allpred=[]
for name,checkpoint in [('published_cnn','runs/mobilenetv3-finetuned/best.pt'),('optimized_cnn','runs/mobilenetv3-optimized/best.pt')]:
 model=Predictor(checkpoint)
 for split,rows in [('test',test),('regression',regression)]:
  pred=[];truth=[]
  for row in rows:
   with Image.open(row['path']) as image:payload=model.predict(image)
   pred.append(payload['produce']);truth.append(row['produce']);allpred.append({'model':name,'split':split,'id':row['id'],'actual':row['produce'],'predicted':payload['produce'],'accepted':payload['status']=='ok'})
  results[name+'_'+split]={'n':len(rows),'correct':int((np.array(pred)==np.array(truth)).sum()),'accuracy':accuracy_score(truth,pred),
    'macro_f1':f1_score(truth,pred,labels=PRODUCE,average='macro',zero_division=0),
    'per_produce':classification_report(truth,pred,labels=PRODUCE,output_dict=True,zero_division=0)}
 print(name,results[name+'_test']['correct'],len(test),flush=True)
model=Identifier('runs/identifier/best.pt')
for split,rows in [('test',test),('regression',regression)]:
 pred=[];truth=[];accept=[]
 for row in rows:
  with Image.open(row['path']) as image:payload=model.predict(image)
  pred.append(payload['produce']);truth.append(row['produce']);accept.append(payload['status']=='ok');allpred.append({'model':'separate_identifier','split':split,'id':row['id'],'actual':row['produce'],'predicted':payload['produce'],'accepted':payload['status']=='ok'})
 correct=int((np.array(pred)==np.array(truth)).sum());lo,hi=wilson(correct,len(truth))
 results['separate_identifier_'+split]={'n':len(rows),'correct':correct,'accuracy':correct/len(truth),'macro_f1':f1_score(truth,pred,labels=PRODUCE,average='macro',zero_division=0),
  'wilson95':[lo,hi],'accepted':sum(accept),'per_produce':classification_report(truth,pred,labels=PRODUCE,output_dict=True,zero_division=0)}
 pd.DataFrame(confusion_matrix(truth,pred,labels=PRODUCE),index=PRODUCE,columns=PRODUCE).to_csv(OUT/('identifier-'+split+'-confusion.csv'))
print({k:(r['correct'],r['n']) for k,r in results.items()},flush=True)
(OUT/'comparison.json').write_text(json.dumps(results,indent=2));pd.DataFrame(allpred).to_csv(OUT/'comparison-predictions.csv',index=False)
