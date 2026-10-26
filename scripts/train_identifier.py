"""Frozen CLIP encoder, validation-selected eight-class head and freshness-head experiment.
The fresh web TEST split is read for final metrics only, after all selections.
"""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps
from sklearn.metrics import classification_report,accuracy_score,f1_score
from torch.utils.data import DataLoader,Dataset
from scipy.optimize import minimize_scalar
from freshpick.identifier import PRODUCE,create_encoder,text_prototypes,calibrate_margin,ENCODER_REVISION

OUT=Path('runs/identifier');REPORT=Path('reports/identity-pilot')
class Images(Dataset):
 def __init__(self,rows,transform):self.rows=rows;self.transform=transform
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  with Image.open(self.rows[i]['path']) as image:return self.transform(ImageOps.exif_transpose(image).convert('RGB'))

def encode(model,transform,rows,name):
 path=OUT/(name+'.npz');digest=hashlib.sha256(json.dumps([(r['path'],r.get('sha256','')) for r in rows]).encode()).hexdigest()
 if path.exists():
  cached=np.load(path)
  if str(cached['input_digest'])==digest:return torch.tensor(cached['features'],device='cuda')
 results=[]
 with torch.inference_mode():
  for index,inputs in enumerate(DataLoader(Images(rows,transform),batch_size=32,num_workers=4,pin_memory=True)):
   features=model.encode_image(inputs.to('cuda')).float();features=features/features.norm(dim=-1,keepdim=True);results.append(features.cpu())
   if index%30==0:print(name,index,flush=True)
 features=torch.cat(results);np.savez_compressed(path,features=features.numpy(),input_digest=digest)
 return features.to('cuda')

def metrics(actual,pred):
 return {'n':len(actual),'correct':int((np.array(actual)==np.array(pred)).sum()),'accuracy':accuracy_score(actual,pred),
  'macro_f1':f1_score(actual,pred,labels=list(range(8)),average='macro',zero_division=0),
  'per_class':classification_report(actual,pred,labels=list(range(8)),target_names=PRODUCE,output_dict=True,zero_division=0)}

def main():
 torch.manual_seed(43);np.random.seed(43);OUT.mkdir(exist_ok=True)
 web=json.loads((REPORT/'manifest.json').read_text())
 assert len(set(r['group_id'] for r in web if r['split']=='train')&set(r['group_id'] for r in web if r['split']!='train'))==0
 model,transform=create_encoder();prototypes=text_prototypes(model);positive=prototypes[:8];negative=prototypes[8:]
 rows=pd.read_csv('runs/mobilenetv3-finetuned/splits.csv')
 source=[{'path':str(Path('prepared_dataset')/r.image_path),'produce':r.class_name.split('_',2)[-1] if r.class_name.startswith('semi_fresh') else r.class_name.split('_',1)[1],'split':r.split,'class_name':r.class_name} for _,r in rows.iterrows()]
 # Only supervised train/validation features are extracted before candidate selection.
 train=[r for r in source if r['split']=='train'];val=[r for r in source if r['split']=='val']
 webtrain=[r for r in web if r['split']=='train'];webval=[r for r in web if r['split']=='val']
 xt=encode(model,transform,train,'agrifresh-train');xv=encode(model,transform,val,'agrifresh-val')
 wt=encode(model,transform,webtrain,'web-train');wv=encode(model,transform,webval,'web-val')
 ytrain=torch.tensor([PRODUCE.index(r['produce']) for r in train],device='cuda')
 ytweb=torch.tensor([PRODUCE.index(r['produce']) for r in webtrain],device='cuda')
 yval=np.array([PRODUCE.index(r['produce']) for r in val]);yvweb=np.array([PRODUCE.index(r['produce']) for r in webval])
 x=torch.cat([xt,wt]);y=torch.cat([ytrain,ytweb])
 weights=torch.cat([torch.full((len(xt),),.5/len(xt),device='cuda'),torch.full((len(wt),),.5/len(wt),device='cuda')])
 candidates=[]
 def consider(name,weight):
  with torch.no_grad():
   agr=(xv@weight.T).argmax(1).cpu().numpy();wp=(wv@weight.T).argmax(1).cpu().numpy()
  agrf=f1_score(yval,agr,labels=list(range(8)),average='macro',zero_division=0)
  webf=f1_score(yvweb,wp,labels=list(range(8)),average='macro',zero_division=0)
  score=.25*agrf+.75*webf
  candidates.append({'name':name,'validation_score':score,'dataset_validation_macro_f1':agrf,'web_validation_macro_f1':webf,'weight':weight.detach().clone()})
  print('candidate',name,score,webf,flush=True)
 consider('text_prototypes',positive)
 for regularization in [.1,1.0]:
  head=torch.nn.Parameter(positive.detach().clone());optimizer=torch.optim.AdamW([head],lr=.001,weight_decay=0)
  for epoch in range(120):
   optimizer.zero_grad();normalized=head/head.norm(dim=-1,keepdim=True)
   loss=(torch.nn.functional.cross_entropy(100*x@normalized.T,y,reduction='none')*weights).sum()+regularization*((normalized-positive)**2).sum(1).mean()
   loss.backward();optimizer.step()
  consider(f'supervised_identity_reg_{regularization}',head/head.norm(dim=-1,keepdim=True))
 eligible=[c for c in candidates if c['dataset_validation_macro_f1'] >= .98]
 if not eligible:raise RuntimeError('No candidate preserves dataset validation macro F1 >= 0.98; do not replace the baseline')
 selected=max(eligible,key=lambda c:c['validation_score']);weight=selected['weight']
 # Temperature fitted to known web validation only. It calibrates conditional class scores, not food safety.
 validation_logits=(100*wv@weight.T).detach().cpu().numpy()
 def nll(log_t):
  logits=validation_logits/np.exp(log_t);logits-=logits.max(1,keepdims=True)
  return float((-logits[np.arange(len(yvweb)),yvweb]+np.log(np.exp(logits).sum(1))).mean())
 temperature=float(np.exp(minimize_scalar(nll,bounds=(np.log(.25),np.log(10)),method='bounded').x))
 # Rejection uses independent calibration images only; no OOD test samples tune it.
 ood=json.loads((REPORT/'ood-manifest.json').read_text());oodval=[r for r in ood if r['split']=='val']
 ox=encode(model,transform,oodval,'ood-val')
 with torch.no_grad():
  known=((wv@positive.T).max(1).values-(wv@negative.T).max(1).values).cpu().numpy()
  unknown=((ox@positive.T).max(1).values-(ox@negative.T).max(1).values).cpu().numpy()
 with torch.no_grad():
  dataset_known=((xv@positive.T).max(1).values-(xv@negative.T).max(1).values).cpu().numpy()
 calibration=calibrate_margin({'web':known,'agrifresh':dataset_known},unknown)
 threshold=calibration['margin_threshold']
 # Separate supervised freshness head: train only AgriFresh labels, select by its validation macro F1.
 stages=['fresh','semi_fresh','rotten']
 def stage(row):return 'semi_fresh' if row['class_name'].startswith('semi_fresh') else row['class_name'].split('_')[0]
 fy=torch.tensor([stages.index(stage(r)) for r in train],device='cuda');fvy=np.array([stages.index(stage(r)) for r in val])
 fresh=torch.nn.Sequential(torch.nn.Linear(512+8,128),torch.nn.ReLU(),torch.nn.Linear(128,3)).to('cuda')
 fx=torch.cat([xt,torch.nn.functional.one_hot(ytrain,8).float()],1);fv=torch.cat([xv,torch.nn.functional.one_hot(torch.tensor(yval,device='cuda'),8).float()],1)
 optim=torch.optim.AdamW(fresh.parameters(),lr=.005);best=(-1,None,0)
 for epoch in range(1,151):
  fresh.train();optim.zero_grad();loss=torch.nn.functional.cross_entropy(fresh(fx),fy);loss.backward();optim.step()
  fresh.eval()
  with torch.no_grad():pred=fresh(fv).argmax(1).cpu().numpy()
  score=f1_score(fvy,pred,labels=[0,1,2],average='macro',zero_division=0)
  if score>best[0]:best=(score,{k:v.detach().cpu().clone() for k,v in fresh.state_dict().items()},epoch)
 fresh.load_state_dict(best[1]);print('freshness selected',best[0],best[2],flush=True)
 saved={'model_name':'ViT-B-32','pretrained':'laion2b_s34b_b79k','produce_names':PRODUCE,
  'identity_weight':weight.detach().cpu(),'positive_prototypes':positive.detach().cpu(),'negative_prototypes':negative.detach().cpu(),
  'encoder_revision':ENCODER_REVISION,'temperature':temperature,'margin_threshold':threshold,'calibration_scope':calibration,
  'selection':{k:v for k,v in selected.items() if k!='weight'},'freshness_head':best[1],
  'manifest_sha256':hashlib.sha256((REPORT/'manifest.json').read_bytes()).hexdigest()}
 torch.save(saved,OUT/'best.pt')
 # Freeze the chosen head, temperature and threshold BEFORE exposing any held-out test results.
 test=[r for r in source if r['split']=='test'];webtest=[r for r in web if r['split']=='test'];oodtest=[r for r in ood if r['split']=='test']
 tx=encode(model,transform,test,'agrifresh-test');wx=encode(model,transform,webtest,'web-test');ot=encode(model,transform,oodtest,'ood-test')
 ty=np.array([PRODUCE.index(r['produce']) for r in test]);wy=np.array([PRODUCE.index(r['produce']) for r in webtest])
 with torch.no_grad():
  tp=(tx@weight.T).argmax(1).cpu().numpy();wp=(wx@weight.T).argmax(1).cpu().numpy()
  dm=((tx@positive.T).max(1).values-(tx@negative.T).max(1).values).cpu().numpy()
  wm=((wx@positive.T).max(1).values-(wx@negative.T).max(1).values).cpu().numpy()
  om=((ot@positive.T).max(1).values-(ot@negative.T).max(1).values).cpu().numpy()
  tf=torch.cat([tx,torch.nn.functional.one_hot(torch.tensor(tp,device='cuda'),8).float()],1)
  fp=fresh(tf).argmax(1).cpu().numpy()
 ftruth=np.array([stages.index(stage(r)) for r in test]);accepted=wm>=threshold
 result={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'hip':torch.version.hip,'gpu':torch.cuda.get_device_name(0)},
  'encoder':{'model_name':'ViT-B-32','pretrained':'laion2b_s34b_b79k','revision':ENCODER_REVISION,'frozen':True,'pretraining_overlap_with_web':'Unknown; not an independent pretraining benchmark'},
  'validation_guard':'Dataset produce validation macro F1 must be at least 0.98; then select weighted validation score.',
  'selection':saved['selection'],'candidates':[{k:v for k,v in c.items() if k!='weight'} for c in candidates],
  'temperature':temperature,'calibration':calibration,'web_test':metrics(wy,wp),'dataset_test':metrics(ty,tp),
  'selective_dataset_test':{'n':len(dm),'accepted':int((dm>=threshold).sum()),'coverage':float((dm>=threshold).mean())},
  'selective_web_test':{'accepted':int(accepted.sum()),'n':len(accepted),'coverage':float(accepted.mean()),'accuracy_if_accepted':float((wy[accepted]==wp[accepted]).mean()) if accepted.any() else None},
  'ood_test':{'n':len(om),'rejected':int((om<threshold).sum()),'rejection_rate':float((om<threshold).mean())},
  'separate_freshness_head':{'validation_macro_f1':best[0],'epoch':best[2],'test_accuracy':accuracy_score(ftruth,fp),'test_macro_f1':f1_score(ftruth,fp,average='macro'),'web_accuracy':None,'deployment':'Experimental only; web freshness labels unavailable'},
  'checkpoint_sha256':hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest()}
 (REPORT/'experiment.json').write_text(json.dumps(result,indent=2))
 predictions=[]
 for r,yhat,margin in zip(webtest,wp,wm):predictions.append({'id':r['id'],'actual':r['produce'],'predicted':PRODUCE[yhat],'accepted':bool(margin>=threshold),'margin':float(margin),'source_page':r['source_page']})
 pd.DataFrame(predictions).to_csv(REPORT/'test-predictions.csv',index=False)
 print(json.dumps({k:result[k] for k in ['selection','selective_web_test','ood_test','separate_freshness_head']},indent=2))
 print('Web test',result['web_test']['correct'],result['web_test']['n'],'dataset',result['dataset_test']['accuracy'],flush=True)

if __name__=='__main__':main()
