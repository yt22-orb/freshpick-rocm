"""Reproduce dataset, calibration, degradation and provenance-recorded web audits.
No training, checkpoint selection, or deployment changes are performed.
"""
import argparse
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

from freshpick.data import split_digest
from freshpick.predict import Predictor, image_transform

PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
STAGES = ['fresh','semi_fresh','rotten']
OUT = Path('reports/evaluation')
CACHE = Path('runs/evaluation-web')


def parse(name):
    for stage in ['semi_fresh','fresh','rotten']:
        if name.startswith(stage+'_'): return stage, name[len(stage)+1:]
    raise ValueError(name)


def wilson(correct, total):
    if not total: return [None,None]
    z=1.959963984540054; p=correct/total; den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(mid-half),float(mid+half)]


def calibration(probabilities, truth):
    pred=probabilities.argmax(1); conf=probabilities.max(1); correct=pred==truth
    rows=[]; ece=0.
    for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
        mask=(conf>=lo)&((conf<hi) if hi<.999 else (conf<=1))
        n=int(mask.sum())
        if n:
            acc=float(correct[mask].mean()); avg=float(conf[mask].mean())
            ece+=n/len(truth)*abs(acc-avg)
            rows.append({'lower':float(lo),'upper':float(hi),'count':n,'accuracy':acc,'mean_confidence':avg})
    nll=float(-np.log(np.maximum(probabilities[np.arange(len(truth)),truth],1e-12)).mean())
    onehot=np.eye(probabilities.shape[1])[truth]
    brier=float(((probabilities-onehot)**2).sum(1).mean())
    return {'ece_10_equal_width_bins':ece,'nll':nll,'multiclass_brier_sum':brier,'bins':rows}


class Images(Dataset):
    def __init__(self,rows,variant='clean'):
        self.rows=rows.reset_index(drop=True);self.variant=variant;self.transform=image_transform()
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]
        with Image.open(Path('prepared_dataset')/row.image_path) as source:
            image=source.convert('RGB')
            if self.variant=='dark_50pct': image=ImageEnhance.Brightness(image).enhance(.5)
            elif self.variant=='bright_150pct': image=ImageEnhance.Brightness(image).enhance(1.5)
            elif self.variant=='blur_radius_2': image=image.filter(ImageFilter.GaussianBlur(2))
            elif self.variant=='jpeg_quality_30':
                memory=BytesIO();image.save(memory,format='JPEG',quality=30);memory.seek(0)
                image=Image.open(memory).convert('RGB')
            return self.transform(image),int(row.class_id)


def infer(predictor, rows, cache_name, variant='clean'):
    path=CACHE/(cache_name+'.npz')
    if path.exists():
        result=np.load(path); assert str(result['model_version'])==predictor.version
        return result['probabilities'], result['truth']
    all_probs=[];truth=[]
    loader=DataLoader(Images(rows,variant),batch_size=64,num_workers=4,pin_memory=True)
    with torch.inference_mode():
        for inputs,targets in loader:
            # FP32 matches the deployed Predictor; training's reported test pass used AMP.
            probs=predictor.model(inputs.to('cuda')).softmax(1).cpu().numpy()
            all_probs.append(probs);truth.extend(targets.tolist())
    probabilities=np.concatenate(all_probs);truth=np.array(truth)
    np.savez_compressed(path,probabilities=probabilities,truth=truth,model_version=predictor.version)
    return probabilities,truth


def measure(names, probs, truth, prefix):
    pred=probs.argmax(1);parsed=[parse(name) for name in names]
    produce_ids=np.array([PRODUCE.index(p) for s,p in parsed]);stage_ids=np.array([STAGES.index(s) for s,p in parsed])
    produce_true=produce_ids[truth];produce_pred=produce_ids[pred]
    stage_true=stage_ids[truth];stage_pred=stage_ids[pred]
    stage_probs=np.stack([probs[:,stage_ids==i].sum(1) for i in range(3)],axis=1)
    rows=[]
    for produce_index,produce in enumerate(PRODUCE):
        mask=produce_true==produce_index;n=int(mask.sum());correct=int((truth[mask]==pred[mask]).sum())
        indices=[i for i,x in enumerate(parsed) if x[1]==produce]
        precision,recall,f1,_=precision_recall_fscore_support(truth,pred,labels=indices,average='macro',zero_division=0)
        tp=int(((produce_true==produce_index)&(produce_pred==produce_index)).sum())
        fp=int(((produce_true!=produce_index)&(produce_pred==produce_index)).sum())
        fn=n-tp
        rows.append({'produce':produce,'n':n,'joint_correct':correct,'joint_accuracy':correct/n,
            'joint_accuracy_ci95_low':wilson(correct,n)[0],'joint_accuracy_ci95_high':wilson(correct,n)[1],
            'stage_macro_precision':precision,'stage_macro_recall':recall,'stage_macro_f1':f1,
            'produce_precision':tp/max(tp+fp,1),'produce_recall':tp/n,'produce_f1':2*tp/max(2*tp+fp+fn,1),
            'freshness_accuracy_within_true_produce':float((stage_true[mask]==stage_pred[mask]).mean())})
    per_produce=pd.DataFrame(rows);per_produce.to_csv(OUT/(prefix+'-per-produce.csv'),index=False)
    joint_report=classification_report(truth,pred,labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0)
    pd.DataFrame(joint_report).T.to_csv(OUT/(prefix+'-per-class.csv'))
    fresh_report=classification_report(stage_true,stage_pred,labels=[0,1,2],target_names=STAGES,output_dict=True,zero_division=0)
    pd.DataFrame(fresh_report).T.to_csv(OUT/(prefix+'-freshness.csv'))
    for label,actual,predicted,labels in [('joint',truth,pred,names),('produce',produce_true,produce_pred,PRODUCE),('freshness',stage_true,stage_pred,STAGES)]:
        matrix=confusion_matrix(actual,predicted,labels=list(range(len(labels))))
        pd.DataFrame(matrix,index=labels,columns=labels).to_csv(OUT/(prefix+'-'+label+'-confusion.csv'))
    prediction_rows=pd.DataFrame({'actual_class':[names[i] for i in truth],'predicted_class':[names[i] for i in pred],
        'joint_confidence':probs.max(1),'freshness_confidence':stage_probs[np.arange(len(pred)),stage_pred],
        'joint_correct':truth==pred})
    prediction_rows.to_csv(OUT/(prefix+'-predictions.csv'),index=False)
    thresholds=[]
    for threshold in [.5,.7,.8,.9,.95,.99]:
        mask=probs.max(1)>=threshold
        thresholds.append({'threshold':threshold,'n':int(mask.sum()),'coverage':float(mask.mean()),'joint_accuracy':float((pred[mask]==truth[mask]).mean()) if mask.any() else None})
    correct=int((truth==pred).sum())
    return {'n':len(truth),'joint_correct':correct,'joint_accuracy':correct/len(truth),
        'joint_accuracy_wilson95':wilson(correct,len(truth)), 'macro':joint_report['macro avg'],
        'weighted':joint_report['weighted avg'],
        'produce_accuracy_joint_winner':float((produce_true==produce_pred).mean()),
        'freshness_accuracy_joint_winner':float((stage_true==stage_pred).mean()),
        'freshness_macro':fresh_report['macro avg'],
        'joint_calibration':calibration(probs,truth),
        'freshness_marginal_calibration':calibration(stage_probs,stage_true),
        'marginal_freshness_argmax_disagrees_with_ui':int((stage_probs.argmax(1)!=stage_pred).sum()),
        'rotten_as_fresh':int(((stage_true==2)&(stage_pred==0)).sum()),
        'rotten_as_semi_fresh':int(((stage_true==2)&(stage_pred==1)).sum()),
        'confidence_slices_descriptive_not_validated_thresholds':thresholds}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    test=pd.read_csv('runs/mobilenetv3-finetuned/splits.csv');all_rows=test.copy();test=test[test.split=='test'].reset_index(drop=True)
    val=all_rows[all_rows.split=='val'].reset_index(drop=True)
    results={'inference':'FP32 on ROCm, identical to deployed Predictor; saved training metrics used AMP',
        'split_sha256':split_digest(all_rows),'web_freshness_ground_truth':'unavailable; no web freshness accuracy is computed','models':{}}
    for label,path in [('original','runs/mobilenetv3-small/best.pt'),('finetuned','runs/mobilenetv3-finetuned/best.pt')]:
        predictor=Predictor(path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['split_sha256']==split_digest(all_rows)
        probs,truth=infer(predictor,test,label+'-test')
        results['models'][label]={'version':predictor.version,'checkpoint_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),**measure(predictor.names,probs,truth,label)}
        prediction_path=OUT/(label+'-predictions.csv')
        prediction_table=pd.read_csv(prediction_path)
        prediction_table.insert(0,'sample_id',test.sample_id)
        prediction_table.insert(1,'image_path',test.image_path)
        prediction_table.to_csv(prediction_path,index=False)
        print(label,results['models'][label]['joint_accuracy'],flush=True)
        if label=='finetuned':
            robustness=[]
            for variant in ['clean','dark_50pct','bright_150pct','blur_radius_2','jpeg_quality_30']:
                vp,yt=(probs,truth) if variant=='clean' else infer(predictor,test,label+'-'+variant,variant)
                pred=vp.argmax(1)
                robustness.append({'variant':variant,'n':len(yt),'joint_accuracy':float((pred==yt).mean()),
                    'macro_f1':classification_report(yt,pred,output_dict=True,zero_division=0)['macro avg']['f1-score']})
            pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
            for variant in ['dark_50pct','bright_150pct','blur_radius_2','jpeg_quality_30']:
                vp,yt=infer(predictor,test,label+'-'+variant,variant)
                parsed=[parse(name)[1] for name in predictor.names]
                for row in robustness:
                    if row['variant']==variant:
                        for produce in PRODUCE:
                            mask=np.array([parsed[t]==produce for t in yt]);row[produce]=float((vp.argmax(1)[mask]==yt[mask]).mean())
            pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
            results['robustness']=robustness
        web_path=CACHE/'selected.json'
        if web_path.exists():
            web=[]
            for row in json.loads(web_path.read_text()):
                with Image.open(row['path']) as image:
                    prediction=predictor.predict(image)
                web.append({**row,'model':label,'predicted_produce':prediction['produce'],'predicted_freshness':prediction['freshness'],
                  'class_confidence':prediction['class_confidence'],'freshness_confidence':prediction['freshness_confidence'],
                  'produce_correct':prediction['produce']==row['produce'],'model_version':prediction['model_version']})
            for row in web:
                row.pop('description',None)
            pd.DataFrame(web).to_csv(OUT/(label+'-web-predictions.csv'),index=False)
            (CACHE/(label+'-web-predictions.json')).write_text(json.dumps(web,indent=2))
            results['models'][label]['web']={'n':len(web),'produce_correct':sum(r['produce_correct'] for r in web),
                'produce_accuracy':float(np.mean([r['produce_correct'] for r in web])),
                'freshness_accuracy':None,'note':'Nonrandom convenience sample; unknown freshness labels; not a domain benchmark'}
    from scipy.stats import binomtest
    old=pd.read_csv(OUT/'original-predictions.csv').joint_correct.to_numpy()
    new=pd.read_csv(OUT/'finetuned-predictions.csv').joint_correct.to_numpy()
    wins=int((~old&new).sum());losses=int((old&~new).sum())
    paired={'original_wrong_finetuned_right':wins,'original_right_finetuned_wrong':losses,
        'both_wrong':int((~old&~new).sum()),
        'exact_mcnemar_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.0,
        'note':'Exploratory paired test; correlated images violate independence; test set was reused.'}
    (OUT/'paired-comparison.json').write_text(json.dumps(paired,indent=2))
    (OUT/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
