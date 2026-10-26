"""Photo-based form suggestions, separate from freshness and food safety.

These are pilot-tested visual appearance proposals, not measured ripeness,
remaining shelf life, odor, or internal quality. A user may edit each proposal.
"""
import hashlib
import json
from pathlib import Path

import torch

PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
LABELS = {'banana':'banana','bittermelon':'bitter melon','cucumber':'cucumber','eggplant':'eggplant','orange':'orange','papaya':'papaya','pineapple':'pineapple','tomato':'tomato'}
CONDITION_PROMPTS = {
    'whole': ['a photo of whole unpeeled {p} with its skin intact.', 'a photo of an intact {p}, not sliced or peeled.'],
    'cut': ['a photo of sliced, cut open or peeled {p} with exposed flesh.', 'a photo of {p} pieces with the inside visible.'],
}
RIPENESS_PROMPTS = {
    'banana': {
        'unripe':['a photo of an unripe green banana.','a photo of green bananas with unripe skins.'],
        'ripe':['a photo of a ripe yellow banana.','a photo of yellow bananas with a few brown freckles.'],
        'unknown':['a photo of badly decayed, moldy or very black bananas.','a photo of damaged bananas whose ripeness is unclear.'],
    },
    'papaya': {
        'unripe':['a photo of a whole green unripe papaya.','a photo of green unripe papayas on a tree.'],
        'ripe':['a photo of ripe orange or yellow papaya.','a photo of ripe papaya with orange flesh.'],
        'unknown':['a photo of moldy or badly spoiled papaya.','a photo of a mixture of ripe and unripe papayas.'],
    },
    'tomato': {
        'unripe':['a photo of hard green unripe tomatoes.','a photo of unripe green tomato fruit.'],
        'ripe':['a photo of ripe red, yellow or purple tomatoes.','a photo of ripe colorful tomatoes ready for use.'],
        'unknown':['a photo of moldy or badly rotten tomatoes.','a photo of a mixture of ripe and unripe tomatoes.'],
    },
}
PROMPT_SHA256=hashlib.sha256(json.dumps([CONDITION_PROMPTS,RIPENESS_PROMPTS],sort_keys=True).encode()).hexdigest()
CALIBRATION_PATH=Path(__file__).with_name('photo_calibration.json')


def select_observation(probabilities, names, threshold):
    """Only return a proposal if validation supports applying this confidence cutoff."""
    index=max(range(len(probabilities)),key=probabilities.__getitem__)
    if threshold is None or probabilities[index] < threshold:
        return 'unknown'
    return names[index]


class PhotoObserver:
    def __init__(self,model,calibration_path=CALIBRATION_PATH):
        import open_clip
        self.calibration={}
        if Path(calibration_path).is_file():
            saved=json.loads(Path(calibration_path).read_text())
            if saved.get('prompt_sha256')==PROMPT_SHA256:self.calibration=saved
        tokenizer=open_clip.get_tokenizer('ViT-B-32')
        self.weights={}
        with torch.inference_mode():
            for produce in PRODUCE:
                fields={'condition':{key:[p.format(p=LABELS[produce]) for p in values] for key,values in CONDITION_PROMPTS.items()}}
                if produce in RIPENESS_PROMPTS:fields['ripeness']=RIPENESS_PROMPTS[produce]
                self.weights[produce]={}
                for field,groups in fields.items():
                    vectors=[]
                    for prompts in groups.values():
                        vector=model.encode_text(tokenizer(prompts).to('cuda')).float()
                        vector=vector/vector.norm(dim=-1,keepdim=True)
                        mean=vector.mean(0);vectors.append(mean/mean.norm())
                    self.weights[produce][field]=(list(groups),torch.stack(vectors))

    def raw(self,feature,produce):
        return {field:dict(zip(names,(100*feature@weights.T).softmax(-1).flatten().cpu().tolist()))
                for field,(names,weights) in self.weights[produce].items()}

    def observe(self,feature):
        by_produce={}
        for produce in PRODUCE:
            raw=self.raw(feature,produce)
            proposals={}
            for field in ['condition','ripeness']:
                if field not in raw:
                    proposals[field]={'value':'unknown','status':'not_assessed','reason':'Ripeness is not assessed for this produce; green color alone does not imply spoilage.'}
                    continue
                probabilities=raw[field];threshold=self.calibration.get(field,{}).get('threshold')
                value=select_observation(list(probabilities.values()),list(probabilities),threshold)
                proposals[field]={'value':value,'status':'suggested' if value!='unknown' else 'unclear',
                                  'probabilities':probabilities}
            by_produce[produce]=proposals
        return {'by_produce':by_produce,'method':'CLIP visual appearance proposals',
                'version':'photo-appearance-'+PROMPT_SHA256[:12],
                'scope':'Pilot visual suggestions; not verified maturity, freshness, safety or days remaining.',
                'calibration_available':bool(self.calibration)}
