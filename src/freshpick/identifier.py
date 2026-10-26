"""Eight-produce identifier using a frozen, broadly pretrained image encoder.

Identity and freshness remain separate decisions; no storage duration is learned.
"""
import hashlib
from pathlib import Path
import torch
from PIL import ImageOps

ENCODER_REVISION = '1a25a446712ba5ee05982a381eed697ef9b435cf'
PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
DISPLAY = ['banana fruit','bitter melon vegetable (karela or bitter gourd)','cucumber vegetable','eggplant vegetable (aubergine)','orange citrus fruit','papaya fruit','pineapple fruit','tomato fruit']
NEGATIVES = ['an apple','a mango','a watermelon','a potato','a carrot','a bell pepper','a pair of shoes','a street or building','a person','a packaged product','a prepared meal','a flower','a document or map','a pet animal','an empty table']
TEMPLATES = ['a photo of {}.', 'a close-up photo of {}.', 'a photo of fresh whole or sliced {}.']


def create_encoder():
    import open_clip
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download('laion/CLIP-ViT-B-32-laion2B-s34B-b79K', 'open_clip_model.safetensors', revision=ENCODER_REVISION)
    cfg = open_clip.get_pretrained_cfg('ViT-B-32', 'laion2b_s34b_b79k')
    model, _, transform = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained=weights, device='cuda', image_mean=cfg['mean'], image_std=cfg['std'],
        image_interpolation=cfg['interpolation'], image_resize_mode=cfg['resize_mode'])
    model.eval()
    return model, transform


def text_prototypes(model):
    import open_clip
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    with torch.inference_mode():
        features=model.encode_text(tokenizer([t.format(label) for label in DISPLAY+NEGATIVES for t in TEMPLATES]).to('cuda'))
        features=features/features.norm(dim=-1,keepdim=True)
        features=features.reshape(len(DISPLAY+NEGATIVES),len(TEMPLATES),-1).mean(1)
        return (features/features.norm(dim=-1,keepdim=True)).float()


def calibrate_margin(known_by_domain, unknown):
    """Domain-balanced validation only, using midpoint thresholds for numerical stability."""
    import numpy as np
    known_by_domain = {key: np.asarray(values, dtype=float) for key, values in known_by_domain.items()}
    unknown = np.asarray(unknown, dtype=float)
    if not len(unknown) or any(not len(values) for values in known_by_domain.values()):
        raise ValueError('Calibration needs nonempty known domains and unsupported validation samples')
    values = np.unique(np.concatenate([*known_by_domain.values(), unknown]))
    candidates = np.r_[values[0]-1e-5, (values[:-1]+values[1:])/2, values[-1]+1e-5]
    def score(threshold):
        known = np.mean([(domain >= threshold).mean() for domain in known_by_domain.values()])
        rejection = (unknown < threshold).mean()
        return ((known + rejection)/2, known)
    threshold = float(max(candidates, key=score))
    return {'margin_threshold':threshold, 'known_domain_acceptance':{key:float((domain>=threshold).mean()) for key,domain in known_by_domain.items()},
            'known_domain_n':{key:len(domain) for key,domain in known_by_domain.items()},
            'ood_n':len(unknown),'ood_rejection':float((unknown<threshold).mean()),
            'validation_balanced_score':float(score(threshold)[0]),
            'scope':'Small source-grouped Commons pilot plus AgriFresh validation; not verified general OOD recognition.'}


class Identifier:
    def __init__(self, checkpoint):
        self.saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if self.saved['produce_names'] != PRODUCE:
            raise ValueError('Identifier class order mismatch')
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('ROCm and an available AMD GPU are required')
        self.model,self.transform=create_encoder()
        self.weight=self.saved['identity_weight'].to('cuda')
        self.negative=self.saved['negative_prototypes'].to('cuda')
        self.positive=self.saved['positive_prototypes'].to('cuda')
        self.temperature=float(self.saved.get('temperature',1))
        self.threshold=float(self.saved['margin_threshold'])
        from .photo_observations import PhotoObserver
        self.photo_observer = PhotoObserver(self.model)
        self.version='clip-vit-b32-'+hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]

    def predict(self,image):
        with torch.inference_mode():
            feature=self.model.encode_image(self.transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to('cuda')).float()
            feature=feature/feature.norm(dim=-1,keepdim=True)
            logits=100*feature@self.weight.T
            probs=(logits/self.temperature).softmax(1)[0].cpu().tolist()
            margin=float((feature@self.positive.T).max()-(feature@self.negative.T).max())
        index=max(range(8),key=probs.__getitem__)
        accepted=margin>=self.threshold
        observations=self.photo_observer.observe(feature) if accepted else None
        return {'photo_observations':observations,'produce':PRODUCE[index],'produce_confidence':probs[index],
                'produce_probabilities':dict(zip(PRODUCE,probs)), 'identifier_version':self.version,
                'status':'ok' if accepted else 'uncertain','identity_margin':margin,
                'uncertainty_reason':None if accepted else 'The image did not pass the pilot-calibrated supported-produce check. Try one clear supported item.',
                'calibration_scope':self.saved['calibration_scope']}
