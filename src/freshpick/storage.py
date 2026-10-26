"""Source-backed household storage guidance; never infer days remaining from pixels."""
from typing import Literal
from pydantic import BaseModel, Field

PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
REVIEWED = '2026-10-26'
SOURCES = {
    'vce': {'title':'Virginia Cooperative Extension — Food Storage Guidelines for Consumers', 'url':'https://www.pubs.ext.vt.edu/348/348-960/348-960.html'},
    'clemson': {'title':'Clemson Extension — Selecting & Storing Fruits & Vegetables', 'url':'https://hgic.clemson.edu/factsheet/selecting-storing-fruits-vegetables/'},
    'cpma': {'title':'Canadian Produce Marketing Association — Bitter Melon', 'url':'https://www.halfyourplate.ca/produce/veggies/bitter-melon/'},
    'maine': {'title':'University of Maine Extension — Storing and Washing Fresh Fruits and Vegetables', 'url':'https://extension.umaine.edu/food-health/2025/05/29/storing-and-washing-fresh-fruits-and-vegetables/'},
    'canada': {'title':'Health Canada — Food safety tips for tomatoes', 'url':'https://www.canada.ca/en/health-canada/services/food-safety-fruits-vegetables/tomatoes.html'},
    'psu': {'title':'Penn State Extension — Proper Care and Handling of Fruits and Vegetables', 'url':'https://extension.psu.edu/proper-care-and-handling-of-fruits-and-vegetables'},
    'maryland': {'title':'University of Maryland Extension — Storing Garden Fruits and Vegetables', 'url':'https://www.extension.umd.edu/resource/storing-garden-fruits-and-vegetables'},
}
# Household quality/storage windows from the linked sources, not publisher day labels.
WHOLE_REFRIGERATED = {
    'banana': (3,3,'vce','Ripen on the counter first; chill ripe bananas if needed.'),
    'bittermelon': (3,5,'cpma','Keep whole in a loose bag; cut shortly before preparation.'),
    'cucumber': (4,6,'vce','Keep refrigerated and inspect for soft or damaged areas before preparation.'),
    'eggplant': (4,7,'vce','Use promptly; avoid bruising and check its condition before preparation.'),
    'orange': (10,21,'vce','Keep whole citrus in the refrigerator for this general storage window.'),
    'papaya': (1,2,'clemson','Use the short refrigerated window after the fruit is ripe.'),
    'pineapple': (5,7,'vce','Use this whole-fruit refrigerator range as general guidance, not a ripening countdown.'),
}

class StorageRequest(BaseModel):
    produce: Literal['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
    confirmed: bool
    condition: Literal['whole','cut']
    ripeness: Literal['unripe','ripe','unknown']
    location: Literal['counter','refrigerator']
    spoilage: Literal['no','yes','unknown']
    days_stored: float | None = Field(default=None, ge=0, le=365)


def storage_guidance(request: StorageRequest):
    result = {'status':'needs_details', 'produce':request.produce, 'typical_window_days':None,
              'window_label':'No numerical window for these conditions', 'title':'Typical storage guidance',
              'assumptions':[], 'suggestions':[], 'reviewed_on':REVIEWED,
              'disclaimer':'General household guidance, not a prediction of this item’s remaining shelf life or a food-safety assessment.',
              'sources':[], 'inputs':request.model_dump()}
    ids=[]
    def source(key):
        if key not in ids: ids.append(key)
    if not request.confirmed:
        result['suggestions']=['Confirm the produce yourself before requesting item-specific guidance.']
    elif request.spoilage != 'no':
        result['status']='inspect_first'
        result['suggestions']=['Check for mold, unusual odor, leakage, or major softening. If spoilage is present, discard the item; do not use a storage range to justify keeping it.']
        source('maryland')
    elif request.condition == 'cut':
        source('maine'); source('psu')
        if request.location == 'counter':
            result['status']='action_required'
            result['suggestions']=['Move cut or peeled produce into a clean covered container in the refrigerator promptly, within two hours of cutting. If it has been left out longer than two hours, discard it.']
            source('canada' if request.produce == 'tomato' else 'maine')
        else:
            days=(3,3) if request.produce=='tomato' else (3,4)
            result['typical_window_days']={'min':days[0],'max':days[1]}
            result['window_label']='3 days after cutting' if days[0]==days[1] else '3–4 days after cutting'
            result['status']='guidance'
            result['assumptions']=['Refrigerator at or below 4°C / 40°F.', 'Prompt chilling within two hours of cutting; no spoilage signs.', 'Window starts at cutting, not at scanning.']
            result['suggestions']=['Use a clean covered container and keep it separate from raw meat, poultry, and fish. Follow an earlier package use-by date if applicable.']
            if request.produce=='tomato': source('canada')
    elif request.produce in ['banana','papaya','tomato'] and request.ripeness != 'ripe':
        result['suggestions']=['Check ripeness yourself. For whole unripe fruit, allow ripening at room temperature away from direct sun; then reassess storage. No fixed ripening time is estimated.']
        source('canada' if request.produce=='tomato' else 'clemson')
    elif request.location == 'counter':
        source('vce')
        if request.produce=='orange':
            result['typical_window_days']={'min':10,'max':10};result['window_label']='About 10 days at room temperature';result['status']='guidance'
            result['assumptions']=['Whole citrus initially in good condition; ordinary indoor room temperature, not a hot outdoor setting.']
            result['suggestions']=['Inspect regularly; use refrigerated guidance instead for longer storage.']
        else:
            result['status']='action_required'
            result['suggestions']=['Consider refrigerator storage and use the matching guidance there. No sourced counter-storage duration is supplied for this combination.']
    elif request.produce=='tomato':
        source('canada');result['status']='guidance';result['window_label']='Use within a few days once ripe and refrigerated'
        result['suggestions']=['Refrigerate ripe whole tomatoes; keep them away from raw meat, poultry, and seafood. The source gives no precise day range for this whole-fruit condition.']
    else:
        low,high,key,action=WHOLE_REFRIGERATED[request.produce];source(key)
        result['status']='guidance';result['typical_window_days']={'min':low,'max':high}
        result['window_label']=f'{low} days in the refrigerator' if low==high else f'{low}–{high} days in the refrigerator'
        result['assumptions']=['Whole produce initially in good condition.', 'Household refrigerator around 1–4°C / 33–40°F; check condition regularly.', 'Unknown purchase age, cultivar and prior handling can shorten this typical window.']
        result['suggestions']=[action]
    if request.days_stored is not None:
        result['suggestions'].append(f'You report {request.days_stored:g} days already stored. This range does not restart today and is not a remaining-days calculation.')
    if request.days_stored is not None and result['typical_window_days'] and request.days_stored >= result['typical_window_days']['max']:
        result['suggestions'].append('The reported storage time is already at or beyond the quoted typical window. Do not treat this as permission to keep it longer; reassess condition and discard if spoiled.')
    result['sources']=[dict(SOURCES[key],reviewed_on=REVIEWED) for key in ids]
    return result
