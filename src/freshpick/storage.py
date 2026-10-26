"""Source-backed household storage guidance; never infer days remaining from pixels."""
from typing import Literal
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
REVIEWED = '2026-10-26'
SOURCES = {
    'fda': {'title':'FDA — Tips to Reduce Food Waste', 'url':'https://www.fda.gov/food/consumers/tips-reduce-food-waste'},
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
    condition: Literal['whole','cut','unknown']
    ripeness: Literal['unripe','ripe','unknown']
    location: Literal['counter','refrigerator']
    spoilage: Literal['no','yes','unknown']
    observation_source: Literal['user','photo_model','user_edited_photo'] = 'user'
    days_stored: float | None = Field(default=None, ge=0, le=365, allow_inf_nan=False)


def _days(value):
    number = format(value.normalize(), 'f')
    return f"{number} day" + ("" if value == 1 else "s")


def storage_timing(request: StorageRequest, result):
    """Reference-window arithmetic, with an explicit conservative discard policy.

    This never classifies biological spoilage, infers safety, or trains expiry
    targets. Earlier unsafe handling and spoilage override any remaining reference
    interval. A range retains both endpoints instead of promising its maximum.
    """
    timing = {'state':'unavailable','action':'provide_details','heading':'No storage-time estimate available',
              'body':'A matching reference window and storage details are needed.',
              'elapsed_days':request.days_stored,'reference_days_left':None,
              'calculated_at':datetime.now(timezone.utc).isoformat(),
              'limit_days':None,'policy':'reference_window_only',
              'basis':'since_cutting_or_peeling' if request.condition == 'cut' else 'since_purchase_or_harvest',
              'caveat':'This compares your reported age with a typical storage window. It is not guaranteed safe time or the model’s prediction of remaining shelf life.',
              'details_are_photo_suggestions':request.observation_source == 'photo_model'}
    if not request.confirmed:
        timing['heading']='Confirm the produce first'
        return timing
    if request.spoilage == 'yes':
        timing.update(state='discard',action='discard',heading='Discard: reported signs of spoilage',
                      body='Do not use a storage window to override unusual odor, leakage or significant texture changes.',policy='reported_spoilage')
        return timing
    if request.spoilage == 'unknown':
        timing.update(state='inspect_first',action='inspect',heading='Check the item before using storage estimates',
                      body='A photo and storage age cannot establish internal condition or food safety.')
        return timing
    if request.condition == 'cut' and request.location == 'counter':
        if request.days_stored is not None and Decimal(str(request.days_stored)) >= Decimal(str(2/24)):
            timing.update(state='discard',action='discard',heading='Discard: cut-produce handling limit reached',
                          body='You report at least two hours since cutting/peeling under counter-storage conditions. Discard as a conservative handling decision. The limit is shorter in hot conditions.',
                          policy='conservative_two_hour_handling_limit')
        else:
            timing.update(state='action_required',action='refrigerate_now',heading='Refrigerate cut or peeled produce promptly',
                          body='Do not treat time on a counter as safe time remaining. The usual maximum is two hours, or one hour above 32°C / 90°F; discard if these exposure limits have been exceeded.',
                          policy='cut_produce_handling')
        return timing
    window = result['typical_window_days']
    if window is None:
        timing['body']='No numerical reference matches these conditions; the guidance below still applies.'
        return timing
    timing['limit_days']=window['max']
    timing['policy']='conservative_discard_at_reference_maximum'
    if request.days_stored is None:
        timing.update(state='age_needed',action='provide_days',heading='Enter days stored for a storage-window estimate',
                      body='The comparison will use the reference window for the stated condition and storage location.')
        return timing
    elapsed=Decimal(str(request.days_stored));low=Decimal(str(window['min']));high=Decimal(str(window['max']))
    left_low=max(Decimal('0'),low-elapsed);left_high=max(Decimal('0'),high-elapsed)
    timing['reference_days_left']={'min':float(left_low),'max':float(left_high)}
    if elapsed >= high:
        at_limit=elapsed == high
        timing.update(state='at_limit' if at_limit else 'past_limit',action='discard',
                      heading='Storage limit reached — discard conservatively' if at_limit else 'Past the storage limit — discard conservatively',
                      body=f"You report {_days(elapsed)} stored against a maximum {_days(high)} reference. The app recommends discarding at this cutoff as a conservative storage policy; time alone does not prove spoilage.")
    elif left_low == 0 or left_high <= 1:
        timing.update(state='near_limit',action='use_promptly',heading='Near the typical storage limit',
                      body=f"Up to {_days(left_high)} remain in the reference window. Use promptly only if condition and handling are appropriate; the app recommends discarding once its reference maximum is reached.")
    else:
        amount=_days(left_high) if left_low == left_high else f"{format(left_low.normalize(),'f')}–{_days(left_high)}"
        timing.update(state='within_window',action='check_before_use',heading=f"You still have about {amount} within the typical storage window",
                      body='This is reference-window arithmetic under the stated conditions, not a guarantee that the item is safe or will last that long. Check for spoilage before use.')
    return timing


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
    elif request.condition == 'unknown':
        result['suggestions']=['The photo does not clearly show whether this item is whole or cut/peeled. Use the optional details editor if you know; no numerical window is given for an unclear condition.']
    elif request.condition == 'cut':
        source('maine'); source('psu')
        if request.location == 'counter': source('fda')
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
    if request.observation_source == 'photo_model':
        result['assumptions'].insert(0, 'Photo-based details are suggestions, not confirmed maturity or condition. Any general storage window is conditional on those suggestions being correct; edit them if needed.')
    if request.days_stored is not None:
        result['suggestions'].append(f'You report {request.days_stored:g} days already stored. The reference window does not restart today; its remaining interval is a planning comparison, not predicted shelf life.')
    result['timing'] = storage_timing(request, result)
    if result['timing']['action'] == 'discard':
        result['status'] = 'discard_recommended'
        result['suggestions'] = [result['timing']['body'], 'A new scan or storage-location change does not reset elapsed storage time.']
    result['sources']=[dict(SOURCES[key],reviewed_on=REVIEWED) for key in ids]
    return result
