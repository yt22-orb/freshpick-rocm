"""Source-backed household storage guidance; never infer days remaining from pixels."""
from typing import Literal
from decimal import Decimal
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from .shelf_life import shelf_life_reference, storage_reference

PRODUCE = ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato']
REFERENCE = shelf_life_reference()
REVIEWED = REFERENCE['reviewed_on']
SOURCES = REFERENCE['sources']

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


def _window_days(low, high):
    low, high = Decimal(str(low)), Decimal(str(high))
    return _days(low) if low == high else f"{format(low.normalize(), 'f')}–{_days(high)}"


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
                      body=f"You report {_days(elapsed)} stored against a maximum {_days(high)} reference. The app recommends discarding at this cutoff even when no spoilage signs are observed, as a conservative storage policy; time alone does not prove spoilage.")
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
              'sources':[], 'inputs':request.model_dump(), 'dataset_version':REFERENCE['dataset_version'], 'reference':None}
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
        reference = storage_reference(request.produce, request.condition, request.location)
        result['reference'] = dict(reference, dataset_version=REFERENCE['dataset_version'])
        for key in reference['source_ids']: source(key)
        if request.location == 'counter': source('fda')
        if request.location == 'counter':
            result['status']='action_required'
            result['suggestions']=['Move cut or peeled produce into a clean covered container in the refrigerator promptly, within two hours of cutting. If it has been left out longer than two hours, discard it.']
            source('canada' if request.produce == 'tomato' else 'maine')
        else:
            days=(reference['min_days'], reference['max_days'])
            result['typical_window_days']={'min':days[0],'max':days[1]}
            result['window_label']=f'{_window_days(*days)} after cutting'
            result['status']='guidance'
            result['assumptions']=['Refrigerator at or below 4°C / 40°F.', 'Prompt chilling within two hours of cutting; no spoilage signs.', 'Window starts at cutting, not at scanning.']
            result['suggestions']=['Use a clean covered container and keep it separate from raw meat, poultry, and fish. Follow an earlier package use-by date if applicable.']
            if request.produce=='tomato': source('canada')
    else:
        reference = storage_reference(request.produce, request.condition, request.location)
        if reference is None:
            result['suggestions']=['No reviewed storage reference matches these details.']
        else:
            result['reference'] = dict(reference, dataset_version=REFERENCE['dataset_version'])
            for key in reference['source_ids']: source(key)
            if reference['required_ripeness'] == 'ripe' and request.ripeness != 'ripe':
                result['suggestions']=['This storage window applies only to ripe whole fruit. Confirm or adjust the ripeness detail first; allow unripe fruit to ripen at room temperature and check daily. No fixed ripening time is predicted.']
            else:
                low, high = reference['min_days'], reference['max_days']
                result['status']='guidance'
                result['typical_window_days']={'min':low,'max':high}
                duration=_window_days(low, high)
                location='in the refrigerator' if request.location=='refrigerator' else 'at room temperature'
                result['window_label']=f'{duration} {location}'
                result['assumptions']=[
                    'Whole produce initially in good condition; unknown prior handling can shorten this window.',
                    REFERENCE['temperature_policy'][request.location],
                    'Compare all reported days since purchase or harvest under these conditions; a new scan or storage-location change does not reset age.',
                ]
                if reference['required_ripeness']=='ripe':
                    result['assumptions'].append('This range assumes ripe fruit. Subtracting the full purchase age is a conservative comparison, not a measurement of time since ripening.')
                result['suggestions']=list(reference['notes'])
    if request.observation_source == 'photo_model':
        result['assumptions'].insert(0, 'Photo-based details are suggestions, not confirmed maturity or condition. Any general storage window is conditional on those suggestions being correct; edit them if needed.')
    if request.spoilage == 'no':
        result['suggestions'].append('No signs observed does not reset the storage clock. The location-specific reference limit still applies to the reported age.')
    if request.days_stored is not None:
        result['suggestions'].append(f'You report {request.days_stored:g} days already stored. The reference window does not restart today; its remaining interval is a planning comparison, not predicted shelf life.')
    result['timing'] = storage_timing(request, result)
    if result['timing']['action'] == 'discard':
        result['status'] = 'discard_recommended'
        result['suggestions'] = [result['timing']['body'], 'A new scan or storage-location change does not reset elapsed storage time.']
    result['sources']=[dict(SOURCES[key],reviewed_on=REVIEWED) for key in ids]
    return result
