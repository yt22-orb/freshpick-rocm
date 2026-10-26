import pytest
from fastapi.testclient import TestClient
from freshpick.api import app
from freshpick.storage import StorageRequest, storage_guidance


def request(**kwargs):
    return StorageRequest(**dict(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no',**kwargs))


@pytest.mark.parametrize('produce', ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato'])
def test_sources_and_review_dates_for_all_supported_produce(produce):
    data=StorageRequest(produce=produce,confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no')
    advice=storage_guidance(data)
    assert advice['status']=='guidance'
    assert advice['sources'] and all(s['url'].startswith('https://') and s['reviewed_on'] for s in advice['sources'])
    assert 'remaining shelf life' in advice['disclaimer']


@pytest.mark.parametrize('overrides',[{'confirmed':False},{'spoilage':'yes'},{'spoilage':'unknown'},{'condition':'cut','location':'counter'},{'ripeness':'unknown'}])
def test_does_not_offer_numeric_window_when_unsafe_or_incomplete(overrides):
    base=dict(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no')
    advice=storage_guidance(StorageRequest(**(base|overrides)))
    assert advice['typical_window_days'] is None


def test_tomato_cut_rule_and_elapsed_days_do_not_reset_window():
    advice=storage_guidance(StorageRequest(produce='tomato',confirmed=True,condition='cut',ripeness='ripe',location='refrigerator',spoilage='no',days_stored=2))
    assert advice['typical_window_days']=={'min':3,'max':3}
    assert any('does not restart' in text for text in advice['suggestions'])


def test_api_rejects_unsupported_items_and_negative_age():
    client=TestClient(app)
    payload=dict(produce='apple',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no')
    assert client.post('/storage-guidance',json=payload).status_code==422
    assert client.post('/storage-guidance',json=payload|{'produce':'banana','days_stored':-1}).status_code==422
    result=client.post('/storage-guidance',json=payload|{'produce':'banana'})
    assert result.status_code==200 and result.json()['typical_window_days']=={'min':3,'max':3}



def test_unclear_photo_condition_never_defaults_to_whole():
    advice=storage_guidance(StorageRequest(produce='banana',confirmed=True,condition='unknown',ripeness='ripe',location='refrigerator',spoilage='no',observation_source='photo_model'))
    assert advice['typical_window_days'] is None
    assert advice['status']=='needs_details'


def test_photo_suggestions_are_not_recorded_as_user_observations():
    advice=storage_guidance(StorageRequest(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no',observation_source='photo_model'))
    assert advice['inputs']['observation_source']=='photo_model'
    assert any('conditional' in text and 'suggestions' in text for text in advice['assumptions'])


@pytest.mark.parametrize('days,state,action,remaining',[  # exact and fractional boundaries
    (0,'within_window','check_before_use',{'min':3.0,'max':3.0}),
    (1,'within_window','check_before_use',{'min':2.0,'max':2.0}),
    (2,'near_limit','use_promptly',{'min':1.0,'max':1.0}),
    (2.5,'near_limit','use_promptly',{'min':.5,'max':.5}),
    (3,'at_limit','discard',{'min':0.0,'max':0.0}),
    (4,'past_limit','discard',{'min':0.0,'max':0.0}),
])
def test_reference_countdown_and_conservative_cutoff(days,state,action,remaining):
    data=StorageRequest(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no',days_stored=days)
    timing=storage_guidance(data)['timing']
    assert timing['state']==state and timing['action']==action
    assert timing['reference_days_left']==remaining
    assert timing['policy']=='conservative_discard_at_reference_maximum'
    assert 'not guaranteed safe time' in timing['caveat']


def test_range_retains_both_endpoints_and_does_not_promise_maximum():
    data=StorageRequest(produce='bittermelon',confirmed=True,condition='whole',ripeness='unknown',location='refrigerator',spoilage='no',days_stored=2)
    timing=storage_guidance(data)['timing']
    assert timing['reference_days_left']=={'min':1.0,'max':3.0}
    assert '1–3 days' in timing['heading']
    timing=storage_guidance(data.model_copy(update={'days_stored':4}))['timing']
    assert timing['state']=='near_limit' and timing['reference_days_left']=={'min':0.0,'max':1.0}


@pytest.mark.parametrize('overrides,state,action',[
    ({'spoilage':'yes'},'discard','discard'),
    ({'spoilage':'unknown'},'inspect_first','inspect'),
    ({'confirmed':False},'unavailable','provide_details'),
    ({'condition':'unknown'},'unavailable','provide_details'),
    ({'days_stored':None},'age_needed','provide_days'),
])
def test_incomplete_details_and_spoilage_override_positive_time(overrides,state,action):
    base=dict(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no',days_stored=0)
    timing=storage_guidance(StorageRequest(**(base|overrides)))['timing']
    assert timing['state']==state and timing['action']==action
    assert timing['reference_days_left'] is None
    assert 'You still have' not in timing['heading']


@pytest.mark.parametrize('days,action',[(0,'refrigerate_now'),(1/24,'refrigerate_now'),(2/24,'discard'),(.1,'discard')])
def test_cut_counter_exposure_never_returns_safe_remaining_time(days,action):
    data=StorageRequest(produce='tomato',confirmed=True,condition='cut',ripeness='ripe',location='counter',spoilage='no',days_stored=days)
    advice=storage_guidance(data)
    assert advice['timing']['action']==action
    assert advice['timing']['reference_days_left'] is None
    assert advice['typical_window_days'] is None
    assert any(source['url'].startswith('https://www.fda.gov/') for source in advice['sources'])


def test_photo_inference_is_an_explicit_timing_assumption():
    advice=storage_guidance(StorageRequest(produce='banana',confirmed=True,condition='whole',ripeness='ripe',location='refrigerator',spoilage='no',days_stored=1,observation_source='photo_model'))
    assert advice['timing']['details_are_photo_suggestions'] is True


@pytest.mark.parametrize('produce', ['banana','bittermelon','cucumber','eggplant','orange','papaya','pineapple','tomato'])
@pytest.mark.parametrize('location', ['counter', 'refrigerator'])
def test_no_signs_observed_still_enforces_every_whole_produce_location_limit(produce, location):
    from freshpick.shelf_life import storage_reference
    row = storage_reference(produce, 'whole', location)
    data = StorageRequest(produce=produce, confirmed=True, condition='whole', ripeness='ripe',
                          location=location, spoilage='no', days_stored=row['max_days'])
    advice = storage_guidance(data)
    assert advice['status'] == 'discard_recommended'
    assert advice['timing']['state'] == 'at_limit'
    assert advice['timing']['action'] == 'discard'
    assert advice['timing']['reference_days_left'] == {'min': 0.0, 'max': 0.0}
    assert 'even when no spoilage signs' in advice['timing']['body']
    assert advice['reference']['id'] == row['id']
    assert advice['dataset_version'] == '2026-10-26.1'


@pytest.mark.parametrize('location,window,remaining', [
    ('counter', {'min':3, 'max':7}, {'min':1.0, 'max':5.0}),
    ('refrigerator', {'min':3, 'max':3}, {'min':1.0, 'max':1.0}),
])
def test_purchase_age_uses_location_specific_banana_reference(location, window, remaining):
    advice = storage_guidance(StorageRequest(produce='banana', confirmed=True, condition='whole',
        ripeness='ripe', location=location, spoilage='no', days_stored=2))
    assert advice['typical_window_days'] == window
    assert advice['timing']['reference_days_left'] == remaining
    assert any('No signs observed does not reset' in item for item in advice['suggestions'])


def test_storage_reference_is_available_without_inference_weights():
    reference = TestClient(app).get('/shelf-life-reference')
    assert reference.status_code == 200
    data = reference.json()
    assert len(data['records']) == 32
    assert len({row['id'] for row in data['records']}) == 32
    for row in data['records']:
        assert row['source_ids'] and row['source_locator'] and row['age_basis']
        assert all(key in data['sources'] for key in row['source_ids'])
        if row['condition'] == 'cut' and row['location'] == 'counter':
            assert row['min_days'] is None and row['max_days'] is None
        else:
            assert 0 < row['min_days'] <= row['max_days']


def test_dataset_csv_and_research_snapshot_provenance():
    import csv
    import hashlib
    import json
    from pathlib import Path
    from freshpick.shelf_life import shelf_life_reference
    root = Path(__file__).resolve().parents[1] / 'datasets' / 'shelf_life'
    rows = list(csv.DictReader((root / 'produce_storage.csv').open()))
    assert {row['id'] for row in rows} == {row['id'] for row in shelf_life_reference()['records']}
    expected = {row['id']: row for row in shelf_life_reference()['records']}
    for row in rows:
        for field in ['min_days', 'max_days']:
            assert (float(row[field]) if row[field] else None) == expected[row['id']][field]
    manifest = json.loads((root / 'foodkeeper_mirror_manifest.json').read_text())
    snapshot = (root / 'foodkeeper_mirror_snapshot.json').read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == manifest['sha256']
    assert len(json.loads(snapshot)['product_data']) == manifest['product_records'] == 661
    assert 'current upstream content not verified' in manifest['upstream_download_status']
