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
