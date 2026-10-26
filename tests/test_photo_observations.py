import json
from freshpick.photo_observations import select_observation, CALIBRATION_PATH, PROMPT_SHA256


def test_uncalibrated_or_low_confidence_observations_remain_unknown():
    assert select_observation([.99,.01],['whole','cut'],None)=='unknown'
    assert select_observation([.55,.45],['whole','cut'],.7)=='unknown'
    assert select_observation([.9,.1],['whole','cut'],.7)=='whole'


def test_spoiled_or_mixed_appearance_is_not_forced_to_ripe():
    assert select_observation([.05,.1,.85],['unripe','ripe','unknown'],.5)=='unknown'


def test_packaged_calibration_matches_the_actual_prompts():
    calibration=json.loads(CALIBRATION_PATH.read_text())
    assert calibration['prompt_sha256']==PROMPT_SHA256
    assert calibration['condition']['validation_n']==20
